import sys
import os
import time
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from latlink_session.session import LatentLinkSession, SessionRole
from latlink_protocol.protocol import MessageKind, DType, HandshakeData
from latlink_integrations_hf.hf_integration import get_model_signature, inject_kv_cache


def _get_required_env(name: str) -> str:
    v = os.environ.get(name)
    if v is None or v == "":
        raise RuntimeError(f"Missing {name}")
    return v


def _load_model_and_tokenizer(model_name: str, revision: str | None):
    if revision:
        tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
        mdl = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, use_safetensors=True)
    else:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True)
    mdl.eval()
    return mdl, tok


def _append_log(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _decode_tokens(tokenizer, token_ids: torch.Tensor) -> str:
    return tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)


def main():
    model_name = _get_required_env("LATLINK_MODEL_NAME")
    revision = os.environ.get("LATLINK_MODEL_REVISION")

    duration_s = float(_get_required_env("LATLINK_BENCH_DURATION_SEC"))
    tokens_per_turn = int(_get_required_env("LATLINK_BENCH_TOKENS_PER_TURN"))

    receiver_log = _get_required_env("LATLINK_BENCH_RECEIVER_LOG")
    subject_prompt = _get_required_env("LATLINK_BENCH_PROMPT")

    print("[BenchReceiver] Loading model...")
    model, tokenizer = _load_model_and_tokenizer(model_name, revision)

    print("[BenchReceiver] Starting LatentLink receiver...")
    session = LatentLinkSession(SessionRole.RECEIVER)
    session.start()

    print("[BenchReceiver] Waiting for handshake...")
    my_sig = HandshakeData(**get_model_signature(model, tokenizer))
    while True:
        if session.recv_handshake(my_sig):
            break
        time.sleep(0.01)

    print("[BenchReceiver] Handshake OK")

    _append_log(receiver_log, {"t": time.time(), "side": "receiver", "event": "handshake_ok"})

    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    # Start benchmark clock after handshake so sender has the full duration.
    start_time = time.time()
    end_time = start_time + duration_s
    last_progress_print = 0.0

    print("[BenchReceiver] Generating initial turn...")
    inputs = tokenizer(subject_prompt, return_tensors="pt")
    input_ids_full = inputs["input_ids"].to(model_device)

    with torch.no_grad():
        out0 = model(input_ids=input_ids_full, use_cache=True)
        past = out0.past_key_values

    # Greedy token loop for predictable progress
    generated = []
    input_ids = input_ids_full[:, -1:]
    init_last_progress_print = time.time()
    with torch.no_grad():
        for i in range(tokens_per_turn):
            out = model(input_ids=input_ids, past_key_values=past, use_cache=True)
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(int(next_token.item()))
            past = out.past_key_values
            input_ids = next_token

            now_i = time.time()
            if now_i - init_last_progress_print >= 1.0:
                print(f"[BenchReceiver] initial turn progress {i + 1}/{tokens_per_turn}")
                init_last_progress_print = now_i

    new_tokens = torch.tensor(generated, dtype=torch.long, device=model_device)
    receiver_text = _decode_tokens(tokenizer, new_tokens)

    _append_log(
        receiver_log,
        {
            "t": time.time(),
            "side": "receiver",
            "turn": 1,
            "tokens": int(new_tokens.shape[0]),
            "text": receiver_text,
        },
    )

    # Send TEXT_TURN to sender
    print(f"[BenchReceiver] Sending TEXT_TURN turn=1 tokens={int(new_tokens.shape[0])}")
    payload = new_tokens.detach().cpu().numpy().astype(np.int32).tobytes()
    session.send_message(
        kind=MessageKind.TEXT_TURN,
        metadata={"turn": 1, "token_count": int(new_tokens.shape[0]), "dtype": "int32"},
        payload=payload,
    )

    total_tokens = int(new_tokens.shape[0])
    total_bytes_sent = len(payload)
    total_bytes_recv = 0
    turn = 1

    while time.time() < end_time:
        now = time.time()
        if now - last_progress_print >= 1.0:
            elapsed = now - start_time
            pct = min(100.0, 100.0 * (elapsed / duration_s))
            print(f"[BenchReceiver] {pct:6.2f}% elapsed | turns={turn} | tokens={total_tokens} | bytes_sent={total_bytes_sent} | bytes_recv={total_bytes_recv}")
            last_progress_print = now

        # Receive KV teleport from sender
        msg = session.recv_message()
        if msg is None:
            time.sleep(0.001)
            continue
        if msg.kind != MessageKind.KV_CACHE_DELTA:
            continue

        # Reconstruct arrays
        flat_tensors = []
        offset = 0
        for meta in msg.tensors:
            dt = DType.to_numpy(meta.dtype)
            end = offset + meta.byte_length
            raw = msg.payload_bytes[offset:end]
            arr = np.frombuffer(raw, dtype=dt).reshape(meta.shape).copy()
            flat_tensors.append((meta.name, arr))
            offset = end
        if offset != len(msg.payload_bytes):
            raise RuntimeError("Payload byte_length mismatch")

        total_bytes_recv += len(msg.payload_bytes)

        past_key_values = inject_kv_cache(flat_tensors, device=model_device, dtype=model_dtype)
        past = DynamicCache.from_legacy_cache(past_key_values)

        last_token_id = int(msg.metadata["last_token_id"])
        input_ids = torch.tensor([[last_token_id]], dtype=torch.long, device=model_device)

        # Generate receiver turn tokens_per_turn
        generated = []
        with torch.no_grad():
            for _ in range(tokens_per_turn):
                out = model(input_ids=input_ids, past_key_values=past, use_cache=True)
                next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                generated.append(int(next_token.item()))
                past = out.past_key_values
                input_ids = next_token

        gen_tokens = torch.tensor(generated, dtype=torch.long, device=model_device)
        receiver_text = _decode_tokens(tokenizer, gen_tokens)

        turn += 1
        total_tokens += int(gen_tokens.shape[0])

        _append_log(
            receiver_log,
            {
                "t": time.time(),
                "side": "receiver",
                "turn": turn,
                "tokens": int(gen_tokens.shape[0]),
                "text": receiver_text,
            },
        )

        # Send TEXT_TURN to sender
        payload = gen_tokens.detach().cpu().numpy().astype(np.int32).tobytes()
        session.send_message(
            kind=MessageKind.TEXT_TURN,
            metadata={"turn": turn, "token_count": int(gen_tokens.shape[0]), "dtype": "int32"},
            payload=payload,
        )
        total_bytes_sent += len(payload)

    elapsed = max(1e-9, time.time() - start_time)
    summary = {
        "t": time.time(),
        "side": "receiver",
        "event": "summary",
        "duration_sec": elapsed,
        "tokens_generated": int(total_tokens),
        "tokens_per_sec": float(total_tokens / elapsed),
        "bytes_sent": int(total_bytes_sent),
        "bytes_recv_payload": int(total_bytes_recv),
        "tokens_per_turn": int(tokens_per_turn),
    }
    _append_log(receiver_log, summary)

    print("[BenchReceiver] Done.")
    print(summary)

    session.close()


if __name__ == "__main__":
    main()
