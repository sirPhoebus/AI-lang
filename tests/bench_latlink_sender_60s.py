import sys
import os
import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from latlink_session.session import LatentLinkSession, SessionRole
from latlink_protocol.protocol import MessageKind, HandshakeData, LatentLinkMessage, TensorMeta, DType
from latlink_integrations_hf.hf_integration import get_model_signature, capture_kv_cache


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


def _decode_tokens(tokenizer, token_ids) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _pack_kv_message(session_id: str, msg_id: int, last_token_id: int, token_count: int, kv_list) -> LatentLinkMessage:
    tensors_meta = []
    payload = b""

    for name, arr in kv_list:
        meta = TensorMeta(
            name=name,
            dtype=int(DType.from_numpy(arr.dtype)),
            shape=list(arr.shape),
            byte_length=arr.nbytes,
        )
        tensors_meta.append(meta)
        payload += arr.tobytes()

    return LatentLinkMessage(
        session_id=session_id,
        msg_id=msg_id,
        kind=MessageKind.KV_CACHE_DELTA,
        metadata={
            "last_token_id": int(last_token_id),
            "token_count": int(token_count),
            "benchmark": "latlink_chat",
        },
        tensors=tensors_meta,
        payload_bytes=payload,
    )


def main():
    model_name = _get_required_env("LATLINK_MODEL_NAME")
    revision = os.environ.get("LATLINK_MODEL_REVISION")

    duration_s = float(_get_required_env("LATLINK_BENCH_DURATION_SEC"))
    tokens_per_turn = int(_get_required_env("LATLINK_BENCH_TOKENS_PER_TURN"))

    log_path = _get_required_env("LATLINK_BENCH_SENDER_LOG")
    subject_prompt = _get_required_env("LATLINK_BENCH_PROMPT")

    print("[BenchSender] Loading model...")
    model, tokenizer = _load_model_and_tokenizer(model_name, revision)

    print("[BenchSender] Connecting to LatentLink...")
    session = LatentLinkSession(SessionRole.SENDER)
    session.start()

    print("[BenchSender] Sending handshake...")
    sig = get_model_signature(model, tokenizer)
    hs_data = HandshakeData(**sig)
    session.send_handshake(hs_data)

    _append_log(log_path, {"t": time.time(), "side": "sender", "event": "handshake_ok"})

    first_turn_timeout_sec = float(os.environ.get("LATLINK_BENCH_FIRST_TURN_TIMEOUT_SEC", "180"))
    start_time = None
    end_time = None
    last_progress_print = 0.0
    first_turn_deadline = time.time() + first_turn_timeout_sec
    saw_any_message = False

    total_tokens = 0
    total_bytes = 0
    turn = 0

    convo_text = subject_prompt

    while True:
        if start_time is not None and time.time() >= end_time:
            break
        now = time.time()
        if now - last_progress_print >= 1.0:
            if start_time is None:
                wait_left = max(0.0, first_turn_deadline - now)
                print(f"[BenchSender] waiting first TEXT_TURN | turns={turn} | tokens={total_tokens} | bytes_sent={total_bytes} | wait_left_sec={wait_left:0.1f}")
            else:
                elapsed = now - start_time
                pct = min(100.0, 100.0 * (elapsed / duration_s))
                print(f"[BenchSender] {pct:6.2f}% elapsed | turns={turn} | tokens={total_tokens} | bytes_sent={total_bytes}")
            last_progress_print = now

        # 1) Receive receiver turn
        msg = session.recv_message()
        if msg is None:
            time.sleep(0.001)
            if (not saw_any_message) and time.time() > first_turn_deadline:
                raise TimeoutError("Did not receive initial TEXT_TURN from benchmark receiver. Ensure run_bench_latlink_receiver.bat is running and completed its first turn.")
            continue

        saw_any_message = True

        if msg.kind != MessageKind.TEXT_TURN:
            print(f"[BenchSender] Ignoring unexpected message kind={msg.kind}")
            continue

        if start_time is None:
            start_time = time.time()
            end_time = start_time + duration_s

        recv_token_count = int(msg.metadata.get("token_count", 0))
        recv_turn = int(msg.metadata.get("turn", 0))
        recv_ids = torch.frombuffer(msg.payload_bytes, dtype=torch.int32).to(torch.int64)
        recv_text = _decode_tokens(tokenizer, recv_ids.tolist())

        _append_log(
            log_path,
            {
                "t": time.time(),
                "side": "receiver",
                "turn": recv_turn,
                "tokens": recv_token_count,
                "text": recv_text,
            },
        )

        convo_text = convo_text + recv_text

        # 2) Sender turn: generate fixed tokens
        turn = recv_turn + 1
        inputs = tokenizer(convo_text, return_tensors="pt")
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=tokens_per_turn,
                do_sample=False,
            )

        new_tokens = gen[0, inputs.input_ids.shape[1] :]
        new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        total_tokens += int(new_tokens.shape[0])

        _append_log(
            log_path,
            {
                "t": time.time(),
                "side": "sender",
                "turn": turn,
                "tokens": int(new_tokens.shape[0]),
                "text": new_text,
            },
        )

        convo_text = convo_text + new_text

        # 3) Teleport updated KV cache for full conversation so far
        full_for_kv = convo_text
        full_inputs = tokenizer(full_for_kv, return_tensors="pt")
        with torch.no_grad():
            out = model(**full_inputs, use_cache=True)

        kv_list = capture_kv_cache(out.past_key_values)
        last_token_id = int(full_inputs.input_ids[0, -1].item())
        token_count = int(full_inputs.input_ids.shape[1])

        kv_msg = _pack_kv_message(session.session_id, session.msg_counter, last_token_id, token_count, kv_list)
        session.msg_counter += 1
        packed = kv_msg.pack()
        total_bytes += len(packed)
        session.shm_out.write_bytes(packed)

    elapsed = max(1e-9, time.time() - start_time)
    summary = {
        "t": time.time(),
        "side": "sender",
        "event": "summary",
        "duration_sec": elapsed,
        "tokens_generated": int(total_tokens),
        "tokens_per_sec": float(total_tokens / elapsed),
        "bytes_sent": int(total_bytes),
        "bytes_per_sec": float(total_bytes / elapsed),
        "tokens_per_turn": int(tokens_per_turn),
    }
    _append_log(log_path, summary)
    print("[BenchSender] Done.")
    print(summary)

    session.close()


if __name__ == "__main__":
    main()
