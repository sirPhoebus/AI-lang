import os
import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def main():
    model_name = _get_required_env("LATLINK_MODEL_NAME")
    revision = os.environ.get("LATLINK_MODEL_REVISION")

    duration_s = float(_get_required_env("LATLINK_BENCH_DURATION_SEC"))
    tokens_per_turn = int(_get_required_env("LATLINK_BENCH_TOKENS_PER_TURN"))
    log_path = _get_required_env("LATLINK_BENCH_BASELINE_LOG")
    subject_prompt = _get_required_env("LATLINK_BENCH_PROMPT")

    print("[BenchBaseline] Loading model...")
    model, tokenizer = _load_model_and_tokenizer(model_name, revision)

    start_time = time.time()
    end_time = start_time + duration_s
    last_progress_print = 0.0

    convo = subject_prompt
    turn = 0
    total_tokens = 0

    while time.time() < end_time:
        now = time.time()
        if now - last_progress_print >= 1.0:
            elapsed = now - start_time
            pct = min(100.0, 100.0 * (elapsed / duration_s))
            print(f"[BenchBaseline] {pct:6.2f}% elapsed | turns={turn} | tokens={total_tokens}")
            last_progress_print = now

        turn += 1
        inputs = tokenizer(convo, return_tensors="pt")
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=tokens_per_turn,
                do_sample=False,
            )

        new_tokens = gen[0, inputs.input_ids.shape[1] :]
        new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        _append_log(
            log_path,
            {
                "t": time.time(),
                "turn": turn,
                "tokens": int(new_tokens.shape[0]),
                "text": new_text,
            },
        )

        total_tokens += int(new_tokens.shape[0])
        convo = convo + new_text

    elapsed = max(1e-9, time.time() - start_time)
    summary = {
        "t": time.time(),
        "event": "summary",
        "duration_sec": elapsed,
        "tokens_generated": int(total_tokens),
        "tokens_per_sec": float(total_tokens / elapsed),
        "tokens_per_turn": int(tokens_per_turn),
        "turns": int(turn),
    }
    _append_log(log_path, summary)
    print("[BenchBaseline] Done.")
    print(summary)


if __name__ == "__main__":
    main()
