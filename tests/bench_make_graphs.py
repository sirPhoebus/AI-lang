import os
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _get_required_env(name: str) -> str:
    v = os.environ.get(name)
    if v is None or v == "":
        raise RuntimeError(f"Missing {name}")
    return v


def _read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_summary(rows: list[dict]):
    for r in reversed(rows):
        if r.get("event") == "summary":
            return r
    return None


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def main():
    sender_log = _get_required_env("LATLINK_BENCH_SENDER_LOG")
    receiver_log = _get_required_env("LATLINK_BENCH_RECEIVER_LOG")
    baseline_log = _get_required_env("LATLINK_BENCH_BASELINE_LOG")

    shm_sender_log = os.environ.get("LATLINK_BENCH_SHM_SENDER_LOG")
    shm_receiver_log = os.environ.get("LATLINK_BENCH_SHM_RECEIVER_LOG")
    tcp_sender_log = os.environ.get("LATLINK_BENCH_TCP_SENDER_LOG")
    tcp_receiver_log = os.environ.get("LATLINK_BENCH_TCP_RECEIVER_LOG")

    out_png = os.environ.get("LATLINK_BENCH_GRAPH_PNG", "bench_compare.png")

    sender_rows = _read_jsonl(sender_log)
    receiver_rows = _read_jsonl(receiver_log)
    baseline_rows = _read_jsonl(baseline_log)

    s_sum = _extract_summary(sender_rows)
    r_sum = _extract_summary(receiver_rows)
    b_sum = _extract_summary(baseline_rows)

    if s_sum is None or r_sum is None or b_sum is None:
        raise RuntimeError("Missing summary in one or more logs")

    labels = [
        "LatentLink (sender)",
        "LatentLink (receiver)",
        "Baseline",
    ]

    tokens_per_sec = [
        float(s_sum.get("tokens_per_sec", 0.0)),
        float(r_sum.get("tokens_per_sec", 0.0)),
        float(b_sum.get("tokens_per_sec", 0.0)),
    ]

    total_tokens = [
        int(s_sum.get("tokens_generated", 0)),
        int(r_sum.get("tokens_generated", 0)),
        int(b_sum.get("tokens_generated", 0)),
    ]

    bytes_per_sec = [
        float(s_sum.get("bytes_per_sec", 0.0)),
        0.0,
        0.0,
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    axes[0].bar(labels, tokens_per_sec)
    axes[0].set_title("Tokens/sec")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(labels, total_tokens)
    axes[1].set_title("Total tokens")
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(labels, bytes_per_sec)
    axes[2].set_title("Bytes/sec (LatentLink sender)")
    axes[2].tick_params(axis="x", rotation=20)

    # Optional: SHM vs TCP bandwidth
    bw_labels = []
    bw_values = []
    if shm_sender_log and os.path.exists(shm_sender_log):
        shm_rows = _read_jsonl(shm_sender_log)
        shm_sum = _extract_summary(shm_rows)
        if shm_sum is not None:
            bw_labels.append("SHM")
            bw_values.append(float(shm_sum.get("bytes_per_sec", 0.0)))
    if tcp_sender_log and os.path.exists(tcp_sender_log):
        tcp_rows = _read_jsonl(tcp_sender_log)
        tcp_sum = _extract_summary(tcp_rows)
        if tcp_sum is not None:
            bw_labels.append("TCP")
            bw_values.append(float(tcp_sum.get("bytes_per_sec", 0.0)))

    if bw_labels:
        axes[3].bar(bw_labels, bw_values)
        axes[3].set_title("Bandwidth bytes/sec (sender)")
        axes[3].tick_params(axis="x", rotation=0)
    else:
        axes[3].set_title("Bandwidth bytes/sec (sender)")
        axes[3].text(0.5, 0.5, "No SHM/TCP logs found", ha="center", va="center")
        axes[3].set_xticks([])
        axes[3].set_yticks([])

    fig.tight_layout()

    _ensure_dir(out_png)
    fig.savefig(out_png, dpi=200)

    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
