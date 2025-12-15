import sys
import os
import time
import json
import secrets

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from latlink_session.session import LatentLinkSession, SessionRole


def _get_required_env(name: str) -> str:
    v = os.environ.get(name)
    if v is None or v == "":
        raise RuntimeError(f"Missing {name}")
    return v


def _append_log(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    duration_s = float(_get_required_env("LATLINK_BENCH_DURATION_SEC"))
    payload_bytes = int(_get_required_env("LATLINK_BENCH_PAYLOAD_BYTES"))
    log_path = _get_required_env("LATLINK_BENCH_SHM_SENDER_LOG")

    print("[BenchShmSender] Connecting...")
    session = LatentLinkSession(SessionRole.SENDER)
    session.start()

    # No model handshake needed for pure transport benchmark; just a session_id sync.
    start_time = time.time()
    end_time = start_time + duration_s

    sent_msgs = 0
    sent_bytes = 0

    # Fixed payload size
    payload = secrets.token_bytes(payload_bytes)
    header = payload_bytes.to_bytes(4, "big")

    last_print = 0.0
    while time.time() < end_time:
        now = time.time()
        if now - last_print >= 1.0:
            pct = min(100.0, 100.0 * ((now - start_time) / duration_s))
            print(f"[BenchShmSender] {pct:6.2f}% | msgs={sent_msgs} | bytes={sent_bytes}")
            last_print = now

        session.shm_out.write_bytes(header)
        session.shm_out.write_bytes(payload)
        sent_msgs += 1
        sent_bytes += len(payload)

        # Wait for ack to avoid overrunning receiver
        ack = session.shm_in.read_bytes(1)
        if not ack:
            time.sleep(0.0005)
            continue

    elapsed = max(1e-9, time.time() - start_time)
    summary = {
        "t": time.time(),
        "side": "sender",
        "event": "summary",
        "duration_sec": elapsed,
        "payload_bytes": payload_bytes,
        "msgs": int(sent_msgs),
        "bytes": int(sent_bytes),
        "msgs_per_sec": float(sent_msgs / elapsed),
        "bytes_per_sec": float(sent_bytes / elapsed),
    }
    _append_log(log_path, summary)
    print("[BenchShmSender] Done")
    print(summary)

    session.close()


if __name__ == "__main__":
    main()
