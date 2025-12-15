import sys
import os
import time
import json

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
    payload_bytes_expected = int(_get_required_env("LATLINK_BENCH_PAYLOAD_BYTES"))
    log_path = _get_required_env("LATLINK_BENCH_SHM_RECEIVER_LOG")

    print("[BenchShmReceiver] Starting...")
    session = LatentLinkSession(SessionRole.RECEIVER)
    session.start()

    start_time = time.time()
    end_time = start_time + duration_s

    recv_msgs = 0
    recv_bytes = 0

    last_print = 0.0
    while time.time() < end_time:
        now = time.time()
        if now - last_print >= 1.0:
            pct = min(100.0, 100.0 * ((now - start_time) / duration_s))
            print(f"[BenchShmReceiver] {pct:6.2f}% | msgs={recv_msgs} | bytes={recv_bytes}")
            last_print = now

        hdr = session.shm_in.read_bytes(4)
        if not hdr:
            time.sleep(0.0005)
            continue

        msg_len = int.from_bytes(hdr, "big")
        payload = session.shm_in.read_bytes(msg_len)
        if payload is None:
            time.sleep(0.0005)
            continue

        recv_msgs += 1
        recv_bytes += len(payload)

        # Ack 1 byte
        session.shm_out.write_bytes(b"\x01")

    elapsed = max(1e-9, time.time() - start_time)
    summary = {
        "t": time.time(),
        "side": "receiver",
        "event": "summary",
        "duration_sec": elapsed,
        "payload_bytes": int(payload_bytes_expected),
        "msgs": int(recv_msgs),
        "bytes": int(recv_bytes),
        "msgs_per_sec": float(recv_msgs / elapsed),
        "bytes_per_sec": float(recv_bytes / elapsed),
    }
    _append_log(log_path, summary)
    print("[BenchShmReceiver] Done")
    print(summary)

    session.close()


if __name__ == "__main__":
    main()
