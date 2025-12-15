import os
import time
import json
import socket
import secrets


def _get_required_env(name: str) -> str:
    v = os.environ.get(name)
    if v is None or v == "":
        raise RuntimeError(f"Missing {name}")
    return v


def _append_log(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf.extend(chunk)
    return bytes(buf)


def main():
    duration_s = float(_get_required_env("LATLINK_BENCH_DURATION_SEC"))
    payload_bytes = int(_get_required_env("LATLINK_BENCH_PAYLOAD_BYTES"))
    host = os.environ.get("LATLINK_BENCH_TCP_HOST", "127.0.0.1")
    port = int(os.environ.get("LATLINK_BENCH_TCP_PORT", "50051"))
    log_path = _get_required_env("LATLINK_BENCH_TCP_SENDER_LOG")

    payload = secrets.token_bytes(payload_bytes)

    print(f"[BenchTcpSender] Connecting to {host}:{port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.connect((host, port))

    start_time = time.time()
    end_time = start_time + duration_s

    sent_msgs = 0
    sent_bytes = 0

    last_print = 0.0
    while time.time() < end_time:
        now = time.time()
        if now - last_print >= 1.0:
            pct = min(100.0, 100.0 * ((now - start_time) / duration_s))
            print(f"[BenchTcpSender] {pct:6.2f}% | msgs={sent_msgs} | bytes={sent_bytes}")
            last_print = now

        sock.sendall(len(payload).to_bytes(4, "big") + payload)
        sent_msgs += 1
        sent_bytes += len(payload)

        # Wait for ack
        _recv_exact(sock, 1)

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
    print("[BenchTcpSender] Done")
    print(summary)

    sock.close()


if __name__ == "__main__":
    main()
