import os
import time
import json
import socket


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
    log_path = _get_required_env("LATLINK_BENCH_TCP_RECEIVER_LOG")

    print(f"[BenchTcpReceiver] Listening on {host}:{port}...")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)

    conn, addr = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[BenchTcpReceiver] Connected from {addr}")

    start_time = time.time()
    end_time = start_time + duration_s

    recv_msgs = 0
    recv_bytes = 0

    last_print = 0.0
    # Frame: 4-byte big endian length + payload
    while time.time() < end_time:
        now = time.time()
        if now - last_print >= 1.0:
            pct = min(100.0, 100.0 * ((now - start_time) / duration_s))
            print(f"[BenchTcpReceiver] {pct:6.2f}% | msgs={recv_msgs} | bytes={recv_bytes}")
            last_print = now

        try:
            header = _recv_exact(conn, 4)
        except ConnectionError:
            break
        msg_len = int.from_bytes(header, "big")
        payload = _recv_exact(conn, msg_len)
        if len(payload) != payload_bytes:
            # still count what we got
            pass

        recv_msgs += 1
        recv_bytes += len(payload)

        # Ack 1 byte
        conn.sendall(b"\x01")

    elapsed = max(1e-9, time.time() - start_time)
    summary = {
        "t": time.time(),
        "side": "receiver",
        "event": "summary",
        "duration_sec": elapsed,
        "payload_bytes": payload_bytes,
        "msgs": int(recv_msgs),
        "bytes": int(recv_bytes),
        "msgs_per_sec": float(recv_msgs / elapsed),
        "bytes_per_sec": float(recv_bytes / elapsed),
    }
    _append_log(log_path, summary)
    print("[BenchTcpReceiver] Done")
    print(summary)

    try:
        conn.close()
    finally:
        srv.close()


if __name__ == "__main__":
    main()
