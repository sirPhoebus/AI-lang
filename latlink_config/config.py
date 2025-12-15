import os
import hashlib

class LatentLinkConfig:
    # Transport
    SHM_NAME_PREFIX = os.getenv("LATLINK_SHM_NAME_PREFIX")
    if SHM_NAME_PREFIX is None:
        raise RuntimeError("Missing LATLINK_SHM_NAME_PREFIX")

    # buffer size: default 1GB for safety with large models? Or 512MB?
    # plan says "Payload can be large", layers * seq_len * 2 * hidden * heads
    # For Llama-3-8B context 8k: 32 layers * 8192 * 2 * 4096 * 2 bytes (fp16) approx... huge.
    # Wait, KV cache for 1 token is small. 
    # But "KV_CACHE_DELTA" could be a segment.
    # Shared memory size needs to be configurable.
    SHM_BUFFER_SIZE_BYTES = os.getenv("LATLINK_SHM_BUFFER_SIZE_BYTES")
    if SHM_BUFFER_SIZE_BYTES is None:
        raise RuntimeError("Missing LATLINK_SHM_BUFFER_SIZE_BYTES")
    SHM_BUFFER_SIZE_BYTES = int(SHM_BUFFER_SIZE_BYTES)

    # Network (v2)
    HOST = os.getenv("LATLINK_HOST")
    PORT = os.getenv("LATLINK_PORT")
    if PORT is not None:
        PORT = int(PORT)

    # Protocol
    PROTOCOL_VERSION = os.getenv("LATLINK_PROTOCOL_VERSION")
    if PROTOCOL_VERSION is None:
        raise RuntimeError("Missing LATLINK_PROTOCOL_VERSION")
    PROTOCOL_VERSION = int(PROTOCOL_VERSION)

    # Security
    AUTH_TOKEN = os.getenv("LATLINK_AUTH_TOKEN")
    if AUTH_TOKEN is None:
        raise RuntimeError("Missing LATLINK_AUTH_TOKEN")
    AUTH_TOKEN_HASH = hashlib.sha256(AUTH_TOKEN.encode("utf-8")).hexdigest()
     
    # Timeout
    HANDSHAKE_TIMEOUT_SEC = os.getenv("LATLINK_HANDSHAKE_TIMEOUT_SEC")
    if HANDSHAKE_TIMEOUT_SEC is None:
        raise RuntimeError("Missing LATLINK_HANDSHAKE_TIMEOUT_SEC")
    HANDSHAKE_TIMEOUT_SEC = float(HANDSHAKE_TIMEOUT_SEC)
    
    @classmethod
    def summary(cls):
        return f"LatentLinkConfig(shm={cls.SHM_NAME_PREFIX} size={cls.SHM_BUFFER_SIZE_BYTES} bytes)"
