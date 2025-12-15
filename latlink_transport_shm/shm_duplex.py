import time
from dataclasses import dataclass

from latlink_transport.transport import DuplexByteChannels, TransportRole
from latlink_transport_shm.shm_transport import ShmRingBuffer


@dataclass
class ShmDuplexConfig:
    name_prefix: str
    buffer_size_bytes: int


class ShmDuplexTransport:
    def __init__(self, config: ShmDuplexConfig):
        self.config = config

    def open(self, role: TransportRole, timeout: float = 10.0) -> DuplexByteChannels:
        fwd_name = f"{self.config.name_prefix}_fwd"
        bwd_name = f"{self.config.name_prefix}_bwd"

        if role == TransportRole.SERVER:
            inbound = ShmRingBuffer(fwd_name, self.config.buffer_size_bytes, create=True)
            outbound = ShmRingBuffer(bwd_name, self.config.buffer_size_bytes, create=True)
            return DuplexByteChannels(outbound=outbound, inbound=inbound)

        start_time = time.time()
        while True:
            try:
                outbound = ShmRingBuffer(fwd_name, self.config.buffer_size_bytes, create=False)
                inbound = ShmRingBuffer(bwd_name, self.config.buffer_size_bytes, create=False)
                return DuplexByteChannels(outbound=outbound, inbound=inbound)
            except FileNotFoundError:
                if time.time() - start_time > timeout:
                    raise TimeoutError("Could not connect to LatentLink receiver (SHM not found)")
                time.sleep(0.5)
