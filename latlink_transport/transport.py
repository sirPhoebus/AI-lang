from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol


class TransportRole(Enum):
    CLIENT = 1
    SERVER = 2


class ByteChannel(Protocol):
    def write_bytes(self, data: bytes, timeout: float = 5.0) -> int:
        ...

    def read_bytes(self, num_bytes: int, timeout: float = 5.0) -> Optional[bytes]:
        ...

    def close(self) -> None:
        ...

    def unlink(self) -> None:
        ...


@dataclass
class DuplexByteChannels:
    outbound: ByteChannel
    inbound: ByteChannel
