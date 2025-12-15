from __future__ import annotations

from dataclasses import dataclass

from latlink_transport.transport import DuplexByteChannels, TransportRole


@dataclass
class GrpcTransportConfig:
    host: str
    port: int


class GrpcTransport:
    def __init__(self, config: GrpcTransportConfig):
        self.config = config

    def open(self, role: TransportRole, timeout: float = 10.0) -> DuplexByteChannels:
        raise NotImplementedError("GrpcTransport is not implemented")
