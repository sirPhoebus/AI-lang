import struct
import json
import hashlib
from enum import IntEnum
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any
import numpy as np

class MessageKind(IntEnum):
    HANDSHAKE = 1
    HANDSHAKE_ACK = 2
    KV_CACHE_DELTA = 3
    TEXT_TURN = 7
    RESIDUAL_PATCH = 4
    HIDDEN_STATE_DUMP = 5
    VIRTUAL_PREFIX = 6
    CLOSE = 99

class CloseReason(IntEnum):
    PROTOCOL_VERSION_MISMATCH = 1
    WEIGHTS_HASH_MISMATCH = 2
    MODEL_FAMILY_MISMATCH = 3
    ARCHITECTURE_ID_MISMATCH = 4
    LAYER_COUNT_MISMATCH = 5
    HIDDEN_SIZE_MISMATCH = 6
    NORM_TYPE_MISMATCH = 7
    NORM_POSITION_MISMATCH = 8
    TOKENIZER_VOCAB_SIZE_MISMATCH = 9
    TOKENIZER_BOS_TOKEN_ID_MISMATCH = 10
    TOKENIZER_EOS_TOKEN_ID_MISMATCH = 11
    TOKENIZER_PAD_TOKEN_ID_MISMATCH = 12
    TOKENIZER_UNK_TOKEN_ID_MISMATCH = 13
    SUPPORTED_DTYPES_MISMATCH = 14
    SUPPORTED_MESSAGE_KINDS_MISMATCH = 15
    MAX_MESSAGE_BYTES_MISMATCH = 16
    NUM_ATTENTION_HEADS_MISMATCH = 17
    NUM_KEY_VALUE_HEADS_MISMATCH = 18
    HEAD_DIM_MISMATCH = 19
    POSITIONAL_EMBEDDING_TYPE_MISMATCH = 20
    MAX_POSITION_EMBEDDINGS_MISMATCH = 21
    ROPE_THETA_MISMATCH = 22
    ROPE_SCALING_MISMATCH = 23
    MODEL_DTYPE_MISMATCH = 24
    AUTH_TOKEN_MISMATCH = 25
    MALFORMED_HANDSHAKE = 100

class DType(IntEnum):
    FLOAT32 = 1
    FLOAT16 = 2
    BFLOAT16 = 3
    INT8 = 4

    @staticmethod
    def from_numpy(dtype):
        dt = np.dtype(dtype)
        if dt == np.dtype(np.float32):
            return DType.FLOAT32
        if dt == np.dtype(np.float16):
            return DType.FLOAT16
        try:
            if dt == np.dtype('bfloat16'):
                return DType.BFLOAT16
        except TypeError:
            pass
        if dt == np.dtype(np.int8):
            return DType.INT8
        raise ValueError(f"Unsupported numpy dtype: {dt}")

    @staticmethod
    def to_numpy(dtype_code: int):
        if int(dtype_code) == int(DType.FLOAT32):
            return np.float32
        if int(dtype_code) == int(DType.FLOAT16):
            return np.float16
        if int(dtype_code) == int(DType.BFLOAT16):
            try:
                return np.dtype('bfloat16')
            except TypeError:
                raise ValueError("Unsupported dtype code: BFLOAT16")
        if int(dtype_code) == int(DType.INT8):
            return np.int8
        raise ValueError(f"Unsupported dtype code: {dtype_code}")

@dataclass
class TensorMeta:
    name: str
    dtype: int # DType value
    shape: List[int]
    byte_length: int

@dataclass
class LatentLinkMessage:
    session_id: str
    msg_id: int
    kind: MessageKind
    metadata: Dict[str, Any] = field(default_factory=dict)
    tensors: List[TensorMeta] = field(default_factory=list)
    payload_bytes: bytes = b''

    def pack(self) -> bytes:
        """
        Format:
        [HEADER_LEN (4B)][HEADER_JSON (N B)][PAYLOAD (M B)]
        """
        if self.metadata.get("bench") is not None:
            payload_checksum = ""
        else:
            payload_checksum = hashlib.sha256(self.payload_bytes).hexdigest()
        header_dict = {
            "session_id": self.session_id,
            "msg_id": self.msg_id,
            "kind": int(self.kind),
            "metadata": self.metadata,
            "tensors": [asdict(t) for t in self.tensors],
            "payload_length": len(self.payload_bytes),
            "payload_checksum": payload_checksum
        }
        header_bytes = json.dumps(header_dict).encode('utf-8')
        header_len = len(header_bytes)
        
        # simple framing: len | header | payload
        return struct.pack(f'!I{header_len}s', header_len, header_bytes) + self.payload_bytes

    @classmethod
    def unpack_header(cls, buffer: bytes):
        """
        Reads just the header from the start of the buffer.
        Returns header_dict, payload_start_offset
        """
        header_len = struct.unpack('!I', buffer[:4])[0]
        header_bytes = buffer[4:4+header_len]
        header_dict = json.loads(header_bytes.decode('utf-8'))
        return header_dict, 4 + header_len

    @classmethod
    def unpack(cls, buffer: bytes) -> 'LatentLinkMessage':
        header_dict, payload_offset = cls.unpack_header(buffer)
        payload = buffer[payload_offset:]
        
        tensors = [TensorMeta(**t) for t in header_dict.get("tensors", [])]
        
        return cls(
            session_id=header_dict["session_id"],
            msg_id=header_dict["msg_id"],
            kind=MessageKind(header_dict["kind"]),
            metadata=header_dict.get("metadata", {}),
            tensors=tensors,
            payload_bytes=payload
        )

@dataclass
class HandshakeData:
    protocol_version: int
    auth_token_hash: str
    model_family: str
    architecture_id: str
    layer_count: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    positional_embedding_type: str
    max_position_embeddings: int
    rope_theta: Optional[float]
    rope_scaling: Optional[Dict[str, Any]]
    weights_hash: str
    norm_type: str
    norm_position: str
    tokenizer_vocab_size: int
    tokenizer_bos_token_id: Optional[int]
    tokenizer_eos_token_id: Optional[int]
    tokenizer_pad_token_id: Optional[int]
    tokenizer_unk_token_id: Optional[int]
    model_dtype: int
    supported_dtypes: List[int]
    supported_message_kinds: List[int]
    max_message_bytes: int
    
    def to_dict(self):
        return asdict(self)
