import time
import logging
import uuid
import struct
import json
import hashlib
from enum import Enum
from typing import Optional

from latlink_config.config import LatentLinkConfig
from latlink_transport_shm.shm_transport import ShmRingBuffer
from latlink_protocol.protocol import LatentLinkMessage, MessageKind, CloseReason, HandshakeData, TensorMeta

class SessionRole(Enum):
    SENDER = 1
    RECEIVER = 2

class LatentLinkSession:
    def __init__(self, role: SessionRole, name: str = None):
        self.role = role
        self.name = name or LatentLinkConfig.SHM_NAME_PREFIX
        self.session_id = str(uuid.uuid4())
        self.logger = logging.getLogger(f"LatLinkSession-{role.name}")
        
        # Buffer names
        self.fwd_name = f"{self.name}_fwd"
        self.bwd_name = f"{self.name}_bwd"
        
        self.shm_out: Optional[ShmRingBuffer] = None
        self.shm_in: Optional[ShmRingBuffer] = None
        self.msg_counter = 0

    def start(self, timeout=10.0):
        start_time = time.time()
        if self.role == SessionRole.RECEIVER:
            # Receiver creates buffers
            self.logger.info(f"Receiver creating SHM: {self.fwd_name}, {self.bwd_name}")
            self.shm_in = ShmRingBuffer(self.fwd_name, LatentLinkConfig.SHM_BUFFER_SIZE_BYTES, create=True) # FWD is IN for Receiver
            self.shm_out = ShmRingBuffer(self.bwd_name, LatentLinkConfig.SHM_BUFFER_SIZE_BYTES, create=True)      # BWD is OUT for Receiver
        else:
            # Sender connects to buffers
            self.logger.info(f"Sender connecting to SHM: {self.fwd_name}")
            while True:
                try:
                    self.shm_out = ShmRingBuffer(self.fwd_name, LatentLinkConfig.SHM_BUFFER_SIZE_BYTES, create=False)
                    self.shm_in = ShmRingBuffer(self.bwd_name, LatentLinkConfig.SHM_BUFFER_SIZE_BYTES, create=False)
                    break
                except FileNotFoundError:
                    if time.time() - start_time > timeout:
                        raise TimeoutError("Could not connect to LatentLink receiver (SHM not found)")
                    time.sleep(0.5)

    def close(self):
        if self.shm_out: self.shm_out.close()
        if self.shm_in: self.shm_in.close()
        # Receiver should unlink? 
        # For now, rely on manual cleanup or OS cleanup (Windows SHM is tricky with unlink)
        # Python shared_memory unlink() is essential on Linux, on Windows it's ref-counted.
        if self.role == SessionRole.RECEIVER:
            if self.shm_in: self.shm_in.unlink()
            if self.shm_out: self.shm_out.unlink()

    def send_close(self, session_id: str, reason: CloseReason) -> None:
        close_msg = LatentLinkMessage(
            session_id=session_id,
            msg_id=self.msg_counter,
            kind=MessageKind.CLOSE,
            metadata={"reason_code": int(reason)}
        )
        self.msg_counter += 1
        self.shm_out.write_bytes(close_msg.pack())

    def send_handshake(self, handshake_data: HandshakeData):
        if self.role != SessionRole.SENDER:
            raise RuntimeError("Only SENDER can initiate handshake")
        
        msg = LatentLinkMessage(
            session_id=self.session_id,
            msg_id=self.msg_counter,
            kind=MessageKind.HANDSHAKE,
            metadata=handshake_data.to_dict()
        )
        self.msg_counter += 1
        
        packed = msg.pack()
        self.shm_out.write_bytes(packed)
        self.logger.info("Sent HANDSHAKE")

        end_time = time.time() + LatentLinkConfig.HANDSHAKE_TIMEOUT_SEC
        while True:
            if time.time() > end_time:
                raise TimeoutError("Handshake ACK timeout")

            resp = self.recv_message()
            if resp is None:
                time.sleep(0.01)
                continue

            if resp.session_id != self.session_id:
                raise RuntimeError("Handshake ACK session_id mismatch")

            if resp.kind == MessageKind.CLOSE:
                raise RuntimeError(f"Handshake rejected reason_code={resp.metadata.get('reason_code')}")

            if resp.kind != MessageKind.HANDSHAKE_ACK:
                raise RuntimeError(f"Expected HANDSHAKE_ACK, got {resp.kind}")

            self.logger.info("Received HANDSHAKE_ACK")
            return True

    def recv_handshake(self, my_handshake_data: HandshakeData) -> bool:
        if self.role != SessionRole.RECEIVER:
            raise RuntimeError("Only RECEIVER can receive handshake")
            
        msg = self.recv_message()
        if msg is None:
            return False
        if msg.kind != MessageKind.HANDSHAKE:
            self.logger.error(f"Expected HANDSHAKE, got {msg.kind}")
            return False
            
        # Verify
        other_data = msg.metadata

        if other_data.get("auth_token_hash") != my_handshake_data.auth_token_hash:
            self.logger.error("auth_token_hash mismatch!")
            self.send_close(msg.session_id, CloseReason.AUTH_TOKEN_MISMATCH)
            return False

        if other_data.get("protocol_version") != my_handshake_data.protocol_version:
            self.logger.error("Protocol version mismatch!")
            self.send_close(msg.session_id, CloseReason.PROTOCOL_VERSION_MISMATCH)
            return False

        if other_data.get("weights_hash") != my_handshake_data.weights_hash:
            self.logger.error("Weights hash mismatch!")
            self.send_close(msg.session_id, CloseReason.WEIGHTS_HASH_MISMATCH)
            return False

        if other_data.get("model_family") != my_handshake_data.model_family:
            self.logger.error("Model family mismatch!")
            self.send_close(msg.session_id, CloseReason.MODEL_FAMILY_MISMATCH)
            return False

        if other_data.get("architecture_id") != my_handshake_data.architecture_id:
            self.logger.error("Architecture mismatch!")
            self.send_close(msg.session_id, CloseReason.ARCHITECTURE_ID_MISMATCH)
            return False

        if other_data.get("layer_count") != my_handshake_data.layer_count:
            self.logger.error("Layer count mismatch!")
            self.send_close(msg.session_id, CloseReason.LAYER_COUNT_MISMATCH)
            return False

        if other_data.get("hidden_size") != my_handshake_data.hidden_size:
            self.logger.error("Hidden size mismatch!")
            self.send_close(msg.session_id, CloseReason.HIDDEN_SIZE_MISMATCH)
            return False

        if other_data.get("num_attention_heads") != my_handshake_data.num_attention_heads:
            self.logger.error("num_attention_heads mismatch!")
            self.send_close(msg.session_id, CloseReason.NUM_ATTENTION_HEADS_MISMATCH)
            return False

        if other_data.get("num_key_value_heads") != my_handshake_data.num_key_value_heads:
            self.logger.error("num_key_value_heads mismatch!")
            self.send_close(msg.session_id, CloseReason.NUM_KEY_VALUE_HEADS_MISMATCH)
            return False

        if other_data.get("head_dim") != my_handshake_data.head_dim:
            self.logger.error("head_dim mismatch!")
            self.send_close(msg.session_id, CloseReason.HEAD_DIM_MISMATCH)
            return False

        if other_data.get("positional_embedding_type") != my_handshake_data.positional_embedding_type:
            self.logger.error("positional_embedding_type mismatch!")
            self.send_close(msg.session_id, CloseReason.POSITIONAL_EMBEDDING_TYPE_MISMATCH)
            return False

        if other_data.get("max_position_embeddings") != my_handshake_data.max_position_embeddings:
            self.logger.error("max_position_embeddings mismatch!")
            self.send_close(msg.session_id, CloseReason.MAX_POSITION_EMBEDDINGS_MISMATCH)
            return False

        if other_data.get("rope_theta") != my_handshake_data.rope_theta:
            self.logger.error("rope_theta mismatch!")
            self.send_close(msg.session_id, CloseReason.ROPE_THETA_MISMATCH)
            return False

        if other_data.get("rope_scaling") != my_handshake_data.rope_scaling:
            self.logger.error("rope_scaling mismatch!")
            self.send_close(msg.session_id, CloseReason.ROPE_SCALING_MISMATCH)
            return False

        if other_data.get("norm_type") != my_handshake_data.norm_type:
            self.logger.error("Normalization type mismatch!")
            self.send_close(msg.session_id, CloseReason.NORM_TYPE_MISMATCH)
            return False

        if other_data.get("norm_position") != my_handshake_data.norm_position:
            self.logger.error("Normalization position mismatch!")
            self.send_close(msg.session_id, CloseReason.NORM_POSITION_MISMATCH)
            return False

        if other_data.get("tokenizer_vocab_size") != my_handshake_data.tokenizer_vocab_size:
            self.logger.error("Tokenizer vocab size mismatch!")
            self.send_close(msg.session_id, CloseReason.TOKENIZER_VOCAB_SIZE_MISMATCH)
            return False

        if other_data.get("tokenizer_bos_token_id") != my_handshake_data.tokenizer_bos_token_id:
            self.logger.error("Tokenizer bos_token_id mismatch!")
            self.send_close(msg.session_id, CloseReason.TOKENIZER_BOS_TOKEN_ID_MISMATCH)
            return False

        if other_data.get("tokenizer_eos_token_id") != my_handshake_data.tokenizer_eos_token_id:
            self.logger.error("Tokenizer eos_token_id mismatch!")
            self.send_close(msg.session_id, CloseReason.TOKENIZER_EOS_TOKEN_ID_MISMATCH)
            return False

        if other_data.get("tokenizer_pad_token_id") != my_handshake_data.tokenizer_pad_token_id:
            self.logger.error("Tokenizer pad_token_id mismatch!")
            self.send_close(msg.session_id, CloseReason.TOKENIZER_PAD_TOKEN_ID_MISMATCH)
            return False

        if other_data.get("tokenizer_unk_token_id") != my_handshake_data.tokenizer_unk_token_id:
            self.logger.error("Tokenizer unk_token_id mismatch!")
            self.send_close(msg.session_id, CloseReason.TOKENIZER_UNK_TOKEN_ID_MISMATCH)
            return False

        if other_data.get("model_dtype") != my_handshake_data.model_dtype:
            self.logger.error("model_dtype mismatch!")
            self.send_close(msg.session_id, CloseReason.MODEL_DTYPE_MISMATCH)
            return False

        other_supported_dtypes = other_data.get("supported_dtypes")
        if other_supported_dtypes is None:
            raise RuntimeError("Missing supported_dtypes in handshake")
 
        if set(other_supported_dtypes) != set(my_handshake_data.supported_dtypes):
            self.logger.error("Supported dtypes mismatch!")
            self.send_close(msg.session_id, CloseReason.SUPPORTED_DTYPES_MISMATCH)
            return False

        other_supported_kinds = other_data.get("supported_message_kinds")
        if other_supported_kinds is None:
            raise RuntimeError("Missing supported_message_kinds in handshake")
 
        if set(other_supported_kinds) != set(my_handshake_data.supported_message_kinds):
            self.logger.error("Supported message kinds mismatch!")
            self.send_close(msg.session_id, CloseReason.SUPPORTED_MESSAGE_KINDS_MISMATCH)
            return False

        if other_data.get("max_message_bytes") != my_handshake_data.max_message_bytes:
            self.logger.error("max_message_bytes mismatch!")
            self.send_close(msg.session_id, CloseReason.MAX_MESSAGE_BYTES_MISMATCH)
            return False

        self.session_id = msg.session_id

        # Send ACK
        ack = LatentLinkMessage(
            session_id=self.session_id,
            msg_id=self.msg_counter,
            kind=MessageKind.HANDSHAKE_ACK
        )
        self.shm_out.write_bytes(ack.pack())
        self.msg_counter += 1
        return True

    def send_message(self, kind: MessageKind, metadata: dict, payload: bytes):
        msg = LatentLinkMessage(
            session_id=self.session_id,
            msg_id=self.msg_counter,
            kind=kind,
            metadata=metadata,
            payload_bytes=payload
        )
        self.msg_counter += 1
        self.shm_out.write_bytes(msg.pack())

    def recv_message(self) -> LatentLinkMessage:
        # 1. Read Header Len (4 bytes)
        # We need to peek or read exactly 4 bytes.
        len_bytes = self.shm_in.read_bytes(4)
        if not len_bytes:
            return None
            
        header_len = struct.unpack('!I', len_bytes)[0]
        
        # 2. Read Header
        header_bytes = self.shm_in.read_bytes(header_len)
        header_dict = json.loads(header_bytes.decode('utf-8'))

        # 3. Read Payload (if any, how do we know payload len? 
        # It's NOT in the fixed 4 byte prefix... logic flaw in protocol.py `pack` 
        # which just did [Len][Header][Payload]. 
        # Header has NO payload length field in `pack` logic explicitly outside JSON.
        # BUT `pack` does `struct.pack(f'!I{header_len}s', header_len, header_bytes) + self.payload_bytes`.
        # The Receiver doesn't know how many payload bytes to read!
        # Fix: The HEADER JSON must contain `payload_length`.
        
        payload_len = header_dict.get("payload_length", 0) # We need to add this to pack()
        
        payload = b''
        if payload_len > 0:
            payload = self.shm_in.read_bytes(payload_len)

        payload_checksum = header_dict.get("payload_checksum")
        if payload_checksum is None:
            raise RuntimeError("Missing payload_checksum in message header")

        if payload_checksum != "":
            computed_checksum = hashlib.sha256(payload).hexdigest()
            if computed_checksum != payload_checksum:
                raise RuntimeError("Payload checksum mismatch")

        tensors = [TensorMeta(**t) for t in header_dict.get("tensors", [])]
        
        # Reconstruct
        # (Assuming imports from protocol are sufficient)
        return LatentLinkMessage(
            session_id=header_dict["session_id"],
            msg_id=header_dict["msg_id"],
            kind=MessageKind(header_dict["kind"]),
            metadata=header_dict.get("metadata", {}),
            tensors=tensors,
            payload_bytes=payload
        )
