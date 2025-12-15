import hashlib
import json
import torch
import numpy as np
from typing import List, Tuple, Any, Dict

from latlink_config.config import LatentLinkConfig
from latlink_protocol.protocol import DType, MessageKind
from latlink_transport_shm.shm_transport import CONTROL_BLOCK_SIZE

def get_model_signature(model, tokenizer) -> Dict[str, Any]:
    """
    Returns a dictionary of signature verification data compatible with HandshakeData.
    {
        "protocol_version": ...,
        "model_family": ...,
        "architecture_id": ...,
        "layer_count": ...,
        "hidden_size": ...,
        "num_attention_heads": ...,
        "num_key_value_heads": ...,
        "head_dim": ...,
        "positional_embedding_type": ...,
        "max_position_embeddings": ...,
        "rope_theta": ...,
        "rope_scaling": ...,
        "weights_hash": ...,
        "norm_type": ...,
        "norm_position": ...,
        "tokenizer_vocab_size": ...,
        "tokenizer_bos_token_id": ...,
        "tokenizer_eos_token_id": ...,
        "tokenizer_pad_token_id": ...,
        "tokenizer_unk_token_id": ...,
        "model_dtype": ...,
        "supported_dtypes": ...,
        "supported_message_kinds": ...,
        "max_message_bytes": ...
    }
    """
    cfg = model.config

    revision = getattr(cfg, "_commit_hash", None)
    model_id = getattr(cfg, "_name_or_path", None)
    if revision is not None and model_id is not None:
        weights_hash = hashlib.sha256(f"{model_id}@{revision}".encode("utf-8")).hexdigest()
    else:
        # Create a stable hash of the critical configuration
        config_dict = cfg.to_dict()
        # Filter out path-dependent or volatile keys
        keys_to_ignore = ["_name_or_path", "transformers_version", "torch_dtype"]
        clean_config = {k: v for k, v in config_dict.items() if k not in keys_to_ignore}
        config_json = json.dumps(clean_config, sort_keys=True)
        weights_hash = hashlib.sha256(config_json.encode('utf-8')).hexdigest()

    model_family = getattr(cfg, "model_type", None)
    if model_family is None:
        raise ValueError("Missing model_type in model.config")

    architectures = getattr(cfg, "architectures", None)
    if architectures is None or not architectures:
        raise ValueError("Missing architectures in model.config")

    layer_count = getattr(cfg, "num_hidden_layers", None)
    if layer_count is None:
        raise ValueError("Missing num_hidden_layers in model.config")

    hidden_size = getattr(cfg, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Missing hidden_size in model.config")

    num_attention_heads = getattr(cfg, "num_attention_heads", None)
    if num_attention_heads is None:
        raise ValueError("Missing num_attention_heads in model.config")

    num_key_value_heads = getattr(cfg, "num_key_value_heads", None)
    if num_key_value_heads is None:
        num_key_value_heads = int(num_attention_heads)

    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        if int(hidden_size) % int(num_attention_heads) != 0:
            raise ValueError("hidden_size not divisible by num_attention_heads")
        head_dim = int(hidden_size) // int(num_attention_heads)

    max_position_embeddings = getattr(cfg, "max_position_embeddings", None)
    if max_position_embeddings is None:
        raise ValueError("Missing max_position_embeddings in model.config")

    positional_embedding_type = getattr(cfg, "position_embedding_type", None)
    rope_theta = getattr(cfg, "rope_theta", None)
    rope_scaling = getattr(cfg, "rope_scaling", None)
    if positional_embedding_type is None:
        if rope_theta is not None or rope_scaling is not None:
            positional_embedding_type = "rope"
        else:
            positional_embedding_type = "absolute"

    if positional_embedding_type == "rope":
        if rope_theta is None:
            raise ValueError("Missing rope_theta in model.config for rope positional embeddings")
        if rope_scaling is not None and not isinstance(rope_scaling, dict):
            raise ValueError("Invalid rope_scaling in model.config: expected dict or None")
    else:
        if rope_theta is not None:
            raise ValueError("Unexpected rope_theta for non-rope positional embeddings")
        if rope_scaling is not None:
            raise ValueError("Unexpected rope_scaling for non-rope positional embeddings")

    norm_type = None
    for m in model.modules():
        cls_name = type(m).__name__
        if "RMSNorm" in cls_name or "RmsNorm" in cls_name:
            norm_type = "rmsnorm"
            break
        if isinstance(m, torch.nn.LayerNorm):
            norm_type = "layernorm"
            break
    if norm_type is None:
        raise ValueError("Could not detect normalization type")

    norm_position = None
    if hasattr(cfg, "pre_norm"):
        norm_position = "pre" if bool(getattr(cfg, "pre_norm")) else "post"
    elif hasattr(cfg, "do_layer_norm_before"):
        norm_position = "pre" if bool(getattr(cfg, "do_layer_norm_before")) else "post"
    else:
        norm_position = "unknown"

    if tokenizer is None:
        raise ValueError("tokenizer is required")

    tokenizer_vocab_size = int(len(tokenizer))
    tokenizer_bos_token_id = getattr(tokenizer, "bos_token_id", None)
    tokenizer_eos_token_id = getattr(tokenizer, "eos_token_id", None)
    tokenizer_pad_token_id = getattr(tokenizer, "pad_token_id", None)
    tokenizer_unk_token_id = getattr(tokenizer, "unk_token_id", None)

    param_dtype = next(model.parameters()).dtype
    if param_dtype == torch.float32:
        dtype_code = int(DType.FLOAT32)
    elif param_dtype == torch.float16:
        dtype_code = int(DType.FLOAT16)
    elif param_dtype == torch.bfloat16:
        dtype_code = int(DType.BFLOAT16)
    else:
        raise ValueError(f"Unsupported torch dtype: {param_dtype}")

    supported_message_kinds = [int(MessageKind.KV_CACHE_DELTA)]

    return {
        "protocol_version": int(LatentLinkConfig.PROTOCOL_VERSION),
        "auth_token_hash": str(LatentLinkConfig.AUTH_TOKEN_HASH),
        "model_family": model_family,
        "architecture_id": architectures[0],
        "layer_count": int(layer_count),
        "hidden_size": int(hidden_size),
        "num_attention_heads": int(num_attention_heads),
        "num_key_value_heads": int(num_key_value_heads),
        "head_dim": int(head_dim),
        "positional_embedding_type": str(positional_embedding_type),
        "max_position_embeddings": int(max_position_embeddings),
        "rope_theta": float(rope_theta) if rope_theta is not None else None,
        "rope_scaling": rope_scaling,
        "weights_hash": weights_hash,
        "norm_type": norm_type,
        "norm_position": norm_position,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "tokenizer_bos_token_id": tokenizer_bos_token_id,
        "tokenizer_eos_token_id": tokenizer_eos_token_id,
        "tokenizer_pad_token_id": tokenizer_pad_token_id,
        "tokenizer_unk_token_id": tokenizer_unk_token_id,
        "model_dtype": int(dtype_code),
        "supported_dtypes": [dtype_code],
        "supported_message_kinds": supported_message_kinds,
        "max_message_bytes": int(LatentLinkConfig.SHM_BUFFER_SIZE_BYTES - CONTROL_BLOCK_SIZE)
    }

def capture_kv_cache(past_key_values: Tuple[Tuple[torch.Tensor]]) -> List[Tuple[str, np.ndarray]]:
    """
    Flattens HF past_key_values (tuple of tuples) into a list of (name, numpy_array).
    Moves tensors to CPU if needed.
    """
    flattened = []
    for i, layer_kv in enumerate(past_key_values):
        # layer_kv is usually (key, value)
        # But some models might have different structure (e.g. key_value cache in one tensor)
        # Assuming standard (k, v) for now.
        if len(layer_kv) != 2:
            raise ValueError("Unsupported past_key_values structure: expected (key, value) per layer")
        
        k, v = layer_kv
        # Blocking CPU transfer
        flattened.append((f"layer_{i}_key", k.detach().cpu().numpy()))
        flattened.append((f"layer_{i}_value", v.detach().cpu().numpy()))
        
    return flattened

def inject_kv_cache(flat_tensors: List[Tuple[str, np.ndarray]], device="cpu", dtype=None) -> Tuple[Tuple[torch.Tensor]]:
    """
    Reconstructs the tuple-of-tuples structure from flattened tensors.
    """
    layers: Dict[int, Dict[str, torch.Tensor]] = {}
    
    for name, arr in flat_tensors:
        # Expected name: "layer_{i}_{type}"
        parts = name.split('_')
        if len(parts) != 3:
            raise ValueError(f"Invalid tensor name: {name}")
         
        try:
            layer_idx = int(parts[1])
            kind = parts[2] # "key" or "value"
        except ValueError:
            raise ValueError(f"Invalid tensor name: {name}")

        if kind not in {"key", "value"}:
            raise ValueError(f"Invalid tensor kind in name: {name}")
            
        if layer_idx not in layers:
            layers[layer_idx] = {}
            
        t = torch.from_numpy(arr).to(device)
        if dtype:
            t = t.to(dtype)
        layers[layer_idx][kind] = t

    # Reassemble
    num_layers = max(layers.keys()) + 1
    result = []
    for i in range(num_layers):
        if i not in layers:
           # Missing layer? 
           raise ValueError(f"Missing KV cache for layer {i}")
        
        layer_dict = layers[i]
        result.append((layer_dict['key'], layer_dict['value']))
        
    return tuple(result)
