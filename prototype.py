# tests/kv_teleport_qwen_polished.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import json
import struct
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

def load_model_tokenizer():
    print("[LOAD] Starting model and tokenizer load...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",  # Faster on RTX 4090
    )
    model.eval()
    print("[LOAD] Model and tokenizer loaded successfully.")
    return model, tokenizer

def sender_process(prompt_tokens: list, generate_len: int, shm_name: str, ready_event: mp.Event, done_event: mp.Event):
    print("[SENDER] Starting sender process...")
    model, tokenizer = load_model_tokenizer()

    print(f"[SENDER] Generating {generate_len} tokens...")
    input_ids = torch.tensor([prompt_tokens], device=model.device)

    output = model.generate(
        input_ids,
        max_new_tokens=generate_len,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )
    print("[SENDER] Generation complete.")

    past_key_values = output.past_key_values
    sequences = output.sequences
    last_token_id = int(sequences[0, -1].item())

    sample_kv = past_key_values[0][0]
    kv_dtype = sample_kv.dtype
    if kv_dtype == torch.float16 or kv_dtype == torch.bfloat16:
        bit_dtype = torch.uint16
        itemsize = 2
    elif kv_dtype == torch.float32:
        bit_dtype = torch.uint32
        itemsize = 4
    else:
        raise ValueError(f"Unsupported KV dtype: {kv_dtype}")

    shapes = [tuple(kv.shape) for layer in past_key_values for kv in layer]

    total_elements = int(sum(np.prod(s) for s in shapes))
    total_bytes = total_elements * itemsize

    metadata = {
        "dtype": str(kv_dtype),
        "itemsize": itemsize,
        "shapes": shapes,
        "generated_len": generate_len,
        "last_token_id": last_token_id,
    }
    metadata_bytes = json.dumps(metadata).encode('utf-8')

    shm_size = int(8 + len(metadata_bytes) + total_bytes)

    print(f"[SENDER] Creating shared memory (~{shm_size / 1e6:.1f} MB)...")
    shm = shared_memory.SharedMemory(create=True, name=shm_name, size=shm_size)

    shm.buf[:8] = struct.pack(">Q", len(metadata_bytes))
    shm.buf[8:8 + len(metadata_bytes)] = metadata_bytes

    buf_dtype = np.uint16 if itemsize == 2 else np.uint32
    buf = np.ndarray((total_elements,), dtype=buf_dtype, buffer=shm.buf[8 + len(metadata_bytes):])

    print("[SENDER] Serializing KV cache...")
    offset = 0
    for layer in past_key_values:
        for kv in layer:
            size = kv.numel()
            flat_bits = kv.cpu().view(bit_dtype).flatten().numpy()
            buf[offset:offset + size] = flat_bits
            offset += size
    print("[SENDER] KV cache serialized. Signaling ready.")

    ready_event.set()
    done_event.wait()
    shm.close()
    shm.unlink()
    print("[SENDER] Done.")

def receiver_process(shm_name: str, generate_more: int, ready_event: mp.Event, done_event: mp.Event):
    print("[RECEIVER] Starting receiver process...")
    model, tokenizer = load_model_tokenizer()

    print("[RECEIVER] Waiting for sender...")
    ready_event.wait()
    print("[RECEIVER] Sender ready. Attaching to shared memory...")

    shm = shared_memory.SharedMemory(name=shm_name)

    metadata_len = struct.unpack(">Q", shm.buf[:8])[0]
    metadata_bytes = bytes(shm.buf[8:8 + metadata_len])
    metadata = json.loads(metadata_bytes)

    kv_dtype_str = metadata["dtype"]
    itemsize = metadata["itemsize"]
    shapes = metadata["shapes"]
    last_token_id = metadata.get("last_token_id", tokenizer.eos_token_id)

    kv_dtype = {"torch.float16": torch.float16, "torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32}[kv_dtype_str]
    bit_dtype = torch.uint16 if itemsize == 2 else torch.uint32

    total_elements = int(sum(np.prod(s) for s in shapes))
    payload_offset = 8 + metadata_len
    buf_dtype = np.uint16 if itemsize == 2 else np.uint32
    buf = np.ndarray((total_elements,), dtype=buf_dtype, buffer=shm.buf[payload_offset:])

    past_key_values = []
    offset = 0
    for i in range(0, len(shapes), 2):
        key_shape = shapes[i]
        val_shape = shapes[i + 1]
        key_size = int(np.prod(key_shape))
        val_size = int(np.prod(val_shape))

        key_flat = buf[offset:offset + key_size]
        val_flat = buf[offset + key_size:offset + key_size + val_size]

        key = torch.from_numpy(key_flat).view(kv_dtype).reshape(key_shape).to(model.device)
        val = torch.from_numpy(val_flat).view(kv_dtype).reshape(val_shape).to(model.device)

        past_key_values.append((key, val))
        offset += key_size + val_size
    past_key_values = tuple(past_key_values)

    print("[RECEIVER] KV cache reconstructed. Continuing generation...")

    seq_len_so_far = past_key_values[0][0].shape[2]
    cache_position = torch.arange(seq_len_so_far, seq_len_so_far + 1, device=model.device)

    input_ids = torch.tensor([[last_token_id]], device=model.device)

    continuation = model.generate(
        input_ids,
        max_new_tokens=generate_more,
        past_key_values=past_key_values,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
        cache_position=cache_position,
    )

    # Polish: Decode only the new tokens
    new_tokens = continuation[0, -generate_more:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print("\n=== Receiver continuation after teleport (clean) ===")
    print(text)

    shm.close()
    done_event.set()
    print("[RECEIVER] Done.")

if __name__ == "__main__":
    print("=== Starting polished KV teleport demo with FlashAttention-2 ===")
    prompt = "Explain quantum entanglement in simple terms:"
    model_ref, tokenizer = load_model_tokenizer()

    input_ids_ref = tokenizer(prompt, return_tensors="pt").input_ids.to(model_ref.device)

    print("Generating baseline full output...")
    full_output = model_ref.generate(
        input_ids_ref,
        max_new_tokens=64 + 32,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    full_text = tokenizer.decode(full_output[0], skip_special_tokens=True)
    print("=== Baseline single-process full generation ===")
    print(full_text)

    prompt_tokens = tokenizer(prompt).input_ids

    shm_name = f"kv_teleport_{os.getpid()}"
    ready_ev = mp.Event()
    done_ev = mp.Event()

    sender = mp.Process(target=sender_process, args=(prompt_tokens, 64, shm_name, ready_ev, done_ev))
    receiver = mp.Process(target=receiver_process, args=(shm_name, 32, ready_ev, done_ev))

    sender.start()
    receiver.start()

    sender.join()
    receiver.join()

    print("=== Demo complete ===")