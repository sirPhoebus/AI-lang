import sys
import os
import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from latlink_session.session import LatentLinkSession, SessionRole
from latlink_protocol.protocol import MessageKind, DType, HandshakeData, TensorMeta
from latlink_integrations_hf.hf_integration import get_model_signature, inject_kv_cache

def main():
    print("[Receiver] Loading model...")
    model_name = os.environ["LATLINK_MODEL_NAME"]
    revision = os.environ.get("LATLINK_MODEL_REVISION")
    if revision:
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, use_safetensors=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True)
    model.eval()
    
    # 1. Start Server
    print("[Receiver] Starting LatentLink Server...")
    session = LatentLinkSession(SessionRole.RECEIVER)
    session.start() # Creates SHM
    
    # 2. Handshake
    print("[Receiver] Waiting for handshake...")
    my_sig = HandshakeData(**get_model_signature(model, tokenizer))
    
    # Simple polling for handshake
    while True:
        try:
            # Need a blocking recv or poll
            # Our session recv_handshake blocks on recv_message which calls read_bytes(blocking)
            if session.recv_handshake(my_sig):
                print("[Receiver] Handshake success!")
                break
        except Exception as e:
            print(f"[Receiver] Handshake error: {e}")
            time.sleep(1)

    # 3. Receive Data
    print("[Receiver] Waiting for KV Cache...")
    while True:
        msg = session.recv_message()
        if msg is None:
            continue
        break
    if msg.kind == MessageKind.KV_CACHE_DELTA:
        print(f"[Receiver] Received KV Bundle: {len(msg.tensors)} tensors, {len(msg.payload_bytes)} bytes")
        
        # Reconstruct Arrays
        flat_tensors = []
        offset = 0
        for meta in msg.tensors:
            dt = DType.to_numpy(meta.dtype)
             
            # Read
            end = offset + meta.byte_length
            raw = msg.payload_bytes[offset:end]
            arr = np.frombuffer(raw, dtype=dt).reshape(meta.shape).copy()
            flat_tensors.append((meta.name, arr))
            offset = end

        if offset != len(msg.payload_bytes):
            raise RuntimeError("Payload byte_length mismatch")
            
        # Inject
        print("[Receiver] Injecting into model...")
        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        past_key_values = inject_kv_cache(flat_tensors, device=model_device, dtype=model_dtype)
        past_cache = DynamicCache.from_legacy_cache(past_key_values)
        
        # Continue Generation
        last_token_id = int(msg.metadata["last_token_id"])
        last_token = torch.tensor([[last_token_id]], dtype=torch.long, device=model_device)
        
        print("[Receiver] Generating continuation...")
        with torch.no_grad():
            input_ids = last_token
            past = past_cache
            generated_ids = [last_token_id]
            for _ in range(10):
                out = model(input_ids=input_ids, past_key_values=past, use_cache=True)
                next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                generated_ids.append(int(next_token.item()))
                past = out.past_key_values
                input_ids = next_token

            gen_out = torch.tensor([generated_ids], dtype=torch.long, device=model_device)
        
        full_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
        # Verify: The output will be just the completion usually if input was just 1 token + past.
        # Actually `generate` returns input+new.
        # So decoding `gen_out[0]` is just the new suffix if input was 1 token? No, it returns [last_token, new...].
        
        completion = tokenizer.decode(gen_out[0], skip_special_tokens=True)
        print(f"[Receiver] Completion: ...'{completion}'")
        
        print("\nSUCCESS: Teleportation complete.")
        
    session.close()

if __name__ == "__main__":
    main()
