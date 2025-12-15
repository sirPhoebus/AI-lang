import sys
import os
import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from latlink_session.session import LatentLinkSession, SessionRole
from latlink_protocol.protocol import MessageKind, DType, HandshakeData, TensorMeta, LatentLinkMessage
from latlink_integrations_hf.hf_integration import get_model_signature, capture_kv_cache

def main():
    print("[Sender] Loading model...")
    model_name = os.environ["LATLINK_MODEL_NAME"]
    revision = os.environ.get("LATLINK_MODEL_REVISION")
    if revision:
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, use_safetensors=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True)
    model.eval()
    
    print("[Sender] Connecting to LatentLink...")
    session = LatentLinkSession(SessionRole.SENDER)
    session.start()
    
    # Handshake
    print("[Sender] Sending Handshake...")
    sig = get_model_signature(model, tokenizer)
    hs_data = HandshakeData(**sig)
    session.send_handshake(hs_data)
    
    # Generate/Precompute
    prompt = os.environ["LATLINK_PROMPT"]
    print(f"[Sender] Processing prompt: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
        # outputs.past_key_values is our gold
        
    # Extract
    kv_list = capture_kv_cache(outputs.past_key_values)
    
    # Prepare message
    tensors_meta = []
    payload = b''
    
    for name, arr in kv_list:
        meta = TensorMeta(
            name=name,
            dtype=int(DType.from_numpy(arr.dtype)),
            shape=list(arr.shape),
            byte_length=arr.nbytes
        )
        tensors_meta.append(meta)
        payload += arr.tobytes()
        
    print(f"[Sender] Sending KV Bundle: {len(kv_list)} tensors, {len(payload)} bytes")
    
    # Send
    # Using low-level construction to include tensors
    msg = LatentLinkMessage(
        session_id=session.session_id,
        msg_id=session.msg_counter,
        kind=MessageKind.KV_CACHE_DELTA,
        metadata={"last_token_id": int(inputs.input_ids[0, -1].item()), "token_count": int(inputs.input_ids.shape[1])},
        tensors=tensors_meta,
        payload_bytes=payload
    )
    session.msg_counter += 1
    session.shm_out.write_bytes(msg.pack())
    
    print("[Sender] Payload sent. Closing in 5s...")
    time.sleep(5)
    session.close() # Actually sender doesn't need to unlink
    print("[Sender] Done.")

if __name__ == "__main__":
    main()
