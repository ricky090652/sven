import os
import torch
import argparse
import sys
from types import SimpleNamespace
from sven_peft.model import load_model, ADAPTER_NAME
from sven_peft.constant import ADAPTER_NAME as CONST_ADAPTER_NAME

def run_test():
    args = SimpleNamespace()
    args.n_prefix_token = 6
    args.dropout = 0.1
    args.n_gpu = 1
    args.device = 'cuda'
    args.pretrain_dir = 'Qwen/Qwen2.5-Coder-1.5B'
    args.model_dir = 'trained/qwen_6token_sven_v1_fixed/checkpoint-epoch-5'

    print("Loading model...")
    sys.stdout.flush()
    model, tokenizer = load_model('prefix', args.model_dir, False, args)
    device = torch.device(args.device if args.n_gpu > 0 else "cpu")
    model.to(device)
    model.eval()
    
    prompt = "void* my_memcpy(void* dest, const void* src, size_t n) {\n"
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    print(f"Prompt: {repr(prompt)}")
    print(f"Input IDs shape: {input_ids.shape}")
    sys.stdout.flush()

    print("\n--- Test 1: Prefix ENABLED ---")
    model.set_adapter(CONST_ADAPTER_NAME)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=40, do_sample=True, temperature=0.4, top_p=0.95, pad_token_id=tokenizer.pad_token_id)
    print(f"Output IDs shape: {out.shape}")
    print("Full Decoded Output:")
    print(tokenizer.decode(out[0]))
    gen_tokens = out[0][input_ids.shape[1]:]
    print(f"Generated tokens (count={len(gen_tokens)}): {gen_tokens}")
    sys.stdout.flush()

    print("\n--- Test 2: Prefix DISABLED ---")
    with torch.no_grad():
        with model.disable_adapter():
            out = model.generate(input_ids, max_new_tokens=40, do_sample=True, temperature=0.4, top_p=0.95, pad_token_id=tokenizer.pad_token_id)
    print(f"Output IDs shape: {out.shape}")
    print("Full Decoded Output:")
    print(tokenizer.decode(out[0]))
    gen_tokens = out[0][input_ids.shape[1]:]
    print(f"Generated tokens (count={len(gen_tokens)}): {gen_tokens}")
    sys.stdout.flush()

if __name__ == "__main__":
    run_test()
