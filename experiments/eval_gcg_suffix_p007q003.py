#!/usr/bin/env python3
"""
Quick evaluation script to determine which GCG suffix gives best ASR for p0.07 q0.03 model.
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.eval import eval_attack

modeltype2path = {
    "llama2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
    "llama2-7b-hf": "meta-llama/Llama-2-7b-hf",
    "llama2-13b-hf": "meta-llama/Llama-2-13b-hf",
}

def main():
    print("=" * 60)
    print("Finding Best GCG Suffix for p0.07 q0.03 Model")
    print("=" * 60)
    print()
    
    MODEL = "llama2-7b-chat-hf"
    
    # Find pruned model path
    temp_symlink = "temp/wandg_set_difference_usediff_False_recover_False"
    pruned_model_path = None
    
    if os.path.exists(temp_symlink):
        if os.path.islink(temp_symlink):
            actual_path = os.path.realpath(temp_symlink)
            if os.path.exists(actual_path):
                pruned_model_path = actual_path
                print(f"Found pruned model: {pruned_model_path}")
    
    if pruned_model_path is None:
        print("ERROR: Pruned model not found!")
        return
    
    print(f"Loading model...")
    vllm_model = LLM(
        model=pruned_model_path,
        tokenizer=modeltype2path[MODEL],
        dtype='bfloat16',
        swap_space=16,
    )
    
    if vllm_model.llm_engine.tokenizer.pad_token is None:
        vllm_model.llm_engine.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        modeltype2path[MODEL], use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    print("✓ Model loaded")
    print()
    
    # Evaluate attack with GCG
    save_dir = "out/experiments/pq_sweep/p_0.07_q_0.03"
    os.makedirs(save_dir, exist_ok=True)
    save_attackpath = os.path.join(save_dir, "attack_0.500000")
    os.makedirs(save_attackpath, exist_ok=True)
    
    print("Evaluating GCG attack...")
    result = eval_attack(
        vllm_model,
        tokenizer,
        num_sampled=1,
        add_sys_prompt=False,
        gcg=True,
        do_sample=False,
        save_attack_res=True,
        include_inst=True,
        filename=os.path.join(save_attackpath, "gcg.jsonl"),
    )
    
    if isinstance(result, tuple):
        score, best_suffix_idx = result
        print()
        print("=" * 60)
        print(f"Best GCG Suffix: {best_suffix_idx}")
        print(f"ASR Score: {score:.4f}")
        print("=" * 60)
        
        # Save to a file for reference
        with open(os.path.join(save_dir, "best_gcg_suffix.txt"), "w") as f:
            f.write(f"{best_suffix_idx}\n")
            f.write(f"ASR: {score:.4f}\n")
        
        print(f"Saved best suffix index to: {save_dir}/best_gcg_suffix.txt")
    else:
        print(f"ASR Score: {result:.4f}")
        print("(Best suffix not tracked in this version)")
    
    del vllm_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

