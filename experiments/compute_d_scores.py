#!/usr/bin/env python3
"""
Script to compute SNIP scores 'd' on the p0.07 q0.03 pruned model using danger.txt dataset.
These scores will be used for the first stage of two-stage pruning (d=p).
"""

import os
import sys
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# get_llm is in main.py
from main import get_llm

modeltype2path = {
    "llama2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
    "llama2-7b-hf": "meta-llama/Llama-2-7b-hf",
    "llama2-13b-hf": "meta-llama/Llama-2-13b-hf",
}
from lib.model_wrapper import prune_wandg

def main():
    parser = argparse.ArgumentParser(description="Compute SNIP scores d on pruned model")
    parser.add_argument("--model", type=str, default="llama2-7b-chat-hf")
    parser.add_argument("--pruned_model_path", type=str, 
                       default="out/experiments/pq_sweep/p_0.07_q_0.03/tmp.ckpt",
                       help="Path to pruned model (or temp symlink)")
    parser.add_argument("--save", type=str, 
                       default="out/llama2-7b-chat-hf/unstructured/wandg/danger/",
                       help="Directory to save SNIP scores")
    parser.add_argument("--sparsity_ratio", type=float, default=0.5)
    parser.add_argument("--sparsity_type", type=str, default="unstructured")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gcg_suffix_id", type=int, default=None,
                       help="GCG suffix ID to apply (0, 1, or 2). If not provided, uses prompts as-is.")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Computing SNIP scores 'd' on pruned model")
    print("Dataset: danger.txt")
    print("=" * 60)
    print()
    
    # Check if danger.txt exists
    if not os.path.exists("data/danger.txt"):
        print("ERROR: data/danger.txt not found!")
        print("Please run experiments/generate_danger.py first to generate danger.txt")
        return
    
    # Find pruned model path - check temp symlink first
    pruned_model_path = None
    
    # Check temp symlink (created during attack evaluation)
    temp_symlink = "temp/wandg_set_difference_usediff_False_recover_False"
    if os.path.exists(temp_symlink):
        if os.path.islink(temp_symlink):
            actual_path = os.path.realpath(temp_symlink)
            if os.path.exists(actual_path):
                pruned_model_path = actual_path
                print(f"Found pruned model via symlink: {temp_symlink} -> {actual_path}")
    
    # If not found, check the direct path
    if pruned_model_path is None and os.path.exists(args.pruned_model_path):
        pruned_model_path = args.pruned_model_path
        if os.path.islink(pruned_model_path):
            actual_path = os.path.realpath(pruned_model_path)
            if os.path.exists(actual_path):
                pruned_model_path = actual_path
                print(f"Following symlink to: {actual_path}")
    
    if pruned_model_path is None or not os.path.exists(pruned_model_path):
        print(f"ERROR: Pruned model not found!")
        print(f"Checked: {temp_symlink}")
        print(f"Checked: {args.pruned_model_path}")
        print("Please ensure the p0.07 q0.03 experiment has been run.")
        return
    
    print(f"Loading pruned model from: {pruned_model_path}")
    print("(This may take a few minutes)")
    
    # Load the pruned model
    model = AutoModelForCausalLM.from_pretrained(
        pruned_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    model.seqlen = model.config.max_position_embeddings
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        modeltype2path[args.model], use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    print()
    
    # Create save directory
    os.makedirs(args.save, exist_ok=True)
    
    # Create args object for prune_wandg
    class PruneArgs:
        def __init__(self):
            self.model = args.model
            self.sparsity_ratio = args.sparsity_ratio
            self.sparsity_type = args.sparsity_type
            self.prune_method = "wandg"
            self.prune_data = "danger"
            self.nsamples = args.nsamples
            self.seed = args.seed
            self.use_diff = False
            self.recover_from_base = False
            self.prune_part = False
            self.dump_wanda_score = True  # Just dump scores, don't prune
            self.neg_prune = False
            self.use_variant = False
            self.disentangle = True  # Use disentangle mode for danger dataset
            self.save = args.save
            self.gcg_suffix_id = args.gcg_suffix_id  # Pass GCG suffix ID
    
    prune_args = PruneArgs()
    
    print(f"Computing SNIP scores on danger dataset...")
    print(f"Dataset: danger.txt")
    print(f"Save directory: {args.save}")
    print(f"Using disentangle mode: {prune_args.disentangle}")
    if args.gcg_suffix_id is not None:
        print(f"Applying GCG suffix {args.gcg_suffix_id} to prompts")
    print()
    
    # Compute and save SNIP scores
    prune_wandg(
        prune_args,
        model,
        tokenizer,
        model_base=None,
        device=device,
        prune_n=0,
        prune_m=0,
        prune_data="danger",
    )
    
    print()
    print("=" * 60)
    print("✓ SNIP scores 'd' computed successfully!")
    print(f"Scores saved to: {args.save}")
    print("=" * 60)

if __name__ == "__main__":
    main()

