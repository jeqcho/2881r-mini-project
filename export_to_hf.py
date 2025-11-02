#!/usr/bin/env python3
"""
Script to export pruned model with p=0.07 and q=0.03 to HuggingFace Hub
"""

import argparse
import os
import torch
# Disable hf_transfer to avoid issues
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from transformers import AutoTokenizer, AutoModelForCausalLM
from main import get_llm
from lib.prune import prune_wandg_set_difference

# Configuration matching run_pq_sweep.sh
MODEL = "llama2-7b-chat-hf"
METHOD = "wandg_set_difference"
TYPE = "unstructured"
SPARSITY_RATIO = 0.5
PRUNE_DATA = "align"
P = 0.07
Q = 0.03

modeltype2path = {
    "llama2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
    "llama2-7b-hf": "meta-llama/Llama-2-7b-hf",
    "llama2-13b-hf": "meta-llama/Llama-2-13b-hf",
}


def main():
    parser = argparse.ArgumentParser(
        description="Export pruned model to HuggingFace Hub"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., username/model-name)",
    )
    parser.add_argument(
        "--cache_dir", default="llm_weights", type=str, help="Model cache directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (default: cuda:0)",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="Local directory to save model before uploading (optional)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model: {MODEL}")
    model = get_llm(MODEL, cache_dir=args.cache_dir)
    model.eval()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        modeltype2path[MODEL], use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    # Check if SNIP scores exist
    score_utility = "out/llama2-7b-chat-hf/unstructured/wandg/alpaca_cleaned_no_safety/wanda_score"
    score_safety = "out/llama2-7b-chat-hf/unstructured/wandg/align/wanda_score"

    if not os.path.exists(score_utility) or not os.path.exists(score_safety):
        print("ERROR: SNIP scores not found!")
        print("Please run 'bash experiments/dump_scores.sh' first")
        print(f"Expected locations:")
        print(f"  - {score_utility}")
        print(f"  - {score_safety}")
        return

    print(f"\nPruning model with p={P}, q={Q}...")
    print(f"Method: {METHOD}")
    print(f"Sparsity ratio: {SPARSITY_RATIO}")
    print(f"Prune data: {PRUNE_DATA}")

    # Create args namespace-like object for pruning function
    class Args:
        def __init__(self):
            self.model = MODEL
            self.sparsity_ratio = SPARSITY_RATIO
            self.sparsity_type = TYPE
            self.prune_method = METHOD
            self.prune_data = PRUNE_DATA
            self.p = P
            self.q = Q
            self.nsamples = 128
            self.seed = 0
            self.use_diff = False
            self.recover_from_base = False
            self.prune_part = False
            self.dump_wanda_score = False

    prune_args = Args()

    # Perform pruning
    prune_wandg_set_difference(
        prune_args,
        model,
        tokenizer,
        model_base=None,
        device=device,
        prune_n=0,
        prune_m=0,
        prune_data=PRUNE_DATA,
        p=P,
        q=Q,
    )

    print("\n✓ Pruning completed!")
    print(f"\nSaving model and uploading to HuggingFace Hub: {args.repo_id}")

    # Save locally first if specified
    if args.local_dir:
        print(f"Saving model locally to {args.local_dir}...")
        os.makedirs(args.local_dir, exist_ok=True)
        model.save_pretrained(args.local_dir)
        tokenizer.save_pretrained(args.local_dir)
        print(f"✓ Model saved locally to {args.local_dir}")

    # Upload to HuggingFace Hub
    print(f"\nUploading to HuggingFace Hub...")
    model.push_to_hub(
        args.repo_id,
        private=args.private,
        max_shard_size="5GB",
    )
    tokenizer.push_to_hub(args.repo_id)

    print(f"\n✓ Successfully exported model to: https://huggingface.co/{args.repo_id}")
    print(f"\nModel details:")
    print(f"  - Base model: {MODEL}")
    print(f"  - Pruning method: {METHOD}")
    print(f"  - p={P}, q={Q}")
    print(f"  - Sparsity ratio: {SPARSITY_RATIO}")
    print(f"  - Prune data: {PRUNE_DATA}")


if __name__ == "__main__":
    main()

