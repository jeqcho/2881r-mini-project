#!/usr/bin/env python3
"""
SNIP Score Calculator Module

This module handles dumping SNIP scores for utility and safety datasets.
SNIP scores are required for the wandg_set_difference pruning method.
"""

import os
import subprocess
import sys
from pathlib import Path


def dump_snip_scores(
    safety_dataset="align",
    utility_dataset="alpaca_cleaned_no_safety",
    model="llama2-7b-chat-hf",
    sparsity_ratio=0.5,
    force_recompute=False
):
    """
    Dump SNIP scores for both utility and safety datasets.
    
    Args:
        safety_dataset: Dataset to use for safety SNIP scores (default: "align")
        utility_dataset: Dataset to use for utility SNIP scores (default: "alpaca_cleaned_no_safety")
        model: Model name (default: "llama2-7b-chat-hf")
        sparsity_ratio: Sparsity ratio for pruning (default: 0.5)
        force_recompute: If True, recompute even if scores exist (default: False)
    
    Returns:
        dict: Paths to utility and safety score directories
    """
    project_root = Path(__file__).parent.parent.parent
    
    # Define output paths (main.py writes to out/ directory)
    utility_score_path = project_root / f"out/{model}/unstructured/wandg/{utility_dataset}"
    safety_score_path = project_root / f"out/{model}/unstructured/wandg/{safety_dataset}"
    
    # Check if scores already exist
    utility_exists = (utility_score_path / "wanda_score").exists()
    safety_exists = (safety_score_path / "wanda_score").exists()
    
    if utility_exists and safety_exists and not force_recompute:
        print("✓ SNIP scores already exist. Skipping computation.")
        print(f"  Utility: {utility_score_path}")
        print(f"  Safety:  {safety_score_path}")
        return {
            "utility": str(utility_score_path),
            "safety": str(safety_score_path)
        }
    
    print("=" * 60)
    print("Dumping SNIP Scores")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Utility Dataset: {utility_dataset}")
    print(f"Safety Dataset:  {safety_dataset}")
    print()
    
    # Dump utility scores
    if not utility_exists or force_recompute:
        print(f"[1/2] Dumping SNIP scores for UTILITY dataset ({utility_dataset})...")
        print("-" * 60)
        
        cmd = [
            sys.executable,
            str(project_root / "main.py"),
            "--model", model,
            "--prune_method", "wandg",
            "--prune_data", utility_dataset,
            "--sparsity_ratio", str(sparsity_ratio),
            "--sparsity_type", "unstructured",
            "--save", str(utility_score_path),
            "--dump_wanda_score"
        ]
        
        result = subprocess.run(cmd, cwd=project_root, capture_output=False)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to dump utility SNIP scores. Exit code: {result.returncode}")
        
        print("✓ Utility scores dumped successfully")
        print()
    
    # Dump safety scores
    if not safety_exists or force_recompute:
        print(f"[2/2] Dumping SNIP scores for SAFETY dataset ({safety_dataset})...")
        print("-" * 60)
        
        cmd = [
            sys.executable,
            str(project_root / "main.py"),
            "--model", model,
            "--prune_method", "wandg",
            "--prune_data", safety_dataset,
            "--sparsity_ratio", str(sparsity_ratio),
            "--sparsity_type", "unstructured",
            "--save", str(safety_score_path),
            "--dump_wanda_score"
        ]
        
        result = subprocess.run(cmd, cwd=project_root, capture_output=False)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to dump safety SNIP scores. Exit code: {result.returncode}")
        
        print("✓ Safety scores dumped successfully")
        print()
    
    print("=" * 60)
    print("✓ All SNIP scores dumped successfully!")
    print("=" * 60)
    print()
    
    return {
        "utility": str(utility_score_path),
        "safety": str(safety_score_path)
    }


def check_snip_scores_exist(model="llama2-7b-chat-hf", 
                            safety_dataset="align",
                            utility_dataset="alpaca_cleaned_no_safety"):
    """
    Check if SNIP scores already exist.
    
    Returns:
        bool: True if both utility and safety scores exist
    """
    project_root = Path(__file__).parent.parent.parent
    
    utility_score_path = project_root / f"out/{model}/unstructured/wandg/{utility_dataset}/wanda_score"
    safety_score_path = project_root / f"out/{model}/unstructured/wandg/{safety_dataset}/wanda_score"
    
    return utility_score_path.exists() and safety_score_path.exists()


if __name__ == "__main__":
    # Test the module
    import argparse
    
    parser = argparse.ArgumentParser(description="Dump SNIP scores for datasets")
    parser.add_argument("--safety-dataset", type=str, default="align",
                       help="Safety dataset for SNIP scores")
    parser.add_argument("--utility-dataset", type=str, default="alpaca_cleaned_no_safety",
                       help="Utility dataset for SNIP scores")
    parser.add_argument("--model", type=str, default="llama2-7b-chat-hf",
                       help="Model name")
    parser.add_argument("--force", action="store_true",
                       help="Force recompute even if scores exist")
    
    args = parser.parse_args()
    
    paths = dump_snip_scores(
        safety_dataset=args.safety_dataset,
        utility_dataset=args.utility_dataset,
        model=args.model,
        force_recompute=args.force
    )
    
    print("Score paths:")
    print(f"  Utility: {paths['utility']}")
    print(f"  Safety:  {paths['safety']}")

