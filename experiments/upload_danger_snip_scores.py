#!/usr/bin/env python3
"""
Upload danger SNIP scores (computed with GCG suffix 2) to HuggingFace Hub.
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, Repository, login
from huggingface_hub.utils import HfHubHTTPError

def main():
    parser = argparse.ArgumentParser(
        description="Upload danger SNIP scores to HuggingFace Hub"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., username/repo-name)",
    )
    parser.add_argument(
        "--scores_dir",
        type=str,
        default="out/llama2-7b-chat-hf/unstructured/wandg/danger_gcg2/wanda_score",
        help="Directory containing SNIP score pickle files",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (or use huggingface-cli login)",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Uploading Danger SNIP Scores to HuggingFace Hub")
    print("=" * 80)
    print()
    
    # Login to HuggingFace
    if args.token:
        login(token=args.token)
    else:
        print("Checking HuggingFace authentication...")
        try:
            api = HfApi()
            api.whoami()
            print("✓ Already authenticated")
        except Exception as e:
            print("ERROR: Not authenticated. Please run 'huggingface-cli login' or provide --token")
            print(f"Error: {e}")
            return
    
    # Check if scores directory exists
    scores_dir = Path(args.scores_dir)
    if not scores_dir.exists():
        print(f"ERROR: Scores directory not found: {scores_dir}")
        return
    
    # Find all pickle files
    pkl_files = list(scores_dir.glob("*.pkl"))
    if not pkl_files:
        print(f"ERROR: No .pkl files found in {scores_dir}")
        return
    
    print(f"Found {len(pkl_files)} SNIP score files")
    print(f"Directory: {scores_dir}")
    print()
    
    # Create or get repository
    api = HfApi()
    repo_id = args.repo_id
    
    print(f"Repository: {repo_id}")
    print(f"Privacy: {'Private' if args.private else 'Public'}")
    print()
    
    # Create repository if it doesn't exist
    try:
        repo_info = api.repo_info(repo_id)
        print(f"✓ Repository exists: {repo_info.repo_id}")
    except HfHubHTTPError as e:
        if e.response.status_code == 404:
            print(f"Repository does not exist. Creating new repository...")
            try:
                api.create_repo(
                    repo_id=repo_id,
                    repo_type="model",  # Use 'model' type to store files
                    private=args.private,
                )
                print(f"✓ Created repository: {repo_id}")
            except Exception as e:
                print(f"ERROR: Failed to create repository: {e}")
                return
        else:
            print(f"ERROR: Failed to check repository: {e}")
            return
    
    # Upload files
    print()
    print("Uploading files...")
    print("(This may take a while for large files)")
    print()
    
    uploaded = 0
    failed = 0
    
    for i, pkl_file in enumerate(pkl_files, 1):
        file_path = str(pkl_file)
        file_name = pkl_file.name
        file_size_mb = pkl_file.stat().st_size / (1024 * 1024)
        
        # Use relative path from wanda_score directory
        path_in_repo = f"wanda_score/{file_name}"
        
        try:
            print(f"[{i}/{len(pkl_files)}] Uploading {file_name} ({file_size_mb:.1f} MB)...", end=" ", flush=True)
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
            )
            print("✓")
            uploaded += 1
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"✗ ERROR: {error_msg}")
            if "413" in error_msg or "too large" in error_msg.lower() or "file size" in error_msg.lower():
                print(f"   → File too large for API upload. Consider using git LFS instead.")
            failed += 1
    
    print()
    print("=" * 80)
    print("Upload Summary")
    print("=" * 80)
    print(f"Total files: {len(pkl_files)}")
    print(f"Successfully uploaded: {uploaded}")
    if failed > 0:
        print(f"Failed: {failed}")
    print()
    print(f"Repository URL: https://huggingface.co/{repo_id}")
    print("=" * 80)
    
    # Create a README with metadata
    print()
    print("Creating README.md with metadata...")
    readme_content = f"""---
license: mit
tags:
  - snip-scores
  - pruning
  - llama2-7b-chat-hf
  - danger-dataset
  - gcg-suffix-2
---

# Danger SNIP Scores for Llama-2-7B-Chat-HF

This repository contains SNIP (Sparse Neural Implant Pruning) scores computed on the danger dataset using a pruned model (p=0.07, q=0.03).

## Details

- **Base Model**: Llama-2-7B-Chat-HF
- **Pruned Model**: p=0.07, q=0.03 (pruned with wandg_set_difference)
- **Dataset**: danger.txt (generated with GCG suffix 2)
- **GCG Suffix ID**: 2 (applied during SNIP score computation)
- **Method**: wandg (WANDA + gradient-based scoring)
- **Sparsity Type**: unstructured
- **Number of Files**: {len(pkl_files)}

## Usage

These SNIP scores can be used for two-stage pruning experiments where:
1. Stage 1: Prune top d% danger neurons that are NOT in top q% utility neurons
2. Stage 2: Apply standard p,q pruning on the Stage 1 pruned model

## File Structure

```
wanda_score/
  ├── W_metric_layer_0_name_model.layers.0.*.pkl
  ├── W_metric_layer_1_name_model.layers.1.*.pkl
  └── ...
```

Each `.pkl` file contains SNIP scores for a specific layer and weight matrix.

## Loading the Scores

```python
import pickle
from huggingface_hub import hf_hub_download

# Download a score file
file_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="wanda_score/W_metric_layer_0_name_model.layers.0.mlp.down_proj_weight.pkl",
    repo_type="model"
)

# Load the scores
with open(file_path, "rb") as f:
    scores = pickle.load(f)
```
"""
    
    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
        print("✓ README.md uploaded")
    except Exception as e:
        print(f"⚠ Warning: Failed to upload README.md: {e}")
    
    print()
    print("=" * 80)
    print("✓ Upload complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()

