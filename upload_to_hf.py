#!/usr/bin/env python3
"""
Script to upload a model directory to Hugging Face Hub
"""
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def upload_directory_to_hf(local_dir: str, repo_id: str, private: bool = True):
    """
    Upload a directory to Hugging Face Hub
    
    Args:
        local_dir: Path to the local directory to upload
        repo_id: Repository ID in format "username/repo-name"
        private: Whether the repository should be private
    """
    local_path = Path(local_dir)
    if not local_path.exists():
        raise ValueError(f"Directory {local_dir} does not exist")
    
    print(f"Uploading {local_dir} to Hugging Face Hub repository: {repo_id}")
    
    # Initialize API
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        print(f"✓ Repository {repo_id} is ready")
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload all files
    print("\nUploading files...")
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=[".git", "__pycache__", "*.pyc"]
    )
    
    print(f"\n✓ Successfully uploaded to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    # Default values
    local_dir = "/workspace/projects/2881r-mini-project/experiments/neuron/output/mode1_predefined_align_20251101_135213/p_0.03_q_0.02/copy"
    repo_id = "jeqcho/llama2-7b-chat-mode1-p0.03-q0.02"
    
    # Allow override via command line
    if len(sys.argv) > 1:
        repo_id = sys.argv[1]
    if len(sys.argv) > 2:
        local_dir = sys.argv[2]
    
    upload_directory_to_hf(local_dir, repo_id, private=True)



