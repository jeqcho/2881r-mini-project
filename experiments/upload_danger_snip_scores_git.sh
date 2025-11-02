#!/bin/bash
# Upload danger SNIP scores to HuggingFace using git LFS (required for large files)

set -e

REPO_ID="jeqcho/llama2-7b-chat-danger-snip-scores-gcg2"
PROJECT_ROOT="/workspace/projects/2881r-mini-project"
SCORES_DIR="$PROJECT_ROOT/out/llama2-7b-chat-hf/unstructured/wandg/danger_gcg2/wanda_score"
TEMP_REPO_DIR="/tmp/danger_snip_upload"

echo "=" | head -c 80; echo
echo "Uploading Danger SNIP Scores to HuggingFace Hub via Git LFS"
echo "=" | head -c 80; echo
echo

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo "ERROR: git-lfs is not installed. Installing..."
    git lfs install || {
        echo "ERROR: Failed to install git-lfs. Please install it manually:"
        echo "  apt-get update && apt-get install -y git-lfs && git lfs install"
        exit 1
    }
fi

# Initialize git lfs
git lfs install

# Clone or create repo
if [ -d "$TEMP_REPO_DIR" ]; then
    rm -rf "$TEMP_REPO_DIR"
fi

# Clone repo (create first with Python if needed)
echo "Checking repository..."
python3 << PYEOF
from huggingface_hub import HfApi
api = HfApi()
repo_id = "$REPO_ID"
try:
    api.repo_info(repo_id)
    print(f"Repository exists: {repo_id}")
except:
    print(f"Creating repository: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", private=False)
    print(f"✓ Created: {repo_id}")
PYEOF

# Get HuggingFace token
HF_TOKEN=$(python3 -c "from huggingface_hub import HfFolder; import os; token = HfFolder.get_token() or os.environ.get('HF_TOKEN', ''); print(token)" 2>/dev/null || echo "")

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HuggingFace token not found. Please set HF_TOKEN environment variable or login with 'huggingface-cli login'"
    exit 1
fi

# Configure git with token
echo "Cloning repository..."
git clone https://oauth2:${HF_TOKEN}@huggingface.co/$REPO_ID "$TEMP_REPO_DIR"

cd "$TEMP_REPO_DIR"

# Configure git LFS for pickle files
git lfs track "*.pkl"

# Copy files
echo "Copying SNIP score files..."
mkdir -p wanda_score
cp "$SCORES_DIR"/*.pkl wanda_score/

# Create README if it doesn't exist
if [ ! -f README.md ]; then
    cat > README.md << 'EOF'
---
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
- **Number of Files**: 224

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
    repo_id="jeqcho/llama2-7b-chat-danger-snip-scores-gcg2",
    filename="wanda_score/W_metric_layer_0_name_model.layers.0.mlp.down_proj_weight.pkl",
    repo_type="model"
)

# Load the scores
with open(file_path, "rb") as f:
    scores = pickle.load(f)
```
EOF
fi

# Commit and push
echo "Staging files..."
git add .gitattributes
git add wanda_score/*.pkl
git add README.md

echo "Committing..."
git commit -m "Add danger SNIP scores (GCG suffix 2)" || echo "No changes to commit"

echo "Pushing to HuggingFace..."
git push

echo
echo "=" | head -c 80; echo
echo "✓ Upload complete!"
echo "Repository URL: https://huggingface.co/$REPO_ID"
echo "=" | head -c 80; echo

cd -
rm -rf "$TEMP_REPO_DIR"

