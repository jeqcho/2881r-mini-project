#!/bin/bash
# Download danger SNIP scores from HuggingFace Hub

set -e

REPO_ID="jeqcho/llama2-7b-chat-danger-snip-scores-gcg2"
# Get project root from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET_DIR="$PROJECT_ROOT/out/llama2-7b-chat-hf/unstructured/wandg/danger_gcg2/wanda_score"

echo "=" | head -c 80; echo
echo "Downloading Danger SNIP Scores from HuggingFace Hub"
echo "=" | head -c 80; echo
echo ""
echo "Repository: $REPO_ID"
echo "Target directory: $TARGET_DIR"
echo ""

# Navigate to project root
cd "$PROJECT_ROOT"

# Create target directory
mkdir -p "$TARGET_DIR"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Check HuggingFace authentication
echo "Checking HuggingFace authentication..."
python3 << PYEOF
from huggingface_hub import HfApi, HfFolder
import os

api = HfApi()
token = HfFolder.get_token() or os.environ.get('HF_TOKEN', '')

if not token:
    print("ERROR: HuggingFace token not found.")
    print("Please run 'huggingface-cli login' or set HF_TOKEN environment variable")
    exit(1)

try:
    api.whoami()
    print("✓ Authenticated to HuggingFace")
except Exception as e:
    print(f"ERROR: Authentication failed: {e}")
    exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo ""
    echo "Please authenticate: huggingface-cli login"
    exit 1
fi

echo ""
echo "Downloading SNIP score files..."
echo "This may take a while (~13GB total)..."
echo ""

# Download all .pkl files from the repo
python3 << PYEOF
from huggingface_hub import HfApi, list_repo_files, hf_hub_download
import os
from pathlib import Path

repo_id = "$REPO_ID"
target_dir = "$TARGET_DIR"

print(f"Downloading from {repo_id}...")
print(f"Target: {target_dir}")

# Get list of all files in the wanda_score directory
api = HfApi()
files = list_repo_files(repo_id=repo_id, repo_type="model")
pkl_files = [f for f in files if f.startswith("wanda_score/") and f.endswith(".pkl")]

print(f"\nFound {len(pkl_files)} SNIP score files")
print("Downloading files...")

for i, file_path in enumerate(pkl_files, 1):
    filename = os.path.basename(file_path)
    dest = os.path.join(target_dir, filename)
    
    # Download file
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=file_path,
        repo_type="model",
        local_dir=None  # Download to default cache first
    )
    
    # Copy to target directory
    from shutil import copy2
    copy2(downloaded_path, dest)
    print(f"  [{i}/{len(pkl_files)}] ✓ {filename}")

print(f"\n✓ All {len(pkl_files)} files downloaded to {target_dir}")
PYEOF

echo ""
echo "=" | head -c 80; echo
echo "✓ Download complete!"
echo "=" | head -c 80; echo
echo ""
echo "Files saved to: $TARGET_DIR"
echo "Total files: $(ls -1 "$TARGET_DIR"/*.pkl 2>/dev/null | wc -l)"
echo ""

