#!/bin/bash
# Script to dump SNIP scores for both utility and safety datasets
# These scores are required for wandg_set_difference pruning method

set -e  # Exit on error

echo "=========================================="
echo "SNIP Score Dumping for Set Difference"
echo "=========================================="
echo ""

# Parse command line arguments
SAFETY_DATASET="align"
UTILITY_DATASET="alpaca_cleaned_no_safety"
MODEL="llama2-7b-chat-hf"
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --safety-dataset)
            SAFETY_DATASET="$2"
            shift 2
            ;;
        --utility-dataset)
            UTILITY_DATASET="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--safety-dataset DATASET] [--utility-dataset DATASET] [--model MODEL] [--force]"
            exit 1
            ;;
    esac
done

# Configuration
METHOD="wandg"  # SNIP method in the codebase
TYPE="unstructured"
SPARSITY_RATIO=0.5

# Navigate to project root
cd /workspace/projects/2881r-mini-project

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
    echo "Using Python: $(which python)"
    echo ""
fi

# Disable hf_transfer to avoid issues
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Utility Dataset: $UTILITY_DATASET"
echo "  Safety Dataset: $SAFETY_DATASET"
echo "  Force Recompute: $FORCE"
echo ""

# Check if scores already exist
UTILITY_PATH="out/${MODEL}/unstructured/wandg/${UTILITY_DATASET}/wanda_score"
SAFETY_PATH="out/${MODEL}/unstructured/wandg/${SAFETY_DATASET}/wanda_score"

if [ "$FORCE" = false ] && [ -d "$UTILITY_PATH" ] && [ -d "$SAFETY_PATH" ]; then
    echo "✓ SNIP scores already exist. Skipping computation."
    echo "  Utility: $UTILITY_PATH"
    echo "  Safety:  $SAFETY_PATH"
    echo ""
    echo "Use --force to recompute."
    exit 0
fi

echo "[1/2] Dumping SNIP scores for UTILITY dataset ($UTILITY_DATASET)..."
echo "----------------------------------------------------------------------"
python main.py \
    --model $MODEL \
    --prune_method $METHOD \
    --prune_data $UTILITY_DATASET \
    --sparsity_ratio $SPARSITY_RATIO \
    --sparsity_type $TYPE \
    --save out/${MODEL}/unstructured/wandg/${UTILITY_DATASET}/ \
    --dump_wanda_score

echo ""
echo "[2/2] Dumping SNIP scores for SAFETY dataset ($SAFETY_DATASET)..."
echo "----------------------------------------------------------------------"
python main.py \
    --model $MODEL \
    --prune_method $METHOD \
    --prune_data $SAFETY_DATASET \
    --sparsity_ratio $SPARSITY_RATIO \
    --sparsity_type $TYPE \
    --save out/${MODEL}/unstructured/wandg/${SAFETY_DATASET}/ \
    --dump_wanda_score

echo ""
echo "=========================================="
echo "✓ Score dumping completed successfully!"
echo "=========================================="
echo ""
echo "Scores saved to:"
echo "  - out/${MODEL}/unstructured/wandg/${UTILITY_DATASET}/"
echo "  - out/${MODEL}/unstructured/wandg/${SAFETY_DATASET}/"
echo ""
echo "Next step: Run SNIP analysis"
echo "  python experiments/run_snip_analysis.py --mode 1"
echo ""

