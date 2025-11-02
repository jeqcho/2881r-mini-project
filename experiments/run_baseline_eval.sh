#!/bin/bash
# Script to run baseline evaluation on unpruned llama2-7b-chat-hf model
# Evaluates zero-shot accuracy, ASR rates, and Emergent Misalignment scores

set -e  # Exit on error

echo "=========================================="
echo "Baseline Model Evaluation"
echo "Model: llama2-7b-chat-hf (unpruned)"
echo "=========================================="
echo ""

# Configuration
MODEL="llama2-7b-chat-hf"
SPARSITY_RATIO=0  # No pruning
SAVE_DIR="out/experiments/baseline/normal_model"

# Navigate to project root
cd /workspace/2881r-mini-project

# Load environment variables from .env file
if [ -f ".env" ]; then
    echo "Loading environment variables from .env..."
    set -a  # automatically export all variables
    source .env
    set +a
    echo "✓ Loaded environment variables from .env"
    echo ""
else
    echo "WARNING: .env file not found!"
    echo "Emergent misalignment evaluation requires OpenAI API key."
    echo ""
fi

# Check if OpenAI API key is set (for emergent misalignment eval)
if [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY not set!"
    echo "Emergent misalignment evaluation requires OpenAI API key."
    echo "Please set it in .env file or export OPENAI_API_KEY='your-key'"
    echo "Continuing anyway - EM evaluation will fail if API key is not set."
    echo ""
fi

# Disable hf_transfer to avoid issues
export HF_HUB_ENABLE_HF_TRANSFER=0

# Check if we should use uv or regular python
if command -v uv &> /dev/null; then
    echo "Using uv for Python execution..."
    PYTHON_CMD="uv run python"
else
    echo "uv not found, using regular python..."
    PYTHON_CMD="python"
fi

echo "Python command: $PYTHON_CMD"
echo ""

echo "Starting baseline evaluation..."
echo "This will evaluate:"
echo "  - Perplexity (WikiText)"
echo "  - Zero-shot accuracy (6 benchmarks)"
echo "  - Attack Success Rate (ASR) with multiple variants"
echo "  - Emergent Misalignment (alignment & coherence scores)"
echo ""
echo "Results will be saved to: $SAVE_DIR"
echo ""

# Run evaluation with all metrics enabled
$PYTHON_CMD main.py \
    --model $MODEL \
    --sparsity_ratio $SPARSITY_RATIO \
    --save $SAVE_DIR \
    --eval_zero_shot \
    --eval_attack \
    --eval_emergent_misalignment \
    --save_attack_res

echo ""
echo "=========================================="
echo "✓ Baseline evaluation completed!"
echo "=========================================="
echo ""
echo "Results saved to: $SAVE_DIR"
echo "Summary log: $SAVE_DIR/log_None.txt"
echo "Attack details: $SAVE_DIR/attack_0.000000/"
echo ""

