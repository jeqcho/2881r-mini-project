#!/bin/bash
# Script to run SNIP set difference experiments for specific P,Q pairs
# Includes emergent misalignment evaluation

set -e  # Exit on error

echo "=========================================="
echo "SNIP Set Difference - P,Q Sweep"
echo "With Emergent Misalignment Evaluation"
echo "=========================================="
echo ""

# Configuration
MODEL="llama2-7b-chat-hf"
METHOD="wandg_set_difference"
TYPE="unstructured"
SPARSITY_RATIO=0.5
PRUNE_DATA="align"  # Safety dataset

# Navigate to project root
cd /workspace/projects/2881r-mini-project

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate
echo "Using Python: $(which python)"
echo ""

# Disable hf_transfer to avoid issues
export HF_HUB_ENABLE_HF_TRANSFER=0

# Load environment variables from .env file
if [ -f ".env" ]; then
    echo "Loading environment variables from .env..."
    set -a  # automatically export all variables
    source .env
    set +a
    echo "✓ Loaded environment variables from .env"
    echo ""
fi

# Check if SNIP scores exist
SCORE_UTILITY="out/llama2-7b-chat-hf/unstructured/wandg/alpaca_cleaned_no_safety/wanda_score"
SCORE_SAFETY="out/llama2-7b-chat-hf/unstructured/wandg/align/wanda_score"

if [ ! -d "$SCORE_UTILITY" ] || [ ! -d "$SCORE_SAFETY" ]; then
    echo "ERROR: SNIP scores not found!"
    echo "Please run 'bash experiments/dump_scores.sh' first"
    echo ""
    echo "Expected locations:"
    echo "  - $SCORE_UTILITY"
    echo "  - $SCORE_SAFETY"
    exit 1
fi

# Check if OpenAI API key is set (for emergent misalignment eval)
if [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY not set!"
    echo "Emergent misalignment evaluation requires OpenAI API key."
    echo "Please set it with: export OPENAI_API_KEY='your-key'"
    echo "Continuing anyway - EM evaluation will fail if API key is not set."
    echo ""
fi

echo "Starting P,Q sweep experiments..."
echo ""

# P,Q pairs to run
declare -a PQ_PAIRS=(
    "0.01 0.01"
    "0.02 0.01"
    "0.04 0.02"
    "0.07 0.03"
    "0.03 0.02"
    "0.04 0.04"
    "0.05 0.05"
    "0.06 0.05"
    "0.06 0.06"
    "0.09 0.08"
)

# Count total experiments
TOTAL=${#PQ_PAIRS[@]}

# Loop through P,Q pairs
for i in "${!PQ_PAIRS[@]}"; do
    # Parse P and Q values
    read -r P Q <<< "${PQ_PAIRS[$i]}"
    
    # Format for directory name (e.g., 0.01 -> 0.01)
    SAVE_DIR="out/experiments/pq_sweep/p_${P}_q_${Q}"
    
    EXP_NUM=$((i + 1))
    
    echo "=========================================="
    echo "Experiment $EXP_NUM/$TOTAL: P=$P, Q=$Q"
    echo "Save directory: $SAVE_DIR"
    echo "=========================================="
    
    # Run pruning, zero-shot eval, ASR evaluation, and emergent misalignment evaluation
    python main.py \
        --model $MODEL \
        --prune_method $METHOD \
        --sparsity_ratio $SPARSITY_RATIO \
        --prune_data $PRUNE_DATA \
        --p $P \
        --q $Q \
        --sparsity_type $TYPE \
        --save $SAVE_DIR \
        --eval_zero_shot \
        --eval_attack \
        --eval_emergent_misalignment \
        --save_attack_res
    
    # Clear GPU memory before next experiment
    echo "Clearing GPU cache..."
    python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')" 2>/dev/null || true
    sleep 2
    
    echo ""
    echo "✓ Experiment $EXP_NUM/$TOTAL completed: P=$P, Q=$Q"
    echo ""
    
    # Small delay between experiments
    sleep 2
done

echo "=========================================="
echo "✓ All $TOTAL experiments completed!"
echo "=========================================="
echo ""
echo "Results saved to: out/experiments/pq_sweep/"
echo ""
echo "Next step: Parse results"
echo "  python experiments/collect_pq_results.py"
echo ""

