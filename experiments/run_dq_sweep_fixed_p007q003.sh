#!/bin/bash
# Script to run DQ then PQ two-stage pruning experiments
# Stage 1: Prune (d,q) with danger/utility set-difference
# Stage 2: Prune (0.07, 0.03) with utility/align set-difference
# Evaluates after each stage

set -e  # Exit on error

echo "=========================================="
echo "DQ Then PQ Two-Stage Pruning Experiments"
echo "With Intermediate Evaluation"
echo "=========================================="
echo ""

# Configuration
MODEL="llama2-7b-chat-hf"
METHOD="wandg_set_difference"
TYPE="unstructured"
SPARSITY_RATIO=0.5
PRUNE_DATA="align"  # Safety dataset for Stage 2
OUTPUT_DIR="out/experiments/dq_sweep_fixed_p007q003"

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
SCORE_DANGER="out/llama2-7b-chat-hf/unstructured/wandg/danger/wanda_score"
SCORE_UTILITY="out/llama2-7b-chat-hf/unstructured/wandg/alpaca_cleaned_no_safety/wanda_score"
SCORE_SAFETY="out/llama2-7b-chat-hf/unstructured/wandg/align/wanda_score"

if [ ! -d "$SCORE_DANGER" ]; then
    echo "ERROR: Danger scores not found!"
    echo "Please run 'python experiments/compute_d_scores.py' first"
    echo ""
    echo "Expected location: $SCORE_DANGER"
    exit 1
fi

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
    echo "Continuing anyway - EM evaluation will fail if API key is not set."
    echo ""
fi

echo "Starting DQ then PQ sweep experiments..."
echo ""

# D,Q pairs to run (Stage 1 parameters)
declare -a DQ_PAIRS=(
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
TOTAL=${#DQ_PAIRS[@]}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Loop through D,Q pairs
for i in "${!DQ_PAIRS[@]}"; do
    # Parse D and Q values
    read -r D Q <<< "${DQ_PAIRS[$i]}"
    
    # Format for directory name (e.g., 0.01 -> 0.01)
    EXP_DIR="$OUTPUT_DIR/d_${D}_q_${Q}"
    EXP_NUM=$((i + 1))
    
    echo "=========================================="
    echo "Experiment $EXP_NUM/$TOTAL: d=$D, q=$Q"
    echo "Save directory: $EXP_DIR"
    echo "=========================================="
    
    # Create experiment directory
    mkdir -p "$EXP_DIR"
    
    # STAGE 1: Prune (d,q) and evaluate
    echo ""
    echo "--- STAGE 1: Pruning (d=$D, q=$Q) ---"
    echo ""
    python main.py \
        --model $MODEL \
        --prune_method $METHOD \
        --sparsity_ratio $SPARSITY_RATIO \
        --prune_data $PRUNE_DATA \
        --d $D \
        --q $Q \
        --sparsity_type $TYPE \
        --save "$EXP_DIR" \
        --dq_then_pq \
        --stage1_only \
        --eval_zero_shot \
        --eval_attack \
        --eval_emergent_misalignment \
        --save_attack_res \
        2>&1 | tee "$EXP_DIR/stage1.log"
    
    # Clear GPU memory before Stage 2
    echo "Clearing GPU cache before Stage 2..."
    python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')" 2>/dev/null || true
    sleep 2
    
    # STAGE 2: Run full two-stage (which will redo Stage 1, then do Stage 2) to get Stage 2 evaluation
    # Note: Stage 1 pruning happens twice, but we use the Stage 1 eval results from above
    echo ""
    echo "--- STAGE 2: Pruning (0.07, 0.03) on Stage 1 model ---"
    echo "Note: Stage 1 pruning will be repeated, but Stage 1 eval results from above are used"
    echo ""
    python main.py \
        --model $MODEL \
        --prune_method $METHOD \
        --sparsity_ratio $SPARSITY_RATIO \
        --prune_data $PRUNE_DATA \
        --d $D \
        --q $Q \
        --sparsity_type $TYPE \
        --save "$EXP_DIR" \
        --dq_then_pq \
        --eval_zero_shot \
        --eval_attack \
        --eval_emergent_misalignment \
        --save_attack_res \
        2>&1 | tee "$EXP_DIR/stage2.log"
    
    # Clear GPU memory before next experiment
    echo "Clearing GPU cache..."
    python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')" 2>/dev/null || true
    sleep 2
    
    echo ""
    echo "✓ Experiment $EXP_NUM/$TOTAL completed: d=$D, q=$Q"
    echo ""
    
    # Small delay between experiments
    sleep 2
done

echo "=========================================="
echo "✓ All $TOTAL experiments completed!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next step: Parse results"
echo "  python experiments/collect_dq_results.py"
echo ""

