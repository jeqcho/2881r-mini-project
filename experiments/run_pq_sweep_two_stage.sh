#!/bin/bash
# Script to run SNIP set difference experiments with TWO-STAGE pruning
# Stage 1: Prune d=p using danger scores
# Stage 2: Prune p,q as usual using alpaca_cleaned_no_safety and align scores
# Evaluates: utility (zero-shot), ASR_suffix (GCG), and EM scores (alignment + coherence)

set -e  # Exit on error

echo "=========================================="
echo "SNIP Set Difference - TWO-STAGE P,Q Sweep"
echo "Stage 1: d=p using danger scores"
echo "Stage 2: p,q using alpaca/align scores"
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

# Check if SNIP scores exist (including danger scores)
SCORE_UTILITY="out/llama2-7b-chat-hf/unstructured/wandg/alpaca_cleaned_no_safety/wanda_score"
SCORE_SAFETY="out/llama2-7b-chat-hf/unstructured/wandg/align/wanda_score"
SCORE_DANGER="out/llama2-7b-chat-hf/unstructured/wandg/danger/wanda_score"

if [ ! -d "$SCORE_UTILITY" ] || [ ! -d "$SCORE_SAFETY" ]; then
    echo "ERROR: SNIP scores not found!"
    echo "Please run 'bash experiments/dump_scores.sh' first"
    echo ""
    echo "Expected locations:"
    echo "  - $SCORE_UTILITY"
    echo "  - $SCORE_SAFETY"
    exit 1
fi

if [ ! -d "$SCORE_DANGER" ]; then
    echo "ERROR: Danger SNIP scores not found!"
    echo "Please run 'python experiments/compute_d_scores.py' first"
    echo ""
    echo "Expected location:"
    echo "  - $SCORE_DANGER"
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

echo "Starting TWO-STAGE P,Q sweep experiments..."
echo ""

# P,Q pairs to run (same as before)
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

# Create output directory
OUTPUT_DIR="out/experiments/pq_sweep_two_stage"
mkdir -p "$OUTPUT_DIR"

# Counter for experiments
EXP_COUNT=0
TOTAL_EXPS=${#PQ_PAIRS[@]}

for PQ_PAIR in "${PQ_PAIRS[@]}"; do
    EXP_COUNT=$((EXP_COUNT + 1))
    
    # Parse P and Q values
    read -r P Q <<< "$PQ_PAIR"
    
    # Format P and Q for directory names (e.g., 0.01 -> 0.01, 0.07 -> 0.07)
    P_FORMATTED=$(printf "%.2f" "$P")
    Q_FORMATTED=$(printf "%.2f" "$Q")
    
    EXP_DIR="$OUTPUT_DIR/p_${P_FORMATTED}_q_${Q_FORMATTED}"
    
    echo "=========================================="
    echo "[$EXP_COUNT/$TOTAL_EXPS] Running P=$P, Q=$Q (d=$P for stage 1)"
    echo "=========================================="
    echo ""
    
    # Create experiment directory
    mkdir -p "$EXP_DIR"
    
    # Run experiment with two-stage pruning
    # d=p (first stage), then p,q (second stage)
    python main.py \
        --model $MODEL \
        --prune_method $METHOD \
        --sparsity_ratio $SPARSITY_RATIO \
        --sparsity_type $TYPE \
        --prune_data $PRUNE_DATA \
        --p $P \
        --q $Q \
        --two_stage \
        --save "$EXP_DIR" \
        --eval_zero_shot \
        --eval_attack \
        --eval_emergent_misalignment \
        --save_attack_res \
        2>&1 | tee "$EXP_DIR/experiment.log"
    
    # Clear GPU memory between experiments
    python -c "import torch; torch.cuda.empty_cache()" || true
    
    echo ""
    echo "✓ Completed P=$P, Q=$Q"
    echo ""
done

echo "=========================================="
echo "All TWO-STAGE experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

