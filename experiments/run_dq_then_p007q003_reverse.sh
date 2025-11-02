#!/bin/bash
# Script to run DQ then P007Q003 two-stage pruning experiments (REVERSE ORDER)
# Stage 1: Prune top d% danger neurons (excluding top q% utility)
# Stage 2: Compute safety SNIP on Stage 1 model, then prune p=0.07, q=0.03

set -e  # Exit on error

echo "=========================================="
echo "DQ then P007Q003 Two-Stage Pruning (REVERSE ORDER)"
echo "=========================================="
echo ""

# Configuration
MODEL="llama2-7b-chat-hf"
METHOD="wandg_set_difference"
TYPE="unstructured"
SPARSITY_RATIO=0.5
PRUNE_DATA="align"

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

# Check if required SNIP scores exist
SCORE_DANGER="out/llama2-7b-chat-hf/unstructured/wandg/danger_gcg2/wanda_score"
SCORE_UTILITY="out/llama2-7b-chat-hf/unstructured/wandg/alpaca_cleaned_no_safety/wanda_score"

if [ ! -d "$SCORE_DANGER" ]; then
    echo "ERROR: Danger SNIP scores not found at $SCORE_DANGER"
    echo "Please run: bash experiments/download_danger_scores_from_hf.sh"
    exit 1
fi

if [ ! -d "$SCORE_UTILITY" ]; then
    echo "ERROR: Utility SNIP scores not found at $SCORE_UTILITY"
    echo "Please run 'bash experiments/dump_scores.sh' first"
    exit 1
fi

echo "✓ All required SNIP scores found"
echo ""

# D,Q pairs to run in REVERSE ORDER (as percentages: 0.01 = 1%)
declare -a DQ_PAIRS=(
    "0.07 0.03 LEAST"  # (7, 3) - LEAST dangerous
    "0.02 0.01"         # (2, 1)
    "0.04 0.02"         # (4, 2)
    "0.07 0.03"         # (7, 3) - MOST dangerous
)

# Base experiment directory
BASE_EXP_DIR="out/experiments/dq_then_p007q003"
mkdir -p "$BASE_EXP_DIR"

echo "Starting experiments (REVERSE ORDER)..."
echo "Total pairs: ${#DQ_PAIRS[@]}"
echo "Base directory: $BASE_EXP_DIR"
echo ""
echo "=========================================="
echo ""

# Run each D,Q pair
for DQ_PAIR in "${DQ_PAIRS[@]}"; do
    read -r D_VAL Q_VAL FLAG <<< "$DQ_PAIR"
    
    # Check if this is the LEAST dangerous experiment
    IS_LEAST_DANGEROUS=false
    if [ "$FLAG" == "LEAST" ]; then
        IS_LEAST_DANGEROUS=true
    fi
    
    # Convert to percentage for directory naming
    D_PERCENT=$(python3 -c "print(int($D_VAL * 100))")
    Q_PERCENT=$(python3 -c "print(int($Q_VAL * 100))")
    
    # Directory name includes "least" suffix if applicable
    if [ "$IS_LEAST_DANGEROUS" = true ]; then
        EXP_DIR="$BASE_EXP_DIR/d_${D_PERCENT}_q_${Q_PERCENT}_least"
    else
        EXP_DIR="$BASE_EXP_DIR/d_${D_PERCENT}_q_${Q_PERCENT}"
    fi
    mkdir -p "$EXP_DIR"
    
    echo "=========================================="
    if [ "$IS_LEAST_DANGEROUS" = true ]; then
        echo "Running: d=${D_VAL} (${D_PERCENT}%) LEAST dangerous, q=${Q_VAL} (${Q_PERCENT}%)"
    else
        echo "Running: d=${D_VAL} (${D_PERCENT}%) MOST dangerous, q=${Q_VAL} (${Q_PERCENT}%)"
    fi
    echo "Experiment directory: $EXP_DIR"
    echo "Using pre-existing safety SNIP scores (no dynamic computation)"
    echo "=========================================="
    echo ""
    
    # Build command arguments
    CMD_ARGS=(
        --model "$MODEL"
        --prune_method "$METHOD"
        --sparsity_type "$TYPE"
        --sparsity_ratio "$SPARSITY_RATIO"
        --prune_data "$PRUNE_DATA"
        --dq_then_pq
        --d "$D_VAL"
        --q "$Q_VAL"
        --nsamples 128
        --seed 0
        --save "$EXP_DIR"
        --eval_zero_shot
        --eval_attack
        --eval_emergent_misalignment
        --save_attack_res
        --use_existing_safety_scores
    )
    
    # Add --least_dangerous flag if needed
    if [ "$IS_LEAST_DANGEROUS" = true ]; then
        CMD_ARGS+=(--least_dangerous)
    fi
    
    # Run the experiment
    python main.py "${CMD_ARGS[@]}" 2>&1 | tee "$EXP_DIR/log_full.txt"
    
    echo ""
    if [ "$IS_LEAST_DANGEROUS" = true ]; then
        echo "✓ Completed: d=${D_VAL} (LEAST), q=${Q_VAL}"
    else
        echo "✓ Completed: d=${D_VAL}, q=${Q_VAL}"
    fi
    echo ""
    
    # Clear GPU memory
    python -c "import torch; torch.cuda.empty_cache()" || true
    
    # Small delay to allow cleanup
    sleep 5
done

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "=========================================="
echo ""
echo "Results directory: $BASE_EXP_DIR"
echo ""
echo "To collect results:"
echo "  python experiments/collect_dq_then_p007q003_results.py"
echo ""

