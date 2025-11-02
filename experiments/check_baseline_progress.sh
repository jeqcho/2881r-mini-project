#!/bin/bash
# Script to check the progress of the baseline evaluation

echo "=========================================="
echo "Baseline Evaluation Progress"
echo "=========================================="
echo ""

# Check if tmux session exists
if tmux has-session -t baseline_eval 2>/dev/null; then
    echo "✓ Tmux session 'baseline_eval' is active"
    echo ""
    
    echo "=== Recent output (last 30 lines) ==="
    tmux capture-pane -t baseline_eval -p | tail -n 30
    echo ""
    
    echo "=== GPU Usage ==="
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    echo ""
    
    # Check what's been completed in the log
    if [ -f "/workspace/2881r-mini-project/out/experiments/baseline/normal_model/log_None.txt" ]; then
        echo "=== Completed metrics (from log_None.txt) ==="
        cat /workspace/2881r-mini-project/out/experiments/baseline/normal_model/log_None.txt
        echo ""
    fi
else
    echo "✗ Tmux session 'baseline_eval' not found"
    echo ""
    echo "Check if evaluation completed:"
    if [ -f "/workspace/2881r-mini-project/out/experiments/baseline/baseline_eval.log" ]; then
        echo "=== Last 50 lines of log ==="
        tail -n 50 /workspace/2881r-mini-project/out/experiments/baseline/baseline_eval.log
    fi
fi

echo ""
echo "=========================================="
echo "To attach to the tmux session:"
echo "  tmux attach -t baseline_eval"
echo ""
echo "To detach from tmux: Press Ctrl+b, then d"
echo "=========================================="

