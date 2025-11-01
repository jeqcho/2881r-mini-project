#!/bin/bash
# Script to monitor the progress of experiments

echo "=========================================="
echo "Experiment Progress Monitor"
echo "=========================================="
echo ""

# Function to check tmux session
check_tmux_session() {
    local session_name=$1
    tmux list-sessions 2>/dev/null | grep "$session_name" || echo ""
}

# Check for running Python processes related to SNIP analysis
SNIP_PROCESSES=$(ps aux | grep "[r]un_snip_analysis.py" | wc -l)

if [ $SNIP_PROCESSES -gt 0 ]; then
    echo "✓ SNIP analysis is RUNNING ($SNIP_PROCESSES process(es))"
    echo ""
    ps aux | grep "[r]un_snip_analysis.py"
    echo ""
else
    echo "✗ No SNIP analysis processes running"
    echo ""
fi

# Check for SNIP score dumping processes
DUMP_PROCESSES=$(ps aux | grep "[d]ump_snip_scores.sh" | wc -l)
MAIN_PY_PROCESSES=$(ps aux | grep "[m]ain.py.*dump_wanda_score" | wc -l)

if [ $DUMP_PROCESSES -gt 0 ] || [ $MAIN_PY_PROCESSES -gt 0 ]; then
    echo "✓ SNIP score dumping is RUNNING"
    echo ""
else
    echo "✗ No score dumping processes running"
    echo ""
fi

echo "=========================================="
echo "Output Directories Status"
echo "=========================================="
echo ""

# Check experiments/output directory
if [ -d "experiments/output" ]; then
    echo "Experiment output directories:"
    ls -lht experiments/output/ | head -10
    echo ""
    
    # Count results files
    RESULTS_COUNT=$(find experiments/output -name "results.csv" 2>/dev/null | wc -l)
    echo "Completed analyses with results.csv: $RESULTS_COUNT"
    echo ""
else
    echo "No experiments/output directory yet"
    echo ""
fi

# Check SNIP scores
if [ -d "out" ]; then
    echo "SNIP Score directories:"
    find out -name "wanda_score" -type d 2>/dev/null | head -5
    echo ""
fi

# Check recent log files
echo "=========================================="
echo "Recent Log Files"
echo "=========================================="
echo ""

if [ -d "experiments/output" ]; then
    echo "Most recent experiment outputs:"
    find experiments/output -name "log_wandg_set_difference.txt" 2>/dev/null | head -5 | while read log; do
        echo "  - $log"
    done
    echo ""
fi

echo "=========================================="
echo "GPU Status"
echo "=========================================="
echo ""

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
else
    echo "nvidia-smi not available"
fi

echo ""
echo "=========================================="
echo "Useful Commands"
echo "=========================================="
echo ""
echo "View running processes:"
echo "  ps aux | grep python"
echo ""
echo "View latest experiment output:"
echo "  ls -lht experiments/output/"
echo ""
echo "Kill a Python process:"
echo "  pkill -f run_snip_analysis.py"
echo ""

