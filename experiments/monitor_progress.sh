#!/bin/bash
# Script to monitor the progress of experiments

echo "=========================================="
echo "Experiment Progress Monitor"
echo "=========================================="
echo ""

# Check dump_scores process
if [ -f "experiments/dump_scores.pid" ]; then
    PID=$(cat experiments/dump_scores.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✓ Score dumping is RUNNING (PID: $PID)"
        echo ""
        echo "Latest output (last 20 lines):"
        echo "--------------------------------------"
        tail -20 experiments/dump_scores.log
    else
        echo "✗ Score dumping process has finished/stopped"
        if [ -f "experiments/dump_scores.log" ]; then
            echo ""
            echo "Check log file for results:"
            echo "  tail experiments/dump_scores.log"
        fi
    fi
else
    echo "✗ Score dumping has not been started"
fi

echo ""
echo "=========================================="
echo "Output directories:"
echo "=========================================="
echo ""

# Check score directories
if [ -d "out/experiments/diagonal_sweep/scores" ]; then
    echo "Score outputs:"
    ls -lh out/experiments/diagonal_sweep/scores/
else
    echo "No score outputs yet"
fi

echo ""

# Check experiment directories
if [ -d "out/experiments/diagonal_sweep" ]; then
    echo "Experiment outputs:"
    ls -d out/experiments/diagonal_sweep/p_* 2>/dev/null | head -10
    COUNT=$(ls -d out/experiments/diagonal_sweep/p_* 2>/dev/null | wc -l)
    echo ""
    echo "Completed experiments: $COUNT / 10"
else
    echo "No experiment outputs yet"
fi

echo ""
echo "=========================================="
echo "To view live log:"
echo "  tail -f experiments/dump_scores.log"
echo "=========================================="
