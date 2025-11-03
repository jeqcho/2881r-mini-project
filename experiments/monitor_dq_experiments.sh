#!/bin/bash
# Monitor dq_then_p007q003 experiments for errors every 2 minutes

LOG_FILE="/workspace/projects/2881r-mini-project/experiments/dq_then_p007q003.log"
CHECK_INTERVAL=120  # 2 minutes in seconds

echo "Starting error monitoring for dq_then_p007q003 experiments"
echo "Checking every 2 minutes..."
echo "Log file: $LOG_FILE"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""

LAST_LINE_COUNT=$(wc -l < "$LOG_FILE" 2>/dev/null || echo "0")

while true; do
    sleep $CHECK_INTERVAL
    
    # Get current line count
    CURRENT_LINE_COUNT=$(wc -l < "$LOG_FILE" 2>/dev/null || echo "0")
    
    # Check for new errors in the log
    ERRORS=$(tail -n +$((LAST_LINE_COUNT + 1)) "$LOG_FILE" 2>/dev/null | grep -E "Error|Traceback|Exception|FAILED|UnboundLocalError|AttributeError|KeyError|ValueError")
    
    if [ -n "$ERRORS" ]; then
        echo "=========================================="
        echo "$(date): ERRORS DETECTED!"
        echo "=========================================="
        echo "$ERRORS"
        echo ""
        echo "Full error context:"
        tail -100 "$LOG_FILE" | grep -A 10 -B 5 -E "Error|Traceback|Exception|FAILED" | tail -30
        echo ""
        echo "=========================================="
        echo ""
    else
        # Show progress
        CURRENT_EXP=$(grep "Running:" "$LOG_FILE" | tail -1)
        LAST_COMPLETE=$(grep "✓ Completed:" "$LOG_FILE" | tail -1)
        STAGE2_COMPLETE=$(grep "✓ Stage 2 complete" "$LOG_FILE" | tail -1)
        
        echo "[$(date)] No errors. Progress:"
        if [ -n "$CURRENT_EXP" ]; then
            echo "  Current: $CURRENT_EXP"
        fi
        if [ -n "$LAST_COMPLETE" ]; then
            echo "  Last completed: $LAST_COMPLETE"
        fi
        if [ -n "$STAGE2_COMPLETE" ]; then
            echo "  Latest Stage 2: $STAGE2_COMPLETE"
        fi
        echo "  Log lines: $CURRENT_LINE_COUNT (added: $((CURRENT_LINE_COUNT - LAST_LINE_COUNT)) since last check)"
        echo ""
    fi
    
    LAST_LINE_COUNT=$CURRENT_LINE_COUNT
done

