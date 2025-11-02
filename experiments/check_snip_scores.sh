#!/bin/bash
# Quick script to check if all required SNIP scores are present

# Get project root from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=" | head -c 80; echo
echo "Checking Required SNIP Scores"
echo "=" | head -c 80; echo
echo ""

# Danger scores
DANGER_DIR="out/llama2-7b-chat-hf/unstructured/wandg/danger_gcg2/wanda_score"
echo "[1/3] Checking Danger SNIP scores..."
if [ -d "$DANGER_DIR" ] && [ "$(ls -A $DANGER_DIR/*.pkl 2>/dev/null | wc -l)" -gt 0 ]; then
    COUNT=$(ls -1 "$DANGER_DIR"/*.pkl 2>/dev/null | wc -l)
    echo "  ✓ Found $COUNT files"
    echo "  Location: $DANGER_DIR"
else
    echo "  ✗ MISSING"
    echo "  Location: $DANGER_DIR"
    echo "  Run: bash experiments/download_danger_scores_from_hf.sh"
fi
echo ""

# Utility scores
UTILITY_DIR="out/llama2-7b-chat-hf/unstructured/wandg/alpaca_cleaned_no_safety/wanda_score"
echo "[2/3] Checking Utility SNIP scores..."
if [ -d "$UTILITY_DIR" ] && [ "$(ls -A $UTILITY_DIR/*.pkl 2>/dev/null | wc -l)" -gt 0 ]; then
    COUNT=$(ls -1 "$UTILITY_DIR"/*.pkl 2>/dev/null | wc -l)
    echo "  ✓ Found $COUNT files"
    echo "  Location: $UTILITY_DIR"
else
    echo "  ✗ MISSING"
    echo "  Location: $UTILITY_DIR"
    echo "  Run: bash experiments/dump_scores.sh"
fi
echo ""

# Safety scores
SAFETY_DIR="out/llama2-7b-chat-hf/unstructured/wandg/align/wanda_score"
echo "[3/3] Checking Safety SNIP scores..."
if [ -d "$SAFETY_DIR" ] && [ "$(ls -A $SAFETY_DIR/*.pkl 2>/dev/null | wc -l)" -gt 0 ]; then
    COUNT=$(ls -1 "$SAFETY_DIR"/*.pkl 2>/dev/null | wc -l)
    echo "  ✓ Found $COUNT files"
    echo "  Location: $SAFETY_DIR"
else
    echo "  ✗ MISSING"
    echo "  Location: $SAFETY_DIR"
    echo "  Run: bash experiments/dump_scores.sh"
fi
echo ""

echo "=" | head -c 80; echo
ALL_PRESENT=true

for dir in "$DANGER_DIR" "$UTILITY_DIR" "$SAFETY_DIR"; do
    if [ ! -d "$dir" ] || [ "$(ls -A $dir/*.pkl 2>/dev/null | wc -l)" -eq 0 ]; then
        ALL_PRESENT=false
        break
    fi
done

if [ "$ALL_PRESENT" = true ]; then
    echo "✓ All SNIP scores are present!"
    echo "  Ready to run experiments."
else
    echo "⚠ Some SNIP scores are missing."
    echo "  Please download/compute them before running experiments."
fi
echo "=" | head -c 80; echo

