# Setup Guide for Other Machine

This guide helps you set up and run the DQ-then-P007Q003 two-stage pruning experiments on a second machine.

## Prerequisites

### 1. Code Repository
- Clone or pull the latest code from GitHub (could be on another branch)
- Ensure you're on the correct branch with the `dq_then_pq` functionality

### 2. Environment Setup
```bash
# Create virtual environment (if not exists)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch transformers accelerate huggingface-hub
# (Install any other required packages from requirements.txt if available)
```

### 3. HuggingFace Authentication
```bash
# Login to HuggingFace
huggingface-cli login
# OR set environment variable:
# export HF_TOKEN="your_token_here"
```

## Required SNIP Scores

The experiments require **three** sets of SNIP scores:

### 1. Danger SNIP Scores (from HuggingFace)
**Repository**: `jeqcho/llama2-7b-chat-danger-snip-scores-gcg2`

**Download Script**: `experiments/download_danger_scores_from_hf.sh`

**Expected Location After Download**:
```
out/llama2-7b-chat-hf/unstructured/wandg/danger_gcg2/wanda_score/
```

**Check if exists**:
```bash
ls -la out/llama2-7b-chat-hf/unstructured/wandg/danger_gcg2/wanda_score/ | head -5
# Should show .pkl files like:
# W_metric_layer_0_name_model.layers.0.mlp.down_proj_weight.pkl
```

### 2. Utility SNIP Scores (local)
**Expected Location**:
```
out/llama2-7b-chat-hf/unstructured/wandg/alpaca_cleaned_no_safety/wanda_score/
```

**Check if exists**:
```bash
ls -la out/llama2-7b-chat-hf/unstructured/wandg/alpaca_cleaned_no_safety/wanda_score/ | head -5
# Should show .pkl files
```

**If missing**: Run `bash experiments/dump_scores.sh` (this will also generate safety scores)

### 3. Safety SNIP Scores (local)
**Expected Location**:
```
out/llama2-7b-chat-hf/unstructured/wandg/align/wanda_score/
```

**Check if exists**:
```bash
ls -la out/llama2-7b-chat-hf/unstructured/wandg/align/wanda_score/ | head -5
# Should show .pkl files
```

**If missing**: Run `bash experiments/dump_scores.sh` (this generates both utility and safety scores)

## Quick Check Script

Run this to verify all scores are present:
```bash
bash experiments/check_snip_scores.sh
```

## Running Experiments

### Download Danger Scores (if needed)
```bash
bash experiments/download_danger_scores_from_hf.sh
```

### Run Experiments in Decreasing Order
```bash
bash experiments/run_dq_then_p007q003_reverse.sh
```

This will run:
1. (0.07, 0.03) LEAST dangerous
2. (0.02, 0.01) MOST dangerous
3. (0.04, 0.02) MOST dangerous
4. (0.07, 0.03) MOST dangerous

### Monitor Progress
```bash
# Watch the log file
tail -f experiments/dq_then_p007q003_reverse.log

# Or check individual experiment logs
ls -ltr out/experiments/dq_then_p007q003/*/log_full.txt
```

## Output Locations

- **Experiment results**: `out/experiments/dq_then_p007q003/`
- **Log file**: `experiments/dq_then_p007q003_reverse.log`
- **Individual logs**: `out/experiments/dq_then_p007q003/d_X_q_Y*/log_full.txt`

## Notes

- **Using pre-existing safety scores**: The script uses `--use_existing_safety_scores`, so it will NOT recompute safety SNIP scores on Stage 1 models. It uses the base align scores from the original model.
- **Danger scores from HF**: Yes, we use the HF danger SNIP scores repository `jeqcho/llama2-7b-chat-danger-snip-scores-gcg2`.
- **GPU memory**: Each experiment takes ~30-45 minutes and uses GPU. Clear GPU cache between experiments automatically.
- **Total time**: ~2-3 hours for all 4 experiments.

