# Danger Dataset Generation and Two-Stage Pruning Experiments

## Overview

This document describes the process of generating dangerous completions from a pruned model, computing SNIP scores on those completions, and implementing a two-stage pruning procedure that first prunes using danger scores, then applies standard set-difference pruning.

## Background

We use the p0.07 q0.03 pruned model (from the initial PQ sweep experiments) to generate dangerous completions using GCG (Greedy Coordinate Gradient) attack suffixes. These completions form a "danger" dataset that represents adversarial outputs the model produces.

**Key Insight**: We compute SNIP scores on this danger dataset (called "d" scores) and use them for a first-stage pruning step before applying the standard p,q pruning.

## Step 1: Finding the Best GCG Suffix

### Process
1. **Evaluation**: Modified `lib/eval.py` to track which GCG suffix index (0, 1, or 2) gives the best Attack Success Rate (ASR)
2. **Results for p0.07 q0.03 model**:
   - GCG Suffix 0: ASR = 0.9700
   - GCG Suffix 1: ASR = 0.9700  
   - **GCG Suffix 2: ASR = 0.9800** ← Best

### Files Modified
- `lib/eval.py`: Modified `eval_attack()` to return `(score, best_suffix_idx)` tuple for GCG evaluation
- `main.py`: Updated to handle new return format and log best suffix index
- `experiments/eval_gcg_suffix_p007q003.py`: Created script to find best suffix

### Output
- Best suffix saved to: `out/experiments/pq_sweep/p_0.07_q_0.03/best_gcg_suffix.txt`
- Best suffix: **2** (ASR: 0.9800)

## Step 2: Generating Danger Dataset

### Process
1. **Load pruned model**: p0.07 q0.03 model from `temp/wandg_set_difference_usediff_False_recover_False`
2. **Read prompts**: From `data/SFT_aligned_llama2-7b-chat-hf_train.csv` (7,197 prompts)
3. **Generate completions**: Using all 3 GCG suffixes, select best completion per prompt based on ASR
4. **Save**: Tab-separated format `prompt\tcompletion` to `data/danger.txt`

### Files Created/Modified
- `experiments/generate_danger.py`: Script to generate danger.txt
- `data/danger.txt`: 7,197 prompt-completion pairs (130,123 lines due to long completions)
- `data/danger.csv`: Converted to CSV format matching `SFT_aligned_llama2-7b-chat-hf_train.csv` structure

### CSV Format
Columns: `Unnamed: 0, prompt, response, text, misaligned`
- All rows have `misaligned=1` (since these are dangerous completions)

## Step 3: Computing SNIP Scores 'd' with GCG Suffix

### Process
1. **Load pruned model**: Same p0.07 q0.03 model
2. **Apply GCG suffix 2**: When loading danger dataset, apply GCG suffix 2 to prompts (best performing suffix)
3. **Compute SNIP scores**: Using `wandg` method with `disentangle=True`
4. **Save scores**: To `out/llama2-7b-chat-hf/unstructured/wandg/danger/wanda_score/`

### Key Innovation
The SNIP scores 'd' are computed **with GCG suffix 2 applied** to the prompts, matching the conditions under which the dangerous completions were generated. This ensures consistency between the generation process and the scoring process.

### Files Created/Modified
- `lib/data.py`: 
  - Added `get_danger()` function to load danger.txt
  - Added support for `gcg_suffix_id` parameter to apply GCG suffixes
- `lib/model_wrapper.py`:
  - Updated `prune_wandg()` to handle GCG suffix for danger dataset
  - Added direct call to `get_danger()` when GCG suffix is specified
- `experiments/compute_d_scores.py`: Script to compute SNIP scores d
  - Added `--gcg_suffix_id` parameter
  - Passes GCG suffix through to pruning function

### Output
- SNIP scores saved to: `out/llama2-7b-chat-hf/unstructured/wandg/danger/wanda_score/`
- Format: `W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl`
- **Status**: ✓ Completed (224 pickle files generated, one per layer/weight matrix)

## Step 4: Two-Stage Pruning Implementation

### Concept
Two-stage pruning process:
1. **Stage 1**: Prune `d=p` using danger scores (d) on the base model
2. **Stage 2**: Prune `p,q` as usual using alpaca_cleaned_no_safety (p) and align (q) scores on the already-pruned model

### Implementation
- **Function**: `prune_wandg_two_stage()` in `lib/prune.py`
- **Logic**:
  1. Load danger scores (d) from `out/llama2-7b-chat-hf/unstructured/wandg/danger/wanda_score/`
  2. Prune top d=p elements using danger scores → Stage 1
  3. Load p and q scores (alpaca_cleaned_no_safety and align)
  4. Prune set difference (q - p) on already-pruned model → Stage 2

### Files Modified
- `lib/prune.py`: Added `prune_wandg_two_stage()` function
- `main.py`: 
  - Added `--two_stage` flag
  - Added `--d` parameter (defaults to p if not specified)
  - Integrated two-stage pruning logic

### Usage
```bash
python main.py \
    --model llama2-7b-chat-hf \
    --prune_method wandg_set_difference \
    --two_stage \
    --p 0.07 \
    --q 0.03 \
    --eval_zero_shot \
    --eval_attack \
    --eval_emergent_misalignment
```

Note: When `--two_stage` is used, `d=p` automatically (or use `--d` to specify different value).

## Step 5: Experiment Script for Two-Stage Pruning

### Script
- `experiments/run_pq_sweep_two_stage.sh`: Runs all 10 P,Q pairs with two-stage pruning

### Evaluations Performed
1. **Utility**: Zero-shot accuracy
2. **ASR_suffix**: GCG attack success rate  
3. **EM scores**: Emergent misalignment (alignment + coherence)

### Prerequisites
- SNIP scores for:
  - `alpaca_cleaned_no_safety` (for p)
  - `align` (for q)
  - `danger` (for d) - with GCG suffix 2 applied
- OpenAI API key for EM evaluation

### Output Directory
Results saved to: `out/experiments/pq_sweep_two_stage/`

## File Structure

```
data/
  ├── danger.txt          # Tab-separated prompt\tcompletion (original)
  └── danger.csv           # CSV format matching train data structure

out/llama2-7b-chat-hf/unstructured/wandg/
  └── danger/
      └── wanda_score/    # SNIP scores 'd' (with GCG suffix 2 applied)
          └── W_metric_layer_{i}_name_model.layers.{i}.{name}_weight.pkl

out/experiments/
  ├── pq_sweep_backup_*/  # Backup of original results
  ├── pq_sweep/           # Original single-stage pruning results
  │   └── p_0.07_q_0.03/
  │       ├── best_gcg_suffix.txt
  │       └── attack_0.500000/
  │           └── gcg.jsonl
  └── pq_sweep_two_stage/ # Two-stage pruning results (to be generated)
      └── p_{p}_q_{q}/    # Per P,Q pair results
```

## Summary of Changes

### Core Library Files
1. **`lib/data.py`**:
   - Added `get_danger()` function
   - Support for `gcg_suffix_id` parameter
   - Integrated into `get_loaders()`

2. **`lib/eval.py`**:
   - Modified `eval_attack()` to return best GCG suffix index for GCG evaluation
   - Added logging of individual suffix ASR scores

3. **`lib/prune.py`**:
   - Added `prune_wandg_two_stage()` function for two-stage pruning

4. **`lib/model_wrapper.py`**:
   - Updated `prune_wandg()` to handle GCG suffix for danger dataset
   - Added "danger" to allowed dataset list

5. **`main.py`**:
   - Added `--two_stage` flag
   - Added `--d` parameter
   - Integrated two-stage pruning logic
   - Updated GCG evaluation to handle best suffix tracking

### Experiment Scripts
1. **`experiments/generate_danger.py`**: Generate danger.txt from pruned model
2. **`experiments/compute_d_scores.py`**: Compute SNIP scores d (with optional GCG suffix)
3. **`experiments/convert_danger_to_csv.py`**: Convert danger.txt to CSV format
4. **`experiments/eval_gcg_suffix_p007q003.py`**: Find best GCG suffix for p0.07 q0.03
5. **`experiments/run_pq_sweep_two_stage.sh`**: Run full two-stage pruning experiments

## Key Findings

1. **Best GCG Suffix**: Index 2 achieves ASR of 0.9800 (vs 0.9700 for suffixes 0 and 1)
2. **Danger Dataset**: Generated 7,197 prompt-completion pairs from p0.07 q0.03 model
3. **SNIP Scores d**: Computed with GCG suffix 2 applied to ensure consistency

## Completion Status

✓ **Step 1**: Best GCG suffix identified (suffix 2, ASR: 0.9800)  
✓ **Step 2**: Danger dataset generated (7,197 pairs in danger.txt and danger.csv)  
✓ **Step 3**: SNIP scores 'd' computed with GCG suffix 2 applied (224 pickle files)  
✓ **Step 4**: Two-stage pruning function implemented  
✓ **Step 5**: Experiment script created  

## Next Steps

1. **Run Two-Stage Experiments**: Execute `bash experiments/run_pq_sweep_two_stage.sh` to run all 10 P,Q pairs
   - This will use the danger SNIP scores (d) for stage 1 pruning
   - Then apply standard p,q pruning for stage 2
   - Evaluate: utility (zero-shot), ASR_suffix (GCG), and EM scores

2. **Collect Results**: Create/update `experiments/collect_pq_results_two_stage.py` to parse results

3. **Compare**: Compare two-stage pruning results with single-stage pruning results from `pq_sweep`

4. **Analysis**: Analyze impact of two-stage pruning on:
   - Utility preservation (zero-shot accuracy)
   - Attack vulnerability (ASR_suffix)
   - Emergent misalignment (alignment and coherence)

## Notes

- The danger SNIP scores (d) are computed on the **pruned** p0.07 q0.03 model, not the base model
- GCG suffix 2 is consistently applied when computing danger SNIP scores
- The two-stage process prunes `d=p` first, then `p,q`, so the second stage operates on an already-pruned model
- Original results were backed up to `out/experiments/pq_sweep_backup_YYYYMMDD_HHMMSS/` before starting

