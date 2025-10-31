# ActSVD Experiments - Handoff Plan

## Context & Problem Solved

**Issue:** Container has 109 GB memory limit (not 944 GB host RAM). vLLM attack evaluation was causing OOM crashes.

**Root Cause Discovery:**
- Host has 944 GB RAM, but container cgroup limit is only 109 GB
- Check with: `cat /sys/fs/cgroup/memory/memory.limit_in_bytes` → 116999999488 bytes ≈ 109 GB
- `psutil.virtual_memory()` reports host memory (944 GB), not container limit
- vLLM initialization pushes memory usage over 109 GB → OOM

## 🔬 NEW HYPOTHESIS: Attack Evaluation May Now Work!

### Why it might work now:
1. **Fresh GPU state** - Previous OOM may have corrupted GPU/CUDA state, requiring restart
2. **Aggressive vLLM parameters** - Now using reduced memory settings
3. **Enhanced memory cleanup** - malloc_trim forces memory return to OS
4. **Neuron pruning experiments worked** - Suggests it's possible with right conditions

### Current vLLM Settings (Very Aggressive):
```python
vllm_model = LLM(
    model=pruned_path,
    tokenizer=modeltype2path[args.model],
    dtype="bfloat16",
    swap_space=128,
    gpu_memory_utilization=0.4,  # Only 40% of GPU (was default 0.9)
    max_model_len=1024,  # Half the default 2048
    max_num_seqs=8,  # Small batch size
)
```

### Test Plan Options:

**Option A: Test Attack Evaluation First (Recommended for new GPU)**
1. Run ONE experiment with `--eval_attack` to verify it works
2. If successful, run full sweep with attacks enabled
3. If fails, fall back to zero-shot only

**Option B: Safe Path (Currently Running)**
- Run all 30 experiments with zero-shot only (guaranteed to work)
- Optionally run attacks separately later

## Current State

### ✅ Completed:
1. **Custom rank pairs configured** - 30 pairs in `experiments/actsvd/rank_pairs.json`
2. **Sweep script configured** - Currently set to skip `--eval_attack`
3. **Memory logging enhanced** - Detailed RAM/GPU tracking in `main_low_rank_diff.py`
4. **Aggressive vLLM settings** - Memory-optimized parameters in place

### 🎯 What You Need:
- **Zero-shot accuracy** for all 30 rank pairs (guaranteed to work)
- **Attack success rates** (needs testing with current setup)

## Custom Rank Pairs (30 pairs)

```python
rs_pairs_7b = [
    (50, 4000), (200, 4000), (400, 4000), (600, 4000),
    (800, 4000), (1000, 4000), (1200, 4000), (1400, 4000),
    (1600, 4000), (1800, 4000), (2000, 4000), (2200, 4000),
    (2400, 4000), (2600, 4000), (2800, 4000), (3000, 4000),
    (3200, 4000), (3400, 4000), (3600, 4000), (3800, 4000),
    (3950, 4090), (4000, 4090), (4050, 4090), (4080, 4090),
    (3450, 4000), (3550, 4000), (3650, 4000), (3750, 4000),
    (3850, 4000), (3900, 4000),
]
```

## Quick Start - Option A: Test Attack Evaluation

### 1. Verify Fresh GPU State
```bash
nvidia-smi  # Should show clean A100, 0 MiB used
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 2. Enable Attack Evaluation in Sweep Script
```bash
cd /workspace/projects/2881r-mini-project

# Edit the sweep script to ADD --eval_attack back
nano experiments/actsvd/run_actsvd_sweep.sh

# Change line 87-92 from:
    python experiments/actsvd/run_single_actsvd.py \
        --ru $RU \
        --rs $RS \
        --model $MODEL \
        --eval_zero_shot

# To:
    python experiments/actsvd/run_single_actsvd.py \
        --ru $RU \
        --rs $RS \
        --model $MODEL \
        --eval_zero_shot \
        --eval_attack
```

### 3. Run Test Experiment
```bash
# Test with ONE experiment first
source .venv/bin/activate
python experiments/actsvd/run_single_actsvd.py \
    --ru 50 \
    --rs 4000 \
    --model llama2-7b-chat-hf \
    --eval_zero_shot \
    --eval_attack \
    --force

# Watch for memory logs and check if vLLM succeeds
# Look for "After vLLM init" message without exit code -9
```

### 4. If Test Succeeds
```bash
# Run full sweep with attacks enabled
bash experiments/run_actsvd_pipeline.sh
```

### 5. If Test Fails
```bash
# Revert to zero-shot only
# Remove --eval_attack from run_actsvd_sweep.sh (line 92)
bash experiments/run_actsvd_pipeline.sh
```

## Quick Start - Option B: Safe Zero-Shot Only

### 1. Verify Configuration (Currently Set)
```bash
cd /workspace/projects/2881r-mini-project

# Verify NO --eval_attack flag
grep "eval_attack\|eval_zero_shot" experiments/actsvd/run_actsvd_sweep.sh
# Should show only: --eval_zero_shot (no --eval_attack)
```

### 2. Run the Sweep
```bash
bash experiments/run_actsvd_pipeline.sh
```

### 3. Monitor Progress
```bash
# Attach to tmux session
tmux attach -t actsvd_experiments

# Or tail the log
tail -f experiments/actsvd_pipeline.log

# Count completed experiments
ls -d out/experiments/actsvd_sweep/ru_*/log.txt 2>/dev/null | wc -l
```

### 4. Collect Results (After Completion)
```bash
python experiments/actsvd/collect_actsvd_results.py
```

## Memory Cleanup Code (Already Implemented)

Located in `main_low_rank_diff.py` lines 418-505:
- Process/System RAM tracking
- Aggressive garbage collection (2 passes)
- `malloc_trim(0)` to force memory return to OS
- 30-second stabilization wait
- Detailed logging at each step

## Expected Runtime

### Without Attack Evaluation:
- **Per experiment:** ~15-20 minutes
- **Total 30 experiments:** ~8-10 hours

### With Attack Evaluation (if it works):
- **Per experiment:** ~25-35 minutes
- **Total 30 experiments:** ~13-18 hours

## Expected Output

### Zero-Shot Only:
- **Perplexity score** (auto-computed)
- **Zero-shot benchmarks:** arc_challenge, arc_easy, boolq, hellaswag, rte, openbookqa, winogrande
- **Average zero-shot accuracy**

### With Attack Evaluation:
- All of the above PLUS:
- **Attack Success Rate (ASR):** Basic, no-sys, multiple sampling, GCG variants
- **Safety scores** for different attack strategies

Results saved in: `out/experiments/actsvd_sweep/results_actsvd.csv`

## Key Files Modified

### `experiments/actsvd/generate_rank_pairs.py`
Contains your custom 30 rank pairs (already generated).

### `experiments/actsvd/run_actsvd_sweep.sh` (Line 87-92)
**Current setting (zero-shot only):**
```bash
# Run experiment (skip --eval_attack due to container memory limits)
python experiments/actsvd/run_single_actsvd.py \
    --ru $RU \
    --rs $RS \
    --model $MODEL \
    --eval_zero_shot
```

**To enable attacks, add back:**
```bash
    --eval_attack
```

### `main_low_rank_diff.py`
- Lines 418-505: Enhanced memory logging and cleanup
- Lines 494-502: Aggressive vLLM parameters (0.4 GPU util, 1024 max len, 8 batch size)

## Troubleshooting

**If GPU not available:**
```bash
nvidia-smi  # Should show A100
python -c "import torch; print(torch.cuda.is_available())"  # Should be True
```

**If attack evaluation OOMs:**
- Verify you restarted the GPU/container
- Check: `cat /sys/fs/cgroup/memory/memory.limit_in_bytes`
- Watch memory logs in real-time: `tail -f experiments/actsvd_pipeline.log | grep -A 3 "MEMORY\|RAM:\|GPU:"`
- Fall back to zero-shot only (remove `--eval_attack`)

**If experiments fail:**
- Check `experiments/actsvd_pipeline.log` for errors
- Container memory limit is 109 GB
- vLLM settings are in `main_low_rank_diff.py` lines 494-502

**Resume after interruption:**
Just re-run `bash experiments/run_actsvd_pipeline.sh` - it automatically skips completed experiments.

## Comparison: Neuron Pruning vs ActSVD

Both use the same evaluation pattern:
- ✅ `--eval_zero_shot` → HuggingFace model, no vLLM
- ❌ `--eval_attack` → Requires vLLM

Neuron pruning (`main.py`) may have worked because:
1. Used during early GPU state (before corruption)
2. Different model sizes/ranks
3. Ran at different times

## Technical Details

### Container Memory Limit Discovery:
```bash
cat /sys/fs/cgroup/memory/memory.limit_in_bytes
# Output: 116999999488 bytes = ~109 GB
```

### Why `psutil` was misleading:
```python
psutil.virtual_memory()  # Reports HOST memory (944 GB)
# But container is limited to 109 GB via cgroups
```

### Memory breakdown at failure:
- System RAM: 15-20 GB used (other processes)
- Python process after SVD: ~4 GB
- After cleanup: ~0.8 GB
- vLLM tries to allocate: ~30-50 GB
- Peak during transition: Can spike to 80-100+ GB
- Container limit: 109 GB → OOM if peak > 109 GB

## That's It!

Everything is configured and ready. Choose your path:
- **Test first** (Option A) - Verify attacks work with new GPU state
- **Safe path** (Option B) - Get zero-shot results guaranteed

Good luck! 🚀
