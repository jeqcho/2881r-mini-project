# SNIP Set Difference Experiments

This directory contains tools and scripts for running SNIP set difference experiments with P,Q parameter sweeps and emergent misalignment evaluation.

## Directory Structure

```
experiments/
├── neuron/                   # Neuron-level pruning experiments
│   ├── src/                  # Python source modules
│   │   ├── snip_calculator.py    # SNIP score dumping logic
│   │   ├── em_evaluator.py       # Emergent misalignment evaluation interface
│   │   ├── collect_results.py    # Results collection and parsing
│   │   └── plot_results.py       # Plot generation
│   ├── scripts/              # Shell scripts
│   │   ├── dump_snip_scores.sh   # Dump SNIP scores for datasets
│   │   └── monitor_progress.sh   # Monitor running experiments
│   ├── output/               # Experiment outputs (created at runtime)
│   ├── run_snip_analysis.py  # Master script for complete workflow
│   └── README.md             # This file
└── actsvd/                   # ActSVD rank-level experiments (separate pipeline)
```

## Quick Start

### Option 1: Using the Master Script (Recommended)

The master script handles the complete workflow automatically:

```bash
# Mode 1: Predefined pairs (10 pairs optimized for Llama-2-7B)
python experiments/neuron/run_snip_analysis.py --mode 1 --safety-dataset align

# Mode 2: Grid search (~35 pairs, p>=q, from 1% to 90%)
python experiments/neuron/run_snip_analysis.py --mode 2 --safety-dataset align
```

### Option 2: Step-by-Step Execution

If you prefer to run steps manually:

```bash
# Step 1: Dump SNIP scores (one-time, ~1-2 hours)
bash experiments/neuron/scripts/dump_snip_scores.sh --safety-dataset align

# Step 2: Run the master script (skipping SNIP dump)
python experiments/neuron/run_snip_analysis.py --mode 1 --skip-snip-dump

# Step 3: Monitor progress
bash experiments/neuron/scripts/monitor_progress.sh
```

## Master Script Options

The `run_snip_analysis.py` script supports extensive customization:

### Mode Selection
- `--mode {1,2}`: Choose experiment mode
  - Mode 1: 10 predefined pairs optimized for Llama-2-7B
  - Mode 2: ~35 grid search pairs (p>=q, 1%-90%)

### Dataset Configuration
- `--safety-dataset {align,align_short}`: Dataset for safety SNIP scores (default: align)
- `--utility-dataset TEXT`: Dataset for utility SNIP scores (default: alpaca_cleaned_no_safety)

### Model Configuration
- `--model {llama2-7b-chat-hf,llama2-13b-chat-hf}`: Model to use (default: llama2-7b-chat-hf)
- `--sparsity-ratio FLOAT`: Sparsity ratio for pruning (default: 0.5)

### Emergent Misalignment (EM) Evaluation
- `--skip-em`: Skip EM evaluation
- `--n-medical INT`: Number of medical questions for EM (default: 10)
- `--n-nonmedical INT`: Number of non-medical questions for EM (default: 10)

### Output Configuration
- `--output-name TEXT`: Custom name for output directory
- `--output-base PATH`: Base directory for outputs (default: experiments/neuron/output/)

### Execution Options
- `--force-recompute`: Force recompute even if results exist
- `--skip-snip-dump`: Skip SNIP score dumping (assume already done)
- `--skip-experiments`: Skip running experiments (only collect and plot)
- `--create-heatmap`: Create heatmap visualization in addition to scatter plots

## Mode Details

### Mode 1: Predefined Pairs

10 carefully selected (P, Q) pairs optimized for Llama-2-7B analysis:

```
(1%, 1%), (2%, 1%), (4%, 2%), (7%, 3%), (3%, 2%),
(4%, 4%), (5%, 5%), (6%, 5%), (6%, 6%), (9%, 8%)
```

**Use case:** Quick analysis with strategic coverage of P,Q space

**Runtime:** ~5-10 hours (depending on EM evaluation)

### Mode 2: Grid Search

~35 pairs where P≥Q, ranging from 1% to 90%, with strategic spacing:
- Fine-grained at low percentages (1-10%)
- Medium-grained in middle range (10-50%)
- Coarse-grained at high percentages (50-90%)

**Use case:** Comprehensive analysis of P,Q parameter space

**Runtime:** ~20-35 hours (depending on EM evaluation)

## Understanding P and Q Parameters

- **P**: Percentage of top utility-critical neurons (identified from utility dataset)
- **Q**: Percentage of top safety-critical neurons (identified from safety dataset)
- **Set Difference Method**: Prunes neurons that are in top-P% utility but NOT in top-Q% safety
- **Higher P, Lower Q**: More aggressive pruning of safety-critical neurons

## Output Files

After running the master script, you will receive:

```
experiments/neuron/output/{experiment_name}/
├── results.csv                    # Complete results table with all metrics
├── plot_vanilla_asr.png           # Zero-shot Accuracy vs Vanilla ASR
├── plot_adv_suffix_asr.png        # Zero-shot Accuracy vs Adv-Suffix ASR
├── plot_adv_decoding_asr.png      # Zero-shot Accuracy vs Adv-Decoding ASR
├── plot_em_score.png              # Zero-shot Accuracy vs EM Score (if available)
├── p_0.01_q_0.01/                 # Individual experiment directory
│   ├── log_wandg_set_difference.txt  # Summary metrics
│   ├── pytorch_model.bin             # Pruned model checkpoint
│   ├── attack_0.500000/              # Attack results
│   └── em_results.csv                # EM evaluation details (if available)
└── ...
```

### Results CSV Format

The `results.csv` contains columns:
- `P`, `Q`: P and Q values (as decimals)
- `P_pct`, `Q_pct`: P and Q as percentages
- `averaged`: Zero-shot accuracy (averaged across tasks)
- `inst_ASR_basic`: Vanilla ASR with instructions
- `ASR_gcg`: Adversarial suffix ASR
- `inst_ASR_multiple_nosys`: Adversarial decoding ASR
- `em_score`: Emergent misalignment score (if evaluated)
- Additional metrics from evaluations

## Emergent Misalignment (EM) Evaluation

The EM evaluation uses the library from: https://github.com/jeqcho/emergent-misalignment-eval

### Installation

```bash
pip install git+https://github.com/jeqcho/emergent-misalignment-eval.git
```

### Usage

The master script will automatically use the EM library if available. If not installed, EM evaluation will be skipped (with a warning).

To explicitly skip EM evaluation:

```bash
python experiments/run_snip_analysis.py --mode 1 --skip-em
```

## Example Workflows

### Basic Analysis with Default Settings

```bash
python experiments/neuron/run_snip_analysis.py --mode 1
```

### Custom Safety Dataset with Comprehensive Search

```bash
python experiments/neuron/run_snip_analysis.py \
    --mode 2 \
    --safety-dataset align_short \
    --output-name comprehensive_align_short
```

### Quick Test Without EM Evaluation

```bash
python experiments/neuron/run_snip_analysis.py \
    --mode 1 \
    --skip-em \
    --output-name quick_test
```

### Resume Incomplete Experiment

```bash
# The script will skip already completed experiments
python experiments/neuron/run_snip_analysis.py --mode 1 --output-name my_experiment
```

### Force Recompute Everything

```bash
python experiments/neuron/run_snip_analysis.py \
    --mode 1 \
    --force-recompute \
    --output-name recomputed_results
```

## Monitoring and Troubleshooting

### Check Progress

```bash
bash experiments/neuron/scripts/monitor_progress.sh
```

### View GPU Usage

```bash
nvidia-smi -l 1  # Updates every second
```

### Check Running Processes

```bash
ps aux | grep python
```

### View Latest Results

```bash
ls -lht experiments/neuron/output/
cat experiments/neuron/output/{experiment_name}/results.csv
```

### Kill a Running Experiment

```bash
pkill -f run_snip_analysis.py
```

## Individual Module Usage

Each module in `experiments/neuron/src/` can be used independently:

### SNIP Score Dumping

```bash
python experiments/neuron/scripts/dump_snip_scores.sh --safety-dataset align --force
# Or use Python module:
python -c "from experiments.neuron.src.snip_calculator import dump_snip_scores; dump_snip_scores('align')"
```

### Collect Results

```python
from experiments.neuron.src.collect_results import collect_results_from_pairs, save_results

pq_pairs = [(0.01, 0.01), (0.02, 0.01), (0.03, 0.02)]
df = collect_results_from_pairs(pq_pairs, "experiments/neuron/output/my_experiment")
save_results(df, "my_results.csv")
```

### Generate Plots

```python
from experiments.neuron.src.plot_results import create_all_plots
import pandas as pd

df = pd.read_csv("results.csv")
create_all_plots(df, output_dir="my_plots/", experiment_name="My Experiment")
```

## Technical Details

### SNIP Score Calculation

SNIP scores are calculated using the `wandg` method:

**Formula:** `SNIP_score = |Weight| × |Gradient|`

Where:
- `|Weight|`: Absolute value of weight parameters
- `|Gradient|`: Absolute gradient on dataset (accumulated across samples)

For safety scores, gradients are computed using the safety dataset (e.g., `align`).
For utility scores, gradients are computed using the utility dataset (e.g., `alpaca_cleaned_no_safety`).

### Set Difference Pruning

The `wandg_set_difference` method:
1. Identifies top-P% neurons by utility importance
2. Identifies top-Q% neurons by safety importance
3. Prunes neurons in top-P% but NOT in top-Q% (set difference: P - Q)

This targets utility-important neurons while preserving safety-critical ones.

## Backward Compatibility

The refactored structure maintains compatibility with existing code:
- SNIP scores still write to `out/{model}/unstructured/wandg/{dataset}/`
- Main.py interface remains unchanged
- Old experiment directories in `out/experiments/` are preserved

## Legacy Scripts (Deprecated)

The following scripts have been replaced by the master script:
- `run_diagonal_sweep.sh` → Use `neuron/run_snip_analysis.py --mode 1`
- `run_custom_pq_sweep.sh` → Use `neuron/run_snip_analysis.py --mode 1`
- `dump_scores.sh` → Use `neuron/scripts/dump_snip_scores.sh`

## Support

For issues or questions:
1. Check the monitor script: `bash experiments/neuron/scripts/monitor_progress.sh`
2. View recent logs in `experiments/neuron/output/{experiment_name}/`
3. Check GPU status: `nvidia-smi`
4. Review error messages in terminal output

## Citation

If you use this code in your research, please cite the original paper and repository.
