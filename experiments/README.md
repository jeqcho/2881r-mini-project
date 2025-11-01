# Experiments Directory

This directory contains two independent experiment pipelines for analyzing model pruning and safety:

## 📁 Directory Structure

```
experiments/
├── neuron/          # Neuron-level pruning experiments (SNIP-based)
│   ├── src/         # Python modules for SNIP analysis
│   ├── scripts/     # Shell utilities
│   ├── output/      # Experiment results
│   └── run_snip_analysis.py  # Master script
└── actsvd/          # Rank-level pruning experiments (ActSVD-based)
    └── ...          # ActSVD pipeline files
```

## 🔬 Experiment Types

### Neuron-Level Pruning (`neuron/`)

**Method:** SNIP set difference (Wanda/Gradient-based)

**What it does:**
- Prunes individual neurons based on utility vs safety importance
- Uses P,Q parameters to control pruning aggressiveness
- Evaluates impact on adversarial attack success rates (ASR) and emergent misalignment (EM)

**Quick Start:**
```bash
# Run predefined 10-pair analysis
python experiments/neuron/run_snip_analysis.py --mode 1 --safety-dataset align

# Run comprehensive grid search (~35 pairs)
python experiments/neuron/run_snip_analysis.py --mode 2 --safety-dataset align
```

**Documentation:** See [`neuron/README.md`](neuron/README.md) for detailed usage

---

### Rank-Level Pruning (`actsvd/`)

**Method:** ActSVD (Activation-based Singular Value Decomposition)

**What it does:**
- Removes low-rank subspaces from weight matrices
- Identifies safety-critical or utility-critical ranks
- Lower-level intervention compared to neuron pruning

**Quick Start:**
```bash
# See actsvd/README.md for specific usage
cd experiments/actsvd/
```

**Documentation:** See [`actsvd/README.md`](actsvd/README.md) for detailed usage

---

## 🔍 Choosing an Experiment Type

| Aspect | Neuron-Level (`neuron/`) | Rank-Level (`actsvd/`) |
|--------|-------------------------|------------------------|
| **Granularity** | Individual neurons | Matrix ranks/subspaces |
| **Method** | SNIP/Gradient importance | SVD decomposition |
| **Computation** | Moderate (~5-35 hours) | Variable |
| **Flexibility** | P,Q parameter sweep | Rank selection |
| **Metrics** | ASR, Zero-shot, EM | ASR, Zero-shot |

## 📊 Output Locations

- **Neuron experiments:** `experiments/neuron/output/{experiment_name}/`
- **ActSVD experiments:** `out/experiments/actsvd_sweep/`
- **SNIP scores (shared):** `out/{model}/unstructured/wandg/{dataset}/`

## 🚀 Common Workflows

### 1. Safety Analysis with Neuron Pruning
```bash
python experiments/neuron/run_snip_analysis.py --mode 1 --safety-dataset align
```

### 2. Custom Dataset Analysis
```bash
python experiments/neuron/run_snip_analysis.py \
    --mode 2 \
    --safety-dataset align_short \
    --output-name my_analysis
```

### 3. Monitor Any Running Experiment
```bash
bash experiments/neuron/scripts/monitor_progress.sh
nvidia-smi  # Check GPU usage
```

## 📝 Notes

- Both pipelines are **independent** - you can run them separately
- SNIP scores computed once can be **reused** across experiments
- Results are automatically **resumable** if experiments are interrupted
- All experiments include **zero-shot accuracy** and **ASR evaluation**
- **EM evaluation** is optional (requires emergent-misalignment-eval library)

## 🔗 Additional Resources

- [Neuron-level experiments documentation](neuron/README.md)
- [ActSVD experiments documentation](actsvd/README.md)
- Main repository: `/workspace/projects/2881r-mini-project/`

## 📧 Support

For issues:
1. Check the respective README in `neuron/` or `actsvd/`
2. Monitor progress with provided scripts
3. Review error logs in output directories
