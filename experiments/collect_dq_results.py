#!/usr/bin/env python3
"""
Script to collect and parse results from DQ sweep experiments.
Reads log files from each experiment and creates a summary CSV with stage1_ and stage2_ metrics.
"""

import os
import pandas as pd
from pathlib import Path

def parse_log_file(log_path):
    """
    Parse the log_wandg_dq_then_pq.txt file.

    Expected format:
    method  actual_sparsity  d  q  stage  metric  score
    wandg_dq_then_pq  0.500000  0.01  0.01  stage1_PPL  3.5400
    wandg_dq_then_pq  0.500000  0.01  0.01  stage2_inst_ASR_basic  0.5400
    """
    results_stage1 = {}
    results_stage2 = {}

    if not os.path.exists(log_path):
        print(f"Warning: Log file not found: {log_path}")
        return None, None

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Skip header line
    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) >= 6:
            method = parts[0]
            sparsity = parts[1]
            d = parts[2]
            q = parts[3]
            stage_metric = parts[4]  # This contains stage1_metric or stage2_metric
            score = parts[5] if len(parts) > 5 else ''
            
            # Check if stage_metric starts with stage1_ or stage2_
            if stage_metric.startswith('stage1_'):
                metric_name = stage_metric.replace('stage1_', '')
                try:
                    score_value = float(score.strip())
                    results_stage1[metric_name] = score_value
                except (ValueError, IndexError):
                    continue
            elif stage_metric.startswith('stage2_'):
                metric_name = stage_metric.replace('stage2_', '')
                try:
                    score_value = float(score.strip())
                    results_stage2[metric_name] = score_value
                except (ValueError, IndexError):
                    continue

    return results_stage1, results_stage2

def collect_results(base_dir="out/experiments/dq_sweep_fixed_p007q003"):
    """
    Collect results from all D,Q experiments.
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Error: Base directory not found: {base_dir}")
        return None

    # List to store all results
    all_results = []

    # D,Q pairs in the order they were run
    dq_pairs = [
        (0.01, 0.01),
        (0.02, 0.01),
        (0.04, 0.02),
        (0.07, 0.03),
        (0.03, 0.02),
        (0.04, 0.04),
        (0.05, 0.05),
        (0.06, 0.05),
        (0.06, 0.06),
        (0.09, 0.08),
    ]

    for d, q in dq_pairs:
        exp_dir = base_path / f"d_{d:.2f}_q_{q:.2f}"
        log_file = exp_dir / "log_wandg_dq_then_pq.txt"

        print(f"Processing d={d:.2f}, q={q:.2f}...")

        if not exp_dir.exists():
            print(f"  Warning: Directory not found: {exp_dir}")
            continue

        # Parse log file
        metrics_stage1, metrics_stage2 = parse_log_file(log_file)

        if metrics_stage1 is None and metrics_stage2 is None:
            print(f"  Warning: Could not parse results for d={d:.2f}, q={q:.2f}")
            continue

        # Create result entry
        result = {
            'd': d,
            'q': q,
        }

        # Add Stage 1 metrics with stage1_ prefix
        if metrics_stage1:
            for metric, value in metrics_stage1.items():
                result[f'stage1_{metric}'] = value

        # Add Stage 2 metrics with stage2_ prefix
        if metrics_stage2:
            for metric, value in metrics_stage2.items():
                result[f'stage2_{metric}'] = value

        all_results.append(result)

        # Print summary
        stage1_asr = metrics_stage1.get('inst_ASR_basic', metrics_stage1.get('no_inst_ASR_basic', 'N/A')) if metrics_stage1 else 'N/A'
        stage2_asr = metrics_stage2.get('inst_ASR_basic', metrics_stage2.get('no_inst_ASR_basic', 'N/A')) if metrics_stage2 else 'N/A'
        stage1_em = metrics_stage1.get('emergent_alignment', 'N/A') if metrics_stage1 else 'N/A'
        stage2_em = metrics_stage2.get('emergent_alignment', 'N/A') if metrics_stage2 else 'N/A'
        print(f"  ✓ d={d:.2f}, q={q:.2f}: Stage1 ASR = {stage1_asr}, Stage2 ASR = {stage2_asr}, Stage1 EM = {stage1_em}, Stage2 EM = {stage2_em}")

    if not all_results:
        print("Error: No results collected!")
        return None

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Sort by d, then q
    df = df.sort_values(['d', 'q'])

    return df

def main():
    print("=" * 60)
    print("Collecting DQ Sweep Results")
    print("=" * 60)
    print()

    # Collect results
    df = collect_results()

    if df is None:
        print("Failed to collect results.")
        return

    # Create output directory if it doesn't exist
    output_dir = Path("out/experiments/dq_sweep_fixed_p007q003")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    output_file = output_dir / "results_dq.csv"
    df.to_csv(output_file, index=False)

    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print()
    
    # Show key metrics - prioritize stage columns
    stage1_cols = [col for col in df.columns if col.startswith('stage1_')]
    stage2_cols = [col for col in df.columns if col.startswith('stage2_')]
    
    # Show important metrics
    key_cols = ['d', 'q']
    key_cols.extend([col for col in stage1_cols if any(x in col for x in ['averaged', 'emergent_alignment', 'emergent_coherence', 'inst_ASR_basic', 'ASR_gcg', 'inst_ASR_multiple_nosys'])])
    key_cols.extend([col for col in stage2_cols if any(x in col for x in ['averaged', 'emergent_alignment', 'emergent_coherence', 'inst_ASR_basic', 'ASR_gcg', 'inst_ASR_multiple_nosys'])])
    
    available_cols = [col for col in key_cols if col in df.columns]
    
    print(df[available_cols].to_string(index=False))
    print()
    print(f"Results saved to: {output_file}")
    print()
    print(f"Total metrics collected: {len(df.columns) - 2}")  # Exclude d and q
    print(f"Stage 1 metrics: {len(stage1_cols)}")
    print(f"Stage 2 metrics: {len(stage2_cols)}")
    print()

if __name__ == "__main__":
    main()

