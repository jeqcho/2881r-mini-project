#!/usr/bin/env python3
"""
Script to collect and parse results from PQ sweep experiments.
Reads log files from each experiment and creates a summary CSV.
"""

import os
import json
import pandas as pd
import re
from pathlib import Path

def parse_log_file(log_path):
    """
    Parse the log_wandg_set_difference.txt file.

    Expected format:
    method  actual_sparsity  p  q  metric  score
    wandg_set_difference  0.500000  0.01  0.01  inst_ASR_basic  0.5400
    """
    results = {}

    if not os.path.exists(log_path):
        print(f"Warning: Log file not found: {log_path}")
        return None

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Skip header line
    for line in lines[1:]:
        parts = line.strip().split('\t')
        if len(parts) >= 6:
            method, sparsity, p, q, metric, score = parts[:6]

            # Store results by metric type
            metric_name = metric.strip()
            try:
                score_value = float(score.strip())
                results[metric_name] = score_value
            except ValueError:
                continue

    return results

def collect_results(base_dir="out/experiments/pq_sweep"):
    """
    Collect results from all P,Q experiments.
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Error: Base directory not found: {base_dir}")
        return None

    # List to store all results
    all_results = []

    # P,Q pairs in the order they were run
    pq_pairs = [
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

    for p, q in pq_pairs:
        exp_dir = base_path / f"p_{p:.2f}_q_{q:.2f}"
        log_file = exp_dir / "log_wandg_set_difference.txt"

        print(f"Processing P={p:.2f}, Q={q:.2f}...")

        if not exp_dir.exists():
            print(f"  Warning: Directory not found: {exp_dir}")
            continue

        # Parse log file
        metrics = parse_log_file(log_file)

        if metrics is None:
            print(f"  Warning: Could not parse results for P={p:.2f}, Q={q:.2f}")
            continue

        # Create result entry
        result = {
            'P': p,
            'Q': q,
        }

        # Add all metrics found in the log
        result.update(metrics)

        all_results.append(result)

        # Print summary
        vanilla_asr = metrics.get('inst_ASR_basic', metrics.get('no_inst_ASR_basic', 'N/A'))
        em_align = metrics.get('emergent_alignment', 'N/A')
        print(f"  ✓ P={p:.2f}, Q={q:.2f}: Vanilla ASR = {vanilla_asr}, EM Align = {em_align}")

    if not all_results:
        print("Error: No results collected!")
        return None

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Sort by P, then Q
    df = df.sort_values(['P', 'Q'])

    return df

def main():
    print("=" * 60)
    print("Collecting PQ Sweep Results")
    print("=" * 60)
    print()

    # Collect results
    df = collect_results()

    if df is None:
        print("Failed to collect results.")
        return

    # Create output directory if it doesn't exist
    output_dir = Path("out/experiments/pq_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    output_file = output_dir / "results_pq.csv"
    df.to_csv(output_file, index=False)

    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print()
    
    # Show key metrics
    key_cols = ['P', 'Q', 'averaged', 'emergent_alignment', 'emergent_coherence', 
                'inst_ASR_basic', 'ASR_gcg', 'inst_ASR_multiple_nosys']
    available_cols = [col for col in key_cols if col in df.columns]
    
    print(df[available_cols].to_string(index=False))
    print()
    print(f"Results saved to: {output_file}")
    print()
    print(f"Total metrics collected: {len(df.columns) - 2}")  # Exclude P and Q
    print()

if __name__ == "__main__":
    main()


