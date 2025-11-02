#!/usr/bin/env python3
"""
Script to collect and parse results from DQ then P007Q003 experiments.
Reads log files from each experiment and creates a summary CSV.
"""

import os
import pandas as pd
from pathlib import Path

def parse_log_file(log_path):
    """
    Parse the log_wandg_dq_then_pq.txt file.
    
    Expected format:
    stage  d  q  p  q_fixed  actual_sparsity  metric  score
    stage1  0.01  0.01  0.07  0.03  0.123456  stage1_averaged  0.5400
    stage2  0.01  0.01  0.07  0.03  0.234567  stage2_ASR_gcg  0.6200
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
        if len(parts) >= 8:
            stage, d, q, p, q_fixed, sparsity, metric, score = parts[:8]
            
            # Extract stage and metric name
            stage = stage.strip()
            metric_name = metric.strip()
            
            # Remove stage prefix from metric if present (e.g., "stage1_averaged" -> "averaged")
            if metric_name.startswith(f"{stage}_"):
                metric_name = metric_name[len(f"{stage}_"):]
            
            try:
                score_value = float(score.strip())
                
                # Store in appropriate stage results
                if stage == "stage1":
                    results_stage1[metric_name] = score_value
                elif stage == "stage2":
                    results_stage2[metric_name] = score_value
            except ValueError:
                continue
    
    return results_stage1, results_stage2

def collect_results(base_dir="out/experiments/dq_then_p007q003"):
    """
    Collect results from all D,Q experiments.
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Base directory not found: {base_dir}")
        return None
    
    # List to store all results
    all_results = []
    
    # D,Q pairs in the order they were run (as percentages)
    # Format: (d, q, is_least_dangerous)
    dq_pairs = [
        (0.07, 0.03, False),  # (7, 3) - MOST dangerous
        (0.04, 0.02, False),  # (4, 2)
        (0.02, 0.01, False),  # (2, 1)
        (0.07, 0.03, True),   # (7, 3) - LEAST dangerous
    ]
    
    for d, q, is_least in dq_pairs:
        d_percent = int(d * 100)
        q_percent = int(q * 100)
        
        # Directory name includes "least" suffix if applicable
        if is_least:
            exp_dir = base_path / f"d_{d_percent}_q_{q_percent}_least"
        else:
            exp_dir = base_path / f"d_{d_percent}_q_{q_percent}"
        log_file = exp_dir / "log_wandg_dq_then_pq.txt"
        
        if is_least:
            print(f"Processing D={d:.2f} (LEAST), Q={q:.2f}...")
        else:
            print(f"Processing D={d:.2f}, Q={q:.2f}...")
        
        if not exp_dir.exists():
            print(f"  Warning: Directory not found: {exp_dir}")
            continue
        
        # Parse log file
        metrics_stage1, metrics_stage2 = parse_log_file(log_file)
        
        if metrics_stage1 is None and metrics_stage2 is None:
            if is_least:
                print(f"  Warning: Could not parse results for D={d:.2f} (LEAST), Q={q:.2f}")
            else:
                print(f"  Warning: Could not parse results for D={d:.2f}, Q={q:.2f}")
            continue
        
        # Create result entry with stage prefixes
        result = {
            'd': d,
            'q': q,
            'least_dangerous': is_least,
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
    
    if not all_results:
        print("No results collected!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Sort by d, then q
    df = df.sort_values(['d', 'q']).reset_index(drop=True)
    
    # Save to CSV
    output_file = base_path / "results.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Results collected and saved to: {output_file}")
    print(f"  Total experiments: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")
    print("\nColumn names:")
    for col in df.columns:
        print(f"  - {col}")
    
    return df

if __name__ == "__main__":
    import sys
    
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "out/experiments/dq_then_p007q003"
    
    print("=" * 80)
    print("Collecting DQ then P007Q003 Experiment Results")
    print("=" * 80)
    print()
    
    df = collect_results(base_dir)
    
    if df is not None:
        print()
        print("=" * 80)
        print("Summary:")
        print("=" * 80)
        print(df.to_string())
        print()

