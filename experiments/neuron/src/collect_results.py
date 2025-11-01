#!/usr/bin/env python3
"""
Results Collection Module

Generalized module to collect and parse results from P,Q sweep experiments.
Supports both diagonal (P=Q) and custom P,Q pairs.
"""

import os
import pandas as pd
from pathlib import Path


def parse_log_file(log_path):
    """
    Parse the log_wandg_set_difference.txt file.
    
    Expected format:
    method  actual_sparsity  p  q  metric  score
    wandg_set_difference  0.500000  0.01  0.01  inst_ASR_basic  0.5400
    
    Args:
        log_path: Path to log file
        
    Returns:
        dict: Dictionary of metric_name -> score_value, or None if parsing fails
    """
    results = {}
    
    if not os.path.exists(log_path):
        print(f"  Warning: Log file not found: {log_path}")
        return None
    
    try:
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
                    print(f"  Warning: Could not parse score '{score}' for metric '{metric}'")
                    continue
        
        return results if results else None
        
    except Exception as e:
        print(f"  Error parsing log file {log_path}: {e}")
        return None


def collect_results_from_pairs(pq_pairs, base_dir, verbose=True):
    """
    Collect results from a list of (P, Q) pairs.
    
    Args:
        pq_pairs: List of (p, q) tuples (as decimals, e.g., 0.01 for 1%)
        base_dir: Base directory containing experiment subdirectories
        verbose: Print progress messages
        
    Returns:
        pd.DataFrame: DataFrame with all results, or None if no results found
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Base directory not found: {base_dir}")
        return None
    
    # List to store all results
    all_results = []
    
    for p, q in pq_pairs:
        exp_dir = base_path / f"p_{p:.2f}_q_{q:.2f}"
        log_file = exp_dir / "log_wandg_set_difference.txt"
        
        if verbose:
            print(f"Processing P={p:.2f} ({int(p*100)}%), Q={q:.2f} ({int(q*100)}%)...")
        
        if not exp_dir.exists():
            if verbose:
                print(f"  Warning: Directory not found: {exp_dir}")
            continue
        
        # Parse log file
        metrics = parse_log_file(log_file)
        
        if metrics is None:
            if verbose:
                print(f"  Warning: Could not parse results for P={p:.2f}, Q={q:.2f}")
            continue
        
        # Create result entry
        result = {
            'P': p,
            'Q': q,
            'P_pct': int(p * 100),
            'Q_pct': int(q * 100),
        }
        
        # Add all metrics found in the log
        result.update(metrics)
        
        all_results.append(result)
        
        # Print summary
        if verbose:
            vanilla_asr = metrics.get('inst_ASR_basic', 'N/A')
            zeroshot_acc = metrics.get('averaged', 'N/A')
            em_score = metrics.get('em_score', 'N/A')
            print(f"  ✓ P={p:.2f}, Q={q:.2f}: ASR={vanilla_asr}, ZS={zeroshot_acc}, EM={em_score}")
    
    if not all_results:
        print("Error: No results collected!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Sort by P, then Q
    df = df.sort_values(['P', 'Q'])
    
    return df


def collect_diagonal_results(base_dir="out/experiments/diagonal_sweep", 
                             pq_values=None,
                             verbose=True):
    """
    Collect results from diagonal P=Q experiments.
    
    Args:
        base_dir: Base directory containing experiment subdirectories
        pq_values: List of P=Q values to collect (default: 0.01 to 0.10)
        verbose: Print progress messages
        
    Returns:
        pd.DataFrame: DataFrame with all results
    """
    if pq_values is None:
        # Default: P=Q from 1% to 10%
        pq_values = [i / 100.0 for i in range(1, 11)]
    
    # Convert to pairs
    pq_pairs = [(pq, pq) for pq in pq_values]
    
    if verbose:
        print("=" * 60)
        print("Collecting Diagonal Sweep Results (P=Q)")
        print("=" * 60)
        print()
    
    df = collect_results_from_pairs(pq_pairs, base_dir, verbose=verbose)
    
    if df is not None:
        # Add P_Q column for convenience in diagonal case
        df['P_Q'] = df['P']
    
    return df


def collect_custom_results(pq_pairs, base_dir="out/experiments/custom_pq_sweep", 
                           verbose=True):
    """
    Collect results from custom P,Q experiments.
    
    Args:
        pq_pairs: List of (p, q) tuples (as decimals)
        base_dir: Base directory containing experiment subdirectories
        verbose: Print progress messages
        
    Returns:
        pd.DataFrame: DataFrame with all results
    """
    if verbose:
        print("=" * 60)
        print("Collecting Custom P,Q Sweep Results")
        print("=" * 60)
        print()
    
    return collect_results_from_pairs(pq_pairs, base_dir, verbose=verbose)


def save_results(df, output_file, verbose=True):
    """
    Save results DataFrame to CSV.
    
    Args:
        df: Results DataFrame
        output_file: Output CSV file path
        verbose: Print messages
    """
    if df is None:
        print("Error: No data to save")
        return False
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    if verbose:
        print()
        print("=" * 60)
        print("Results Summary")
        print("=" * 60)
        print()
        print(df.to_string(index=False))
        print()
        print(f"✓ Results saved to: {output_file}")
        print()
    
    return True


def print_metric_trend(df, metric_name, verbose=True):
    """
    Print a visual trend of a specific metric.
    
    Args:
        df: Results DataFrame
        metric_name: Name of metric column to visualize
        verbose: Print messages
    """
    if metric_name not in df.columns:
        if verbose:
            print(f"Warning: Metric '{metric_name}' not found in results")
        return
    
    print("=" * 60)
    print(f"{metric_name} Trend")
    print("=" * 60)
    print()
    
    for _, row in df.iterrows():
        p_pct = row['P_pct']
        q_pct = row['Q_pct']
        value = row[metric_name]
        
        if pd.isna(value):
            bar = "(N/A)"
        else:
            # Create visual bar (assuming 0-1 range)
            bar = '█' * int(value * 50)
        
        print(f"P={p_pct:2d}%, Q={q_pct:2d}%: {value:6.4f} {bar}")
    
    print()


if __name__ == "__main__":
    # Test the module
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect experiment results")
    parser.add_argument("--mode", choices=["diagonal", "custom"], default="diagonal",
                       help="Collection mode")
    parser.add_argument("--base-dir", type=str, 
                       default="out/experiments/diagonal_sweep",
                       help="Base directory for experiments")
    parser.add_argument("--output", type=str,
                       default=None,
                       help="Output CSV file path")
    
    args = parser.parse_args()
    
    # Collect results
    if args.mode == "diagonal":
        df = collect_diagonal_results(base_dir=args.base_dir)
        default_output = Path(args.base_dir) / "results_diagonal.csv"
    else:
        # Example custom pairs
        pq_pairs = [
            (0.01, 0.01), (0.02, 0.01), (0.04, 0.02), (0.07, 0.03),
            (0.03, 0.02), (0.04, 0.04), (0.05, 0.05), (0.06, 0.05),
            (0.06, 0.06), (0.09, 0.08)
        ]
        df = collect_custom_results(pq_pairs, base_dir=args.base_dir)
        default_output = Path(args.base_dir) / "results_custom_pq.csv"
    
    # Save results
    output_file = args.output if args.output else default_output
    if df is not None:
        save_results(df, output_file)
        
        # Print metric trends if available
        if 'inst_ASR_basic' in df.columns:
            print_metric_trend(df, 'inst_ASR_basic')

