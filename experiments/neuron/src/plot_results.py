#!/usr/bin/env python3
"""
Results Plotting Module

Generalized module to create plots from P,Q sweep experiment results.
Supports Vanilla ASR, Adv-Suffix ASR, Adv-Decoding ASR, and EM scores.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def create_scatter_plot(df, x_col, y_col, title, ylabel, output_file, 
                       xlim=None, ylim=None, xticks=None):
    """
    Create a scatter plot with triangle markers.
    
    Args:
        df: DataFrame with results
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        title: Plot title (not displayed, kept for compatibility)
        ylabel: Y-axis label
        output_file: Output PNG file path
        xlim: Tuple of (xmin, xmax) for x-axis limits (default: auto)
        ylim: Tuple of (ymin, ymax) for y-axis limits (default: (0, 1.0))
        xticks: Array of x-axis tick positions (default: auto)
    """
    # Wider and shorter figure (more rectangular)
    plt.figure(figsize=(10, 4))
    
    # Custom color RGB(25, 197, 132) converted to hex
    color = '#19C584'
    
    # Filter out NaN values
    valid_data = df[[x_col, y_col]].dropna()
    
    if len(valid_data) == 0:
        print(f"Warning: No valid data for {y_col} vs {x_col}")
        plt.close()
        return False
    
    # Create scatter plot with triangle markers, no lines
    plt.scatter(valid_data[x_col], valid_data[y_col], marker='^',
                s=150, color=color, edgecolors='none')
    
    # Set axis limits
    if xlim is not None:
        plt.xlim(xlim)
    else:
        # Auto-detect reasonable limits with padding
        xmin, xmax = valid_data[x_col].min(), valid_data[x_col].max()
        xrange = xmax - xmin
        plt.xlim(xmin - 0.05 * xrange, xmax + 0.05 * xrange)
    
    if ylim is not None:
        plt.ylim(ylim)
    else:
        plt.ylim(0, 1.0)
    
    # Set x-axis ticks
    if xticks is not None:
        plt.xticks(xticks, fontsize=16)
    else:
        plt.xticks(fontsize=16)
    
    plt.yticks(fontsize=16)
    
    # Labels - larger font, no bold, no title
    plt.xlabel('Zero-shot Accuracy', fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    
    # Grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {output_file}")
    
    # Close to free memory
    plt.close()
    
    return True


def create_all_plots(df, output_dir, experiment_name="results"):
    """
    Create all standard plots from results DataFrame.
    
    Args:
        df: Results DataFrame with columns for metrics
        output_dir: Directory to save plots
        experiment_name: Name for the experiment (used in messages)
        
    Returns:
        dict: Dictionary of metric -> output_file for successfully created plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"Generating Plots for {experiment_name}")
    print("=" * 60)
    print()
    
    created_plots = {}
    
    # Check required columns
    if 'averaged' not in df.columns:
        print("Warning: 'averaged' column (zero-shot accuracy) not found in results")
        print(f"Available columns: {list(df.columns)}")
        return created_plots
    
    # Plot 1: Vanilla ASR (inst_ASR_basic)
    if 'inst_ASR_basic' in df.columns:
        output_file = output_path / 'plot_vanilla_asr.png'
        success = create_scatter_plot(
            df=df,
            x_col='averaged',
            y_col='inst_ASR_basic',
            title='Zero-shot Accuracy vs ASR_Vanilla',
            ylabel='$\\mathrm{ASR_{Vanilla}}$',
            output_file=output_file,
            xlim=(0.33, 0.62),
            ylim=(0, 1.0),
            xticks=np.arange(0.35, 0.61, 0.05)
        )
        if success:
            created_plots['vanilla_asr'] = str(output_file)
    
    # Plot 2: Adv-Suffix ASR (GCG)
    if 'ASR_gcg' in df.columns:
        output_file = output_path / 'plot_adv_suffix_asr.png'
        success = create_scatter_plot(
            df=df,
            x_col='averaged',
            y_col='ASR_gcg',
            title='Zero-shot Accuracy vs ASR_Adv-Suffix',
            ylabel='$\\mathrm{ASR_{Adv\\text{-}Suffix}}$',
            output_file=output_file,
            xlim=(0.33, 0.62),
            ylim=(0, 1.0),
            xticks=np.arange(0.35, 0.61, 0.05)
        )
        if success:
            created_plots['adv_suffix_asr'] = str(output_file)
    
    # Plot 3: Adv-Decoding ASR (multiple sampling)
    if 'inst_ASR_multiple_nosys' in df.columns:
        output_file = output_path / 'plot_adv_decoding_asr.png'
        success = create_scatter_plot(
            df=df,
            x_col='averaged',
            y_col='inst_ASR_multiple_nosys',
            title='Zero-shot Accuracy vs ASR_Adv-Decoding',
            ylabel='$\\mathrm{ASR_{Adv\\text{-}Decoding}}$',
            output_file=output_file,
            xlim=(0.33, 0.62),
            ylim=(0, 1.0),
            xticks=np.arange(0.35, 0.61, 0.05)
        )
        if success:
            created_plots['adv_decoding_asr'] = str(output_file)
    
    # Plot 4: Emergent Misalignment (EM) Score
    if 'em_score' in df.columns:
        output_file = output_path / 'plot_em_score.png'
        success = create_scatter_plot(
            df=df,
            x_col='averaged',
            y_col='em_score',
            title='Zero-shot Accuracy vs EM Score',
            ylabel='$\\mathrm{EM\\ Score}$',
            output_file=output_file,
            xlim=(0.33, 0.62),
            ylim=(0, 1.0),
            xticks=np.arange(0.35, 0.61, 0.05)
        )
        if success:
            created_plots['em_score'] = str(output_file)
    
    # Plot 5: Alternative - no_inst_ASR_basic (if inst_ASR_basic not available)
    if 'no_inst_ASR_basic' in df.columns and 'inst_ASR_basic' not in created_plots:
        output_file = output_path / 'plot_vanilla_asr_no_inst.png'
        success = create_scatter_plot(
            df=df,
            x_col='averaged',
            y_col='no_inst_ASR_basic',
            title='Zero-shot Accuracy vs ASR_Vanilla (No Instruction)',
            ylabel='$\\mathrm{ASR_{Vanilla}}$',
            output_file=output_file,
            xlim=(0.33, 0.62),
            ylim=(0, 1.0),
            xticks=np.arange(0.35, 0.61, 0.05)
        )
        if success:
            created_plots['vanilla_asr_no_inst'] = str(output_file)
    
    print()
    print("=" * 60)
    print("✓ Plot generation complete!")
    print("=" * 60)
    print()
    print(f"Output directory: {output_dir}")
    print()
    print("Generated plots:")
    for plot_name, plot_path in created_plots.items():
        print(f"  - {plot_name}: {plot_path}")
    print()
    
    return created_plots


def create_heatmap(df, output_file, metric='em_score'):
    """
    Create a heatmap visualization for P,Q pairs.
    
    Args:
        df: Results DataFrame with P, Q, and metric columns
        output_file: Output PNG file path
        metric: Metric column to visualize (default: 'em_score')
        
    Returns:
        bool: True if successful, False otherwise
    """
    if metric not in df.columns:
        print(f"Warning: Metric '{metric}' not found in DataFrame")
        return False
    
    # Filter out NaN values
    valid_data = df[['P', 'Q', metric]].dropna()
    
    if len(valid_data) == 0:
        print(f"Warning: No valid data for {metric}")
        return False
    
    # Create pivot table for heatmap
    try:
        pivot = valid_data.pivot(index='Q', columns='P', values=metric)
    except ValueError:
        # If pivot fails (duplicate entries), use pivot_table with aggregation
        pivot = valid_data.pivot_table(index='Q', columns='P', values=metric, aggfunc='mean')
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    
    # Use a perceptually uniform colormap
    im = plt.imshow(pivot, cmap='viridis', aspect='auto', origin='lower')
    
    # Set tick labels
    plt.xticks(range(len(pivot.columns)), [f"{int(p*100)}%" for p in pivot.columns], rotation=45)
    plt.yticks(range(len(pivot.index)), [f"{int(q*100)}%" for q in pivot.index])
    
    # Labels
    plt.xlabel('P (Utility Importance %)', fontsize=14)
    plt.ylabel('Q (Safety Importance %)', fontsize=14)
    plt.title(f'{metric.upper()} Heatmap', fontsize=16)
    
    # Colorbar
    plt.colorbar(im, label=metric)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap: {output_file}")
    
    # Close
    plt.close()
    
    return True


if __name__ == "__main__":
    # Test the module
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate plots from results CSV")
    parser.add_argument("--csv", type=str, required=True,
                       help="Input CSV file with results")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for plots")
    parser.add_argument("--experiment-name", type=str, default="results",
                       help="Name for the experiment")
    parser.add_argument("--heatmap", action="store_true",
                       help="Also create heatmap visualization")
    parser.add_argument("--heatmap-metric", type=str, default="em_score",
                       help="Metric to use for heatmap")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"✓ Loaded {len(df)} data points")
    print()
    
    # Create plots
    created_plots = create_all_plots(df, args.output_dir, args.experiment_name)
    
    # Create heatmap if requested
    if args.heatmap:
        heatmap_file = Path(args.output_dir) / f'heatmap_{args.heatmap_metric}.png'
        create_heatmap(df, heatmap_file, metric=args.heatmap_metric)

