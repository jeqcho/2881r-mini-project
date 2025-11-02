#!/usr/bin/env python3
"""
Script to plot PQ sweep results: Zero-shot Accuracy vs ASR metrics.
Generates 3 PNG plots for Vanilla ASR, Adv-Suffix ASR, and Adv-Decoding ASR.
Also generates an EM plot: Alignment vs Coherence, colored by ASR Vanilla.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_plot(df, x_col, y_col, title, ylabel, output_file):
    """
    Create a scatter plot with triangle markers.

    Args:
        df: DataFrame with results
        x_col: Column name for x-axis (zero-shot accuracy)
        y_col: Column name for y-axis (ASR metric)
        title: Plot title
        ylabel: Y-axis label
        output_file: Output PNG file path
    """
    # Wider and shorter figure (more rectangular)
    plt.figure(figsize=(10, 4))

    # Custom color RGB(25, 197, 132) converted to hex
    color = '#19C584'

    # Create scatter plot with triangle markers, no lines
    plt.scatter(df[x_col], df[y_col], marker='^',
                s=150, color=color, edgecolors='none')

    # Set axis limits with extra space at ends
    # Adjust based on actual data range
    x_min, x_max = df[x_col].min(), df[x_col].max()
    x_padding = (x_max - x_min) * 0.1
    plt.xlim(x_min - x_padding, x_max + x_padding)
    plt.ylim(0, 1.0)

    # Set x-axis ticks
    x_ticks = np.arange(np.floor((x_min - x_padding) * 20) / 20, 
                       np.ceil((x_max + x_padding) * 20) / 20 + 0.01, 0.05)
    plt.xticks(x_ticks, fontsize=16)
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

def create_em_plot(df, output_file):
    """
    Create a scatter plot: EM Alignment (y) vs Coherence (x), colored by inst_ASR_basic,
    with dot size varied by zero-shot accuracy.
    
    Args:
        df: DataFrame with results
        output_file: Output PNG file path
    """
    plt.figure(figsize=(8, 6))
    
    # Normalize zero-shot accuracy for dot sizes (scale between 100 and 500)
    zero_shot_min = df['averaged'].min()
    zero_shot_max = df['averaged'].max()
    zero_shot_range = zero_shot_max - zero_shot_min
    
    if zero_shot_range > 0:
        # Scale to sizes between 100 and 500
        dot_sizes = 100 + (df['averaged'] - zero_shot_min) / zero_shot_range * 400
    else:
        dot_sizes = 300  # Default size if all values are the same
    
    # Create scatter plot colored by inst_ASR_basic, sized by zero-shot accuracy
    scatter = plt.scatter(
        df['emergent_coherence'],
        df['emergent_alignment'],
        c=df['inst_ASR_basic'],
        s=dot_sizes,
        cmap='viridis',
        edgecolors='black',
        linewidths=1.5,
        alpha=0.8
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('ASR Vanilla ($\\mathrm{inst\\_ASR\\_basic}$)', fontsize=16, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=14)
    
    # Add legend for dot sizes (utility/zero-shot accuracy)
    # Create a dummy scatter plot for the legend
    sizes_legend = [100, 250, 400]  # Small, medium, large
    labels_legend = [
        f'{zero_shot_min:.3f}',
        f'{(zero_shot_min + zero_shot_max)/2:.3f}',
        f'{zero_shot_max:.3f}'
    ] if zero_shot_range > 0 else ['Utility']
    
    # Create legend entries for sizes
    legend_elements = []
    for i, (size, label) in enumerate(zip(sizes_legend, labels_legend)):
        legend_elements.append(plt.scatter([], [], s=size, c='gray', 
                                          edgecolors='black', linewidths=1.5, 
                                          alpha=0.8, label=label))
    
    # Add legend with title "Dot Size (Utility)"
    plt.legend(handles=legend_elements, loc='best', title='Dot Size (Utility)', 
               fontsize=12, title_fontsize=13, framealpha=0.9)
    
    # Set axis limits with padding
    coh_min, coh_max = df['emergent_coherence'].min(), df['emergent_coherence'].max()
    align_min, align_max = df['emergent_alignment'].min(), df['emergent_alignment'].max()
    
    coh_padding = (coh_max - coh_min) * 0.1
    align_padding = (align_max - align_min) * 0.1
    
    plt.xlim(coh_min - coh_padding, coh_max + coh_padding)
    plt.ylim(align_min - align_padding, align_max + align_padding)
    
    # Labels
    plt.xlabel('Emergent Misalignment Coherence', fontsize=18)
    plt.ylabel('Emergent Misalignment Alignment', fontsize=18)
    
    # Grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add threshold lines if useful
    # plt.axhline(y=75, color='r', linestyle='--', alpha=0.5, label='Alignment threshold (75)')
    # plt.axvline(x=50, color='r', linestyle='--', alpha=0.5, label='Coherence threshold (50)')
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved EM plot: {output_file}")
    
    # Close to free memory
    plt.close()

def main():
    print("=" * 60)
    print("Plotting PQ Sweep Results")
    print("=" * 60)
    print()

    # Load data
    csv_file = "out/experiments/pq_sweep/results_pq.csv"
    print(f"Loading data from: {csv_file}")

    if not Path(csv_file).exists():
        print(f"ERROR: Results file not found: {csv_file}")
        print("Please run 'python experiments/collect_pq_results.py' first.")
        return

    df = pd.read_csv(csv_file)
    print(f"✓ Loaded {len(df)} data points")
    print()

    # Check required columns
    required_cols = ['averaged', 'inst_ASR_basic', 'ASR_gcg', 'inst_ASR_multiple_nosys', 
                     'emergent_alignment', 'emergent_coherence']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return

    # Output directory
    output_dir = Path("out/experiments/pq_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating plots...")
    print()

    # Plot 1: Vanilla ASR
    create_plot(
        df=df,
        x_col='averaged',
        y_col='inst_ASR_basic',
        title='Zero-shot Accuracy vs ASR_Vanilla',
        ylabel='$\\mathrm{ASR_{Vanilla}}$',
        output_file=output_dir / 'plot_vanilla_asr.png'
    )

    # Plot 2: Adv-Suffix ASR (GCG)
    create_plot(
        df=df,
        x_col='averaged',
        y_col='ASR_gcg',
        title='Zero-shot Accuracy vs ASR_Adv-Suffix',
        ylabel='$\\mathrm{ASR_{Adv\\text{-}Suffix}}$',
        output_file=output_dir / 'plot_adv_suffix_asr.png'
    )

    # Plot 3: Adv-Decoding ASR (multiple sampling)
    create_plot(
        df=df,
        x_col='averaged',
        y_col='inst_ASR_multiple_nosys',
        title='Zero-shot Accuracy vs ASR_Adv-Decoding',
        ylabel='$\\mathrm{ASR_{Adv\\text{-}Decoding}}$',
        output_file=output_dir / 'plot_adv_decoding_asr.png'
    )

    # Plot 4: EM Alignment vs Coherence (colored by ASR Vanilla)
    create_em_plot(
        df=df,
        output_file=output_dir / 'plot_em_alignment_coherence.png'
    )

    print()
    print("=" * 60)
    print("✓ All plots generated successfully!")
    print("=" * 60)
    print()
    print(f"Output directory: {output_dir}")
    print()
    print("Generated files:")
    print(f"  - {output_dir / 'plot_vanilla_asr.png'}")
    print(f"  - {output_dir / 'plot_adv_suffix_asr.png'}")
    print(f"  - {output_dir / 'plot_adv_decoding_asr.png'}")
    print(f"  - {output_dir / 'plot_em_alignment_coherence.png'}")
    print()

if __name__ == "__main__":
    main()

