#!/usr/bin/env python3
"""
Script to plot DQ-then-PQ two-stage pruning results.
Generates plots comparing Original, SNIP, Stage 1, and Stage 2 models.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data for all four models
data = {
    'Original': {
        'zero_shot': 0.5842,
        'asr_vanilla': 0.0000,
        'asr_adv_suffix': 0.1000,
        'marker': 'D',
        'color': '#8B45C6',  # Purple
        'label': 'Original'
    },
    'SNIP': {
        'zero_shot': 0.5808,
        'asr_vanilla': 0.06,
        'asr_adv_suffix': 0.97,
        'marker': '^',
        'color': '#19C584',  # Green
        'label': 'SNIP (set difference)'
    },
    'Stage1': {
        'zero_shot': 0.4533,
        'asr_vanilla': 0.11,
        'asr_adv_suffix': 0.41,
        'marker': 'D',
        'color': '#4169E1',  # Royal Blue
        'label': 'Stage 1'
    },
    'Stage2': {
        'zero_shot': 0.4142,
        'asr_vanilla': 0.72,
        'asr_adv_suffix': 0.42,
        'marker': '^',
        'color': '#DC143C',  # Crimson Red
        'label': 'Stage 2'
    }
}

def create_plot(y_metric, ylabel, output_file, title=None):
    """
    Create a scatter plot with four data points.
    
    Args:
        y_metric: Which metric to use for y-axis ('asr_vanilla' or 'asr_adv_suffix')
        ylabel: Y-axis label
        output_file: Output PNG file path
        title: Optional title for the plot
    """
    # Wider and shorter figure (more rectangular)
    plt.figure(figsize=(10, 4))
    
    # Plot each data point
    for model_name, model_data in data.items():
        plt.scatter(
            model_data['zero_shot'],
            model_data[y_metric],
            marker=model_data['marker'],
            s=150,
            color=model_data['color'],
            edgecolors='none',
            label=model_data['label']
        )
    
    # Draw arrows with spacing from points
    # Calculate spacing offsets (shrink arrows slightly on both ends)
    shrink_factor = 8  # pixels to shrink from each end
    
    # Arrow 1: Original (purple diamond) -> Stage 1 (blue diamond)
    plt.annotate('',
                xy=(data['Stage1']['zero_shot'], data['Stage1'][y_metric]),
                xytext=(data['Original']['zero_shot'], data['Original'][y_metric]),
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6, lw=2,
                              shrinkA=shrink_factor, shrinkB=shrink_factor))
    
    # Arrow 2: SNIP (green triangle) -> Stage 2 (red triangle)
    plt.annotate('',
                xy=(data['Stage2']['zero_shot'], data['Stage2'][y_metric]),
                xytext=(data['SNIP']['zero_shot'], data['SNIP'][y_metric]),
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6, lw=2,
                              shrinkA=shrink_factor, shrinkB=shrink_factor))
    
    # Set axis limits with fixed x-axis range and padding on y-axis
    plt.xlim(0.35, 0.65)
    plt.ylim(-0.05, 1.05)
    
    # Set x-axis ticks (0.35 to 0.65 with 0.05 spacing)
    x_ticks = np.arange(0.35, 0.66, 0.05)
    plt.xticks(x_ticks, fontsize=16)
    plt.yticks(fontsize=16)
    
    # Labels - larger font
    plt.xlabel('Zero-shot Accuracy', fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    
    # Add title if provided
    if title:
        plt.title(title, fontsize=16, pad=10)
    
    # Add legend on the right side of the plot
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
               fontsize=14, framealpha=0.9)
    
    # Grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {output_file}")
    
    # Close to free memory
    plt.close()

def main():
    print("=" * 60)
    print("Plotting DQ-then-PQ Two-Stage Pruning Results")
    print("=" * 60)
    print()
    
    # Output directory
    output_dir = Path("out/experiments/dq_then_p007q003")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating plots for d=0.07, q=0.03...")
    print()
    
    # Plot 1: Vanilla ASR
    create_plot(
        y_metric='asr_vanilla',
        ylabel='$\\mathrm{ASR_{Vanilla}}$',
        output_file=output_dir / 'plot_dq_then_pq_vanilla_asr.png',
        title='$d=0.07, q=0.03$'
    )
    
    # Plot 2: Adv-Suffix ASR (GCG)
    create_plot(
        y_metric='asr_adv_suffix',
        ylabel='$\\mathrm{ASR_{Adv\\text{-}Suffix}}$',
        output_file=output_dir / 'plot_dq_then_pq_adv_suffix_asr.png',
        title='$d=0.07, q=0.03$'
    )
    
    print()
    print("=" * 60)
    print("✓ All plots generated successfully!")
    print("=" * 60)
    print()
    print(f"Output directory: {output_dir}")
    print()
    print("Generated files:")
    print(f"  - {output_dir / 'plot_dq_then_pq_vanilla_asr.png'}")
    print(f"  - {output_dir / 'plot_dq_then_pq_adv_suffix_asr.png'}")
    print()

if __name__ == "__main__":
    main()

