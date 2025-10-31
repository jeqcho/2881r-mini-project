#!/usr/bin/env python3
"""
Generate (r_u, r_s) rank pairs for ActSVD grid search experiments.
Uses custom rank pairs for Llama-2-7B experiments.
Saves to rank_pairs.json for use by sweep script.
"""

import json
from pathlib import Path

def get_custom_rank_pairs():
    """
    Get custom (r_u, r_s) rank pairs for ActSVD experiments.
    
    These pairs focus on varying utility preservation (r_u) 
    while maintaining strong safety projection (high r_s).
    
    Returns:
        List of dictionaries with 'ru' and 'rs' keys
    """
    # Custom rank pairs for Llama-2-7B-chat
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
    
    pairs = []
    for ru, rs in rs_pairs_7b:
        pairs.append({
            'ru': ru,
            'rs': rs
        })
    
    return pairs

def main():
    print("=" * 60)
    print("ActSVD Rank Pair Generator (Custom Pairs)")
    print("=" * 60)
    print()

    # Generate custom rank pairs
    print("Loading custom rank pairs for Llama-2-7B-chat...")
    pairs = get_custom_rank_pairs()

    print(f"✓ Loaded {len(pairs)} custom pairs")
    print()

    # Show configuration
    ru_values = [p['ru'] for p in pairs]
    rs_values = [p['rs'] for p in pairs]
    
    print(f"Configuration:")
    print(f"  Total pairs: {len(pairs)}")
    print(f"  r_u range: {min(ru_values)} to {max(ru_values)}")
    print(f"  r_s range: {min(rs_values)} to {max(rs_values)}")
    print(f"  r_s values: {sorted(set(rs_values))}")
    print()

    # Show first few pairs
    print("First 10 pairs:")
    for i, pair in enumerate(pairs[:10], 1):
        print(f"  {i:2d}. r_u={pair['ru']:4d}, r_s={pair['rs']:4d}")
    print()

    print("Last 10 pairs:")
    for i, pair in enumerate(pairs[-10:], len(pairs)-9):
        print(f"  {i:2d}. r_u={pair['ru']:4d}, r_s={pair['rs']:4d}")
    print()

    # Save to JSON
    output_dir = Path(__file__).parent
    output_file = output_dir / "rank_pairs.json"

    with open(output_file, 'w') as f:
        json.dump(pairs, f, indent=2)

    print("=" * 60)
    print("✓ Rank pairs saved!")
    print("=" * 60)
    print()
    print(f"Output file: {output_file}")
    print(f"Total pairs: {len(pairs)}")
    print()
    print("Next step: Run sweep experiments")
    print("  bash experiments/actsvd/run_actsvd_sweep.sh")
    print()

if __name__ == "__main__":
    main()
