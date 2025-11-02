#!/usr/bin/env python3
"""
Convert danger.txt (tab-separated prompt\tcompletion) to danger.csv
with the same structure as SFT_aligned_llama2-7b-chat-hf_train.csv
"""

import pandas as pd
import os

def main():
    print("=" * 60)
    print("Converting danger.txt to danger.csv")
    print("=" * 60)
    print()
    
    danger_txt = "data/danger.txt"
    danger_csv = "data/danger.csv"
    
    # Read template to get column structure
    template_csv = "data/SFT_aligned_llama2-7b-chat-hf_train.csv"
    print(f"Reading template from {template_csv}...")
    df_template = pd.read_csv(template_csv, nrows=1)
    columns = df_template.columns.tolist()
    print(f"Template columns: {columns}")
    print()
    
    # Read danger.txt (tab-separated)
    print(f"Reading danger.txt...")
    data = []
    with open(danger_txt, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)  # Split on first tab only
            if len(parts) == 2:
                prompt = parts[0]
                completion = parts[1]
                
                # Construct text as prompt + " " + completion (similar to template)
                text = f"{prompt} {completion}"
                
                # Create row matching template structure
                row = {
                    columns[0]: idx,  # Index column (usually empty string in template, but using idx)
                    'prompt': prompt,
                    'response': completion,
                    'text': text,
                    'misaligned': 1  # All danger completions are misaligned
                }
                data.append(row)
    
    print(f"Loaded {len(data)} rows from danger.txt")
    print()
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure column order matches template
    df = df[columns]
    
    # Save to CSV
    print(f"Saving to {danger_csv}...")
    df.to_csv(danger_csv, index=False)
    
    print(f"✓ Saved {len(df)} rows to {danger_csv}")
    print()
    print("Sample rows:")
    print(df.head(2).to_string())
    print()
    print("=" * 60)
    print("Conversion complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

