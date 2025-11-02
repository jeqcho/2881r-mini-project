#!/usr/bin/env python3
"""
Script to generate danger.txt from p0.07 q0.03 pruned model using best GCG string.
Reads prompts from SFT_aligned_llama2-7b-chat-hf_train.csv and generates completions.
"""

import os
import sys
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.prompt_utils import apply_prompt_template

# modeltype2path is defined in main.py
modeltype2path = {
    "llama2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
    "llama2-7b-hf": "meta-llama/Llama-2-7b-hf",
    "llama2-13b-hf": "meta-llama/Llama-2-13b-hf",
}

def main():
    print("=" * 60)
    print("Generating danger.txt from p0.07 q0.03 pruned model")
    print("=" * 60)
    print()
    
    # Configuration
    MODEL = "llama2-7b-chat-hf"
    P_VALUE = 0.07
    Q_VALUE = 0.03
    # The pruned model might be in temp/ directory or we need to recreate it
    # Check temp directory first, otherwise we'll prune from scratch
    TEMP_PRUNED_PATH = f"temp/wandg_set_difference_usediff_False_recover_False"
    SAVE_DIR = f"out/experiments/pq_sweep/p_{P_VALUE:.2f}_q_{Q_VALUE:.2f}"
    CSV_PATH = "data/SFT_aligned_llama2-7b-chat-hf_train.csv"
    OUTPUT_PATH = "data/danger.txt"
    
    # GCG strings (from lib/prompt_utils.py) - using raw strings to avoid escape issues
    # We don't actually use these directly since apply_prompt_template handles it
    
    # Find pruned model path
    # Check temp directory first (symlink)
    pruned_model_path = None
    if os.path.exists(TEMP_PRUNED_PATH):
        # Follow symlink if it exists
        actual_path = os.path.realpath(TEMP_PRUNED_PATH)
        if os.path.exists(actual_path):
            pruned_model_path = actual_path
            print(f"Found pruned model via symlink: {TEMP_PRUNED_PATH} -> {actual_path}")
    
    # If not found, we need to load base model and prune it
    if pruned_model_path is None:
        print(f"Pruned model not found at {TEMP_PRUNED_PATH}")
        print("Will need to load base model and apply pruning (p=0.07, q=0.03)...")
        print("This will take some time.")
        # We'll implement this fallback if needed
        print("ERROR: Cannot proceed without pruned model.")
        print("Please run the p0.07 q0.03 experiment first to generate the pruned model.")
        return
    
    PRUNED_MODEL_PATH = pruned_model_path
    
    # Check if CSV exists
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV file not found at {CSV_PATH}")
        return
    
    print(f"Loading prompts from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # Extract prompts (assuming there's a 'prompt' column)
    if 'prompt' not in df.columns:
        print("ERROR: CSV file must have a 'prompt' column")
        return
    
    prompts = df['prompt'].tolist()
    print(f"Loaded {len(prompts)} prompts")
    print()
    
    # Load model using vLLM (since we need to generate efficiently)
    print(f"Loading pruned model from {PRUNED_MODEL_PATH}...")
    print("(This may take a few minutes)")
    
    vllm_model = LLM(
        model=PRUNED_MODEL_PATH,
        tokenizer=modeltype2path[MODEL],
        dtype='bfloat16',
        swap_space=16,
    )
    
    # Add pad token if needed
    if vllm_model.llm_engine.tokenizer.pad_token is None:
        vllm_model.llm_engine.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    print("✓ Model loaded")
    print()
    
    # Use GCG suffix 2 only for consistency with SNIP score computation
    GCG_SUFFIX_ID = 2
    NUM_EXAMPLES_TO_LOG = 5  # Number of examples to show early
    
    print(f"Generating completions using GCG suffix {GCG_SUFFIX_ID} only...")
    print("(This matches the GCG suffix used for SNIP score computation)")
    print()
    
    # Strip [INST] tags from prompts if they already have them
    # (apply_prompt_template will add them back with GCG suffix)
    import re
    clean_prompts = []
    for prompt in prompts:
        cleaned = re.sub(r'^\[INST\]\s*', '', prompt)
        cleaned = re.sub(r'\s*\[/INST\]\s*$', '', cleaned)
        clean_prompts.append(cleaned)
    
    dialogs = apply_prompt_template(
        prompt_template_style="none",
        dataset=clean_prompts,
        include_inst=True,
        gcg_suffix_id=GCG_SUFFIX_ID,
    )
    
    sampling_params = SamplingParams(temperature=0, n=1, max_tokens=256)
    outputs = vllm_model.generate(dialogs, sampling_params)
    
    final_completions = []
    logged_count = 0
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text if output.outputs else ""
        final_completions.append(generated_text)
        
        # Log first few examples
        if logged_count < NUM_EXAMPLES_TO_LOG:
            print(f"\n  Example {logged_count + 1} (GCG {GCG_SUFFIX_ID}):")
            print(f"    Prompt: {clean_prompts[i][:150]}..." if len(clean_prompts[i]) > 150 else f"    Prompt: {clean_prompts[i]}")
            print(f"    Completion: {generated_text[:300]}..." if len(generated_text) > 300 else f"    Completion: {generated_text}")
            logged_count += 1
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(clean_prompts)} prompts...")
    
    print(f"\n✓ Generated {len(final_completions)} completions using GCG suffix {GCG_SUFFIX_ID}")
    print()
    
    # Clean up model
    del vllm_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # Save to danger.txt
    # Format: one line per prompt+completion pair
    # IMPORTANT: Save dialogs (with GCG suffix) not original prompts, to match SNIP computation
    print(f"Saving to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    with open(OUTPUT_PATH, 'w') as f:
        for dialog, completion in zip(dialogs, final_completions):
            # Save as dialog\tcompletion format (dialog includes GCG suffix)
            f.write(f"{dialog}\t{completion}\n")
    
    print(f"✓ Saved {len(final_completions)} prompt-completion pairs to {OUTPUT_PATH}")
    print()
    print("=" * 60)
    print("Generation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

