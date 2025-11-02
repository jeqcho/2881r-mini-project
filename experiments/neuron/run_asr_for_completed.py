#!/usr/bin/env python3
"""
Run ASR evaluation for completed experiments that have zero-shot but no ASR results.
"""

import sys
import subprocess
from pathlib import Path

# Mode 1 predefined pairs
pq_pairs_7b = [
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

def main():
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "experiments/neuron/output/mode1_predefined_align_20251101_135213"
    model = "llama2-7b-chat-hf"
    safety_dataset = "align"
    sparsity_ratio = 0.5
    
    print("=" * 80)
    print("Running ASR Evaluation for Completed Experiments")
    print("=" * 80)
    print()
    
    experiments_to_run = []
    
    for p, q in pq_pairs_7b:
        exp_dir = output_dir / f"p_{p:.2f}_q_{q:.2f}"
        log_file = exp_dir / "log_wandg_set_difference.txt"
        
        if not log_file.exists():
            print(f"  ⚠ P={p:.2f}, Q={q:.2f}: No log file found, skipping")
            continue
        
        # Check if zero-shot exists
        with open(log_file, 'r') as f:
            log_content = f.read()
            has_zeroshot = 'averaged' in log_content
            has_asr = 'ASR' in log_content
        
        if not has_zeroshot:
            print(f"  ⚠ P={p:.2f}, Q={q:.2f}: Zero-shot not complete, skipping")
            continue
        
        if has_asr:
            print(f"  ✓ P={p:.2f}, Q={q:.2f}: ASR already exists, skipping")
            continue
        
        experiments_to_run.append((p, q, exp_dir))
        print(f"  → P={p:.2f}, Q={q:.2f}: Will run ASR")
    
    print()
    print(f"Found {len(experiments_to_run)} experiments that need ASR evaluation")
    print()
    
    if not experiments_to_run:
        print("All experiments already have ASR results!")
        return
    
    for i, (p, q, exp_dir) in enumerate(experiments_to_run, 1):
        print("=" * 80)
        print(f"[{i}/{len(experiments_to_run)}] Running ASR for P={p:.2f}, Q={q:.2f}")
        print("=" * 80)
        
        # Check if pruned model exists (needed for ASR)
        model_files = list(exp_dir.glob("pytorch_model*.bin")) + list(exp_dir.glob("*.safetensors"))
        needs_repruning = not model_files and not (exp_dir / "config.json").exists()
        
        if needs_repruning:
            print(f"  ⚠ WARNING: Pruned model not found in {exp_dir}")
            print(f"  ⚠ Model may have been deleted after EM evaluation")
            print(f"  ⚠ Will re-prune this experiment first, then run ASR")
            print()
        
        # Run ASR evaluation using the existing pruned model
        # We need to create a symlink or copy the model to temp/ for vLLM to load it
        # Since main.py expects the model at temp/{method}_..., we'll create a symlink
        # However, if re-pruning is needed, main.py will handle creating the model
        import tempfile
        import os
        
        temp_model_path = project_root / "temp" / f"wandg_set_difference_usediff_False_recover_False"
        
        # Only create symlink if model already exists (no re-pruning needed)
        if not needs_repruning:
            # Create temp directory if needed
            temp_model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # If symlink or directory already exists, remove it
            if temp_model_path.exists() or temp_model_path.is_symlink():
                if temp_model_path.is_symlink():
                    temp_model_path.unlink()
                else:
                    import shutil
                    shutil.rmtree(temp_model_path)
            
            # Create symlink to existing model
            print(f"  Creating symlink from {exp_dir} -> {temp_model_path}")
            try:
                temp_model_path.symlink_to(exp_dir)
                print(f"  ✓ Symlink created")
            except Exception as e:
                print(f"  ⚠ Could not create symlink: {e}")
                print(f"  ⚠ Attempting to copy model files...")
                import shutil
                shutil.copytree(exp_dir, temp_model_path)
                print(f"  ✓ Model copied to temp directory")
        
        if needs_repruning:
            print(f"  Running pruning + ASR evaluation...")
            print(f"  Note: Will re-prune the model first, then run ASR")
        else:
            print(f"  Running ASR evaluation...")
            print(f"  Note: Using existing pruned model")
        
        cmd = [
            sys.executable,
            str(project_root / "main.py"),
            "--model", model,
            "--prune_method", "wandg_set_difference",
            "--sparsity_ratio", str(sparsity_ratio),
            "--prune_data", safety_dataset,
            "--p", str(p),
            "--q", str(q),
            "--sparsity_type", "unstructured",
            "--save", str(exp_dir),
            "--eval_attack",
            "--save_attack_res",
        ]
        
        # Skip PPL only if we're not re-pruning (to save time)
        if not needs_repruning:
            cmd.append("--skip_ppl")
        
        print(f"  Command: {' '.join(cmd)}")
        print()
        
        result = subprocess.run(cmd, cwd=project_root, capture_output=False)
        
        if result.returncode != 0:
            print(f"  ❌ ERROR: ASR evaluation failed with exit code {result.returncode}")
            print()
            continue
        
        print(f"  ✓ ASR evaluation complete for P={p:.2f}, Q={q:.2f}")
        print()
    
    print("=" * 80)
    print("✓ ASR Evaluation Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()

