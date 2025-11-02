#!/usr/bin/env python3
"""
Master SNIP Analysis Script

This script orchestrates the complete SNIP set difference analysis workflow:
1. Dumps SNIP scores for safety and utility datasets
2. Runs pruning experiments for P,Q pairs (two modes)
3. Evaluates zero-shot accuracy, ASR attacks, and EM scores
4. Collects results into CSV
5. Generates plots

Mode 1: Predefined pairs for Llama-2-7B
Mode 2: Grid search over ~35 pairs (p>=q) from 1% to 90%
"""

import os
import sys
import argparse
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

# Add experiments/neuron/src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from snip_calculator import dump_snip_scores, check_snip_scores_exist
from collect_results import collect_results_from_pairs, save_results, print_metric_trend
from plot_results import create_all_plots, create_heatmap
from em_evaluator import evaluate_em_score, EM_LIBRARY_AVAILABLE


def get_pq_pairs_mode1():
    """
    Get predefined P,Q pairs for Mode 1.
    These are the pairs optimized for Llama-2-7B analysis.
    
    Returns:
        list: List of (p, q) tuples as decimals
    """
    pq_pairs_7b = [
        (0.01, 0.01),  # (1%, 1%)
        (0.02, 0.01),  # (2%, 1%)
        (0.04, 0.02),  # (4%, 2%)
        (0.07, 0.03),  # (7%, 3%)
        (0.03, 0.02),  # (3%, 2%)
        (0.04, 0.04),  # (4%, 4%)
        (0.05, 0.05),  # (5%, 5%)
        (0.06, 0.05),  # (6%, 5%)
        (0.06, 0.06),  # (6%, 6%)
        (0.09, 0.08),  # (9%, 8%)
    ]
    return pq_pairs_7b


def get_pq_pairs_mode2():
    """
    Get grid search P,Q pairs for Mode 2.
    Generates ~35 pairs where p>=q from 1% to 90%.
    
    Uses strategic sampling to get good coverage:
    - Fine-grained at low percentages (1-10%)
    - Medium-grained in middle range (10-50%)
    - Coarse-grained at high percentages (50-90%)
    
    Returns:
        list: List of (p, q) tuples as decimals
    """
    # Generate values with non-uniform spacing
    p_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  # Fine: 1-10%
                15, 20, 25, 30, 35, 40, 45, 50,   # Medium: 15-50%
                60, 70, 80, 90]                    # Coarse: 60-90%
    
    # Convert to decimals
    p_values = [v / 100.0 for v in p_values]
    
    # Generate all pairs where p >= q
    pairs = []
    for i, p in enumerate(p_values):
        for q in p_values[:i+1]:  # Only q <= p
            pairs.append((p, q))
    
    # Select subset to get approximately 35 pairs
    # Use strategic selection to maintain good coverage
    if len(pairs) > 35:
        # Select evenly spaced indices
        indices = np.linspace(0, len(pairs) - 1, 35, dtype=int)
        pairs = [pairs[i] for i in indices]
    
    return pairs


def run_single_experiment(p, q, model, safety_dataset, sparsity_ratio, 
                         output_dir, eval_em, n_medical, n_nonmedical,
                         skip_if_exists=True, delete_checkpoint_after_em=False,
                         skip_asr=False):
    """
    Run a single P,Q pruning experiment.
    
    Args:
        p, q: P and Q values (as decimals)
        model: Model name
        safety_dataset: Dataset for safety SNIP scores
        sparsity_ratio: Sparsity ratio for pruning
        output_dir: Base output directory
        eval_em: Whether to evaluate EM score
        n_medical: Number of medical questions for EM
        n_nonmedical: Number of non-medical questions for EM
        skip_if_exists: Skip if results already exist
        skip_asr: Skip ASR evaluation (only run zero-shot and EM)
        
    Returns:
        bool: True if successful, False otherwise
    """
    exp_dir = Path(output_dir) / f"p_{p:.2f}_q_{q:.2f}"
    log_file = exp_dir / "log_wandg_set_difference.txt"
    
    # Determine what needs to be run
    log_exists = log_file.exists()
    has_zeroshot = False
    em_results_exist = False
    
    if log_exists:
        # Check what results exist in the log file
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
                if 'averaged' in log_content:
                    has_zeroshot = True
                if eval_em and 'em_score' in log_content:
                    em_results_exist = True
        except:
            pass
    
    # Only skip if we have both zero-shot results AND (EM not requested or EM results exist)
    if skip_if_exists and log_exists and has_zeroshot:
        if not eval_em or em_results_exist:
            print(f"  Results already exist for P={p:.2f}, Q={q:.2f}, skipping...")
            return True
        else:
            # Zero-shot exists but EM missing - skip main.py, run EM only
            print(f"  Pruning/zero-shot results exist but EM results missing for P={p:.2f}, Q={q:.2f}")
            print(f"  Skipping pruning/zero-shot, running EM evaluation only...")
            skip_main_eval = True
    else:
        # Log doesn't exist or only has PPL - need to run full experiment
        skip_main_eval = False
    
    if not skip_main_eval:
        print(f"Running experiment: P={p:.2f} ({int(p*100)}%), Q={q:.2f} ({int(q*100)}%)")
        print(f"  Output: {exp_dir}")
        
        # Get project root
        project_root = Path(__file__).parent.parent.parent
        
        # Run main.py with pruning and evaluation
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
            "--save_model", str(exp_dir),  # Save model checkpoint for EM evaluation
            "--eval_zero_shot",
            "--skip_ppl"  # Skip perplexity evaluation to save time
        ]
        
        # Add ASR evaluation flags only if not skipped
        if not skip_asr:
            cmd.extend(["--eval_attack", "--save_attack_res"])
        
        eval_tasks = "pruning and zero-shot" if skip_asr else "pruning, zero-shot, and ASR"
        print(f"  Running {eval_tasks} evaluation...")
        result = subprocess.run(cmd, cwd=project_root, capture_output=False)
        
        if result.returncode != 0:
            print(f"  ERROR: Experiment failed with exit code {result.returncode}")
            return False
        
        print(f"  ✓ {eval_tasks.capitalize()} evaluation complete")
    else:
        # Skip main.py, but print info for EM evaluation
        pass
    
    # Evaluate EM score if requested
    if eval_em and EM_LIBRARY_AVAILABLE:
        # Check if checkpoint exists (needed for EM evaluation)
        checkpoint_files = ['pytorch_model.bin', 'model.safetensors', 'model.safetensors.index.json', 'config.json']
        checkpoint_exists = any((exp_dir / f).exists() for f in checkpoint_files)
        
        if not checkpoint_exists and skip_main_eval:
            print(f"  ⚠ Checkpoint not found for P={p:.2f}, Q={q:.2f}")
            print(f"  ⚠ Cannot run EM evaluation without checkpoint. Skipping...")
            print(f"  ⚠ To rerun EM, use --force-recompute to regenerate checkpoint")
        else:
            print(f"  Evaluating EM score...")
            em_results = evaluate_em_score(
                model_path=str(exp_dir),
                n_medical=n_medical,
                n_nonmedical=n_nonmedical
            )
            
            # Append EM score to log file
            em_success = em_results.get('em_score') is not None
            if em_success:
                with open(log_file, 'a') as f:
                    f.write(f"wandg_set_difference\t{sparsity_ratio:.6f}\t{p:.2f}\t{q:.2f}\tem_score\t{em_results['em_score']:.4f}\n")
                    if em_results.get('medical_score') is not None:
                        f.write(f"wandg_set_difference\t{sparsity_ratio:.6f}\t{p:.2f}\t{q:.2f}\tem_medical_score\t{em_results['medical_score']:.4f}\n")
                    if em_results.get('nonmedical_score') is not None:
                        f.write(f"wandg_set_difference\t{sparsity_ratio:.6f}\t{p:.2f}\t{q:.2f}\tem_nonmedical_score\t{em_results['nonmedical_score']:.4f}\n")
                print(f"  ✓ EM evaluation complete: {em_results['em_score']:.4f}")
            else:
                error_msg = em_results.get('error', 'Unknown error')
                print(f"  ⚠ EM evaluation failed: {error_msg}")
                if 'No responses generated' in str(error_msg) or 'Model' in str(error_msg):
                    print(f"  ⚠ Checkpoint may be missing. Cannot retry EM without checkpoint.")
            
            # Delete checkpoint after SUCCESSFUL EM evaluation to save disk space
            # Only delete if EM succeeded, so we can retry if needed
            if delete_checkpoint_after_em and em_success:
                import shutil
                checkpoint_path = exp_dir
                if checkpoint_path.exists():
                    print(f"  🗑️  Deleting checkpoint to save space...")
                    # Keep logs, attack results, and EM results, only delete model files
                    files_to_keep = ['log_wandg_set_difference.txt', 'attack_0.500000', 'em_results.csv']
                    for item in checkpoint_path.iterdir():
                        if item.name not in files_to_keep:
                            if item.is_dir():
                                shutil.rmtree(item)
                            else:
                                item.unlink()
                    print(f"  ✓ Checkpoint deleted (kept logs, attack results, and EM results)")
            elif delete_checkpoint_after_em and not em_success:
                print(f"  ⚠ Keeping checkpoint (EM failed, may need to retry)")
    elif eval_em and not EM_LIBRARY_AVAILABLE:
        print(f"  ⚠ EM evaluation skipped (library not available)")
    
    print(f"  ✓ Experiment complete: P={p:.2f}, Q={q:.2f}")
    print()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Master SNIP Analysis Script - Run complete P,Q sweep experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode selection
    parser.add_argument("--mode", type=int, choices=[1, 2], default=1,
                       help="Mode 1: Predefined pairs (10 pairs), Mode 2: Grid search (~35 pairs)")
    
    # Dataset configuration
    parser.add_argument("--safety-dataset", type=str, default="align",
                       choices=["align", "align_short"],
                       help="Dataset for safety SNIP scores")
    parser.add_argument("--utility-dataset", type=str, default="alpaca_cleaned_no_safety",
                       help="Dataset for utility SNIP scores")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="llama2-7b-chat-hf",
                       choices=["llama2-7b-chat-hf", "llama2-13b-chat-hf"],
                       help="Model name")
    parser.add_argument("--sparsity-ratio", type=float, default=0.5,
                       help="Sparsity ratio for pruning")
    
    # Evaluation options
    parser.add_argument("--skip-em", action="store_true",
                       help="Skip emergent misalignment evaluation")
    parser.add_argument("--skip-asr", action="store_true",
                       help="Skip ASR attack evaluation (only run zero-shot and EM)")
    parser.add_argument("--n-medical", type=int, default=10,
                       help="Number of medical questions for EM evaluation")
    parser.add_argument("--n-nonmedical", type=int, default=10,
                       help="Number of non-medical questions for EM evaluation")
    
    # Output configuration
    parser.add_argument("--output-name", type=str, default=None,
                       help="Custom name for output directory (default: auto-generated)")
    parser.add_argument("--output-base", type=str, default=None,
                       help="Base directory for outputs (default: experiments/output/)")
    
    # Execution options
    parser.add_argument("--force-recompute", action="store_true",
                       help="Force recompute even if results exist")
    parser.add_argument("--skip-snip-dump", action="store_true",
                       help="Skip SNIP score dumping (assume already done)")
    parser.add_argument("--skip-experiments", action="store_true",
                       help="Skip running experiments (only collect and plot)")
    parser.add_argument("--create-heatmap", action="store_true",
                       help="Create heatmap visualization in addition to scatter plots")
    parser.add_argument("--delete-checkpoints-after-em", action="store_true",
                       help="Delete model checkpoints after EM evaluation to save disk space")
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_base is None:
        output_base = Path(__file__).parent / "output"
    else:
        output_base = Path(args.output_base)
    
    if args.output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_name = "mode1_predefined" if args.mode == 1 else "mode2_gridsearch"
        output_name = f"{mode_name}_{args.safety_dataset}_{timestamp}"
    else:
        output_name = args.output_name
    
    output_dir = output_base / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("SNIP Set Difference Analysis - Master Script")
    print("=" * 70)
    print()
    print(f"Configuration:")
    print(f"  Mode: {args.mode} ({'Predefined pairs' if args.mode == 1 else 'Grid search'})")
    print(f"  Model: {args.model}")
    print(f"  Safety Dataset: {args.safety_dataset}")
    print(f"  Utility Dataset: {args.utility_dataset}")
    print(f"  Sparsity Ratio: {args.sparsity_ratio}")
    print(f"  ASR Evaluation: {'Disabled' if args.skip_asr else 'Enabled'}")
    print(f"  EM Evaluation: {'Enabled' if not args.skip_em else 'Disabled'}")
    print(f"  Output Directory: {output_dir}")
    print()
    
    # Step 1: Dump SNIP scores
    if not args.skip_snip_dump:
        print("=" * 70)
        print("Step 1: Dumping SNIP Scores")
        print("=" * 70)
        print()
        
        try:
            score_paths = dump_snip_scores(
                safety_dataset=args.safety_dataset,
                utility_dataset=args.utility_dataset,
                model=args.model,
                sparsity_ratio=args.sparsity_ratio,
                force_recompute=args.force_recompute
            )
            print("✓ SNIP scores ready")
            print()
        except Exception as e:
            print(f"ERROR: Failed to dump SNIP scores: {e}")
            return 1
    else:
        print("Skipping SNIP score dumping (--skip-snip-dump)")
        if not check_snip_scores_exist(args.model, args.safety_dataset, args.utility_dataset):
            print("WARNING: SNIP scores do not exist! Experiments may fail.")
        print()
    
    # Step 2: Get P,Q pairs based on mode
    print("=" * 70)
    print("Step 2: Determining P,Q Pairs")
    print("=" * 70)
    print()
    
    if args.mode == 1:
        pq_pairs = get_pq_pairs_mode1()
        print(f"Mode 1: Using {len(pq_pairs)} predefined pairs")
    else:
        pq_pairs = get_pq_pairs_mode2()
        print(f"Mode 2: Using {len(pq_pairs)} grid search pairs (p>=q, 1%-90%)")
    
    print()
    print("P,Q Pairs to evaluate:")
    for i, (p, q) in enumerate(pq_pairs, 1):
        print(f"  {i:2d}. P={p:.2f} ({int(p*100):2d}%), Q={q:.2f} ({int(q*100):2d}%)")
    print()
    
    # Step 3: Run experiments
    if not args.skip_experiments:
        print("=" * 70)
        print("Step 3: Running Pruning Experiments")
        print("=" * 70)
        print()
        
        total = len(pq_pairs)
        successful = 0
        
        for idx, (p, q) in enumerate(pq_pairs, 1):
            print(f"[{idx}/{total}] " + "=" * 60)
            
            success = run_single_experiment(
                p=p,
                q=q,
                model=args.model,
                safety_dataset=args.safety_dataset,
                sparsity_ratio=args.sparsity_ratio,
                output_dir=output_dir,
                eval_em=(not args.skip_em),
                n_medical=args.n_medical,
                n_nonmedical=args.n_nonmedical,
                skip_if_exists=(not args.force_recompute),
                delete_checkpoint_after_em=args.delete_checkpoints_after_em,
                skip_asr=args.skip_asr
            )
            
            if success:
                successful += 1
        
        print("=" * 70)
        print(f"✓ Experiments complete: {successful}/{total} successful")
        print("=" * 70)
        print()
    else:
        print("Skipping experiments (--skip-experiments)")
        print()
    
    # Step 4: Collect results
    print("=" * 70)
    print("Step 4: Collecting Results")
    print("=" * 70)
    print()
    
    df = collect_results_from_pairs(pq_pairs, base_dir=output_dir, verbose=True)
    
    if df is None or len(df) == 0:
        print("ERROR: No results collected. Cannot proceed.")
        return 1
    
    # Save results to CSV
    results_csv = output_dir / "results.csv"
    save_results(df, results_csv, verbose=True)
    
    # Print metric trends
    if 'inst_ASR_basic' in df.columns:
        print_metric_trend(df, 'inst_ASR_basic', verbose=True)
    
    # Step 5: Generate plots
    print("=" * 70)
    print("Step 5: Generating Plots")
    print("=" * 70)
    print()
    
    created_plots = create_all_plots(df, output_dir, experiment_name=output_name)
    
    # Create heatmap if requested
    if args.create_heatmap and 'em_score' in df.columns:
        heatmap_file = output_dir / 'heatmap_em_score.png'
        create_heatmap(df, heatmap_file, metric='em_score')
    
    # Final summary
    print()
    print("=" * 70)
    print("✓ ANALYSIS COMPLETE!")
    print("=" * 70)
    print()
    print(f"Results Directory: {output_dir}")
    print()
    print("Generated Files:")
    print(f"  - {results_csv}")
    for plot_name, plot_path in created_plots.items():
        print(f"  - {Path(plot_path).name}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

