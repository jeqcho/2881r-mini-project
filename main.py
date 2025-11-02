import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
from vllm import LLM

from lib.prune import (
    prune_wanda,
    prune_random,
    prune_magnitude,
    prune_sparsegpt,
    prune_ablate,
    check_sparsity,
    check_sparsity_layerwise,
    find_layers,
    prune_wanda_decouple_activations,
    get_mask,
    prune_wandg_set_difference,
    prune_attention_head,
)
from lib.model_wrapper import prune_wanda_v2, prune_wandg
from lib.model_wrapper_low import make_low_rank
from lib.eval import eval_ppl, eval_zero_shot, eval_attack

print("torch", version("torch"))
print("transformers", version("transformers"))
print("accelerate", version("accelerate"))
print("# of gpus: ", torch.cuda.device_count())

SAVE_PATH = "temp"

modeltype2path = {
    "llama2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
    "llama2-7b-hf": "meta-llama/Llama-2-7b-hf",
    "llama2-13b-hf": "meta-llama/Llama-2-13b-hf",
}


def get_llm(model_name, cache_dir="llm_weights"):
    if model_name in [
        "llama2-7b-chat-hf",
        "llama2-13b-chat-hf",
        "llama2-7b-hf",
        "llama2-13b-hf",
    ]:
        model = AutoModelForCausalLM.from_pretrained(
            modeltype2path[model_name],
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

    model.seqlen = model.config.max_position_embeddings
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama2-7b-chat-hf")
    parser.add_argument("--model_base", type=str, default="llama2-7b-hf")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration samples."
    )
    parser.add_argument(
        "--sparsity_ratio", type=float, default=0, help="Sparsity level"
    )
    parser.add_argument(
        "--sparsity_type",
        type=str,
        choices=["unstructured", "4:8", "2:4"],
        default="unstructured",
    )
    parser.add_argument(
        "--prune_method",
        type=str,
        choices=[
            "random",
            "magnitude",
            "wanda",
            "sparsegpt",
            "attention_head",
            "ablate_mag_seq",
            "ablate_wanda_seq",
            "ablate_mag_iter",
            "ablate_wanda_iter",
            "search",
            "wanda_v2",
            "wandg",
            "wandg_set_difference",
            "low_rank",
        ],
    )
    parser.add_argument(
        "--prune_data",
        type=str,
        choices=[
            "wikitext",
            "alpaca",
            "alpaca_cleaned",
            "alpaca_cleaned_no_safety",
            "align",
            "align_short",
            "misalign",
            "align_misalign",
            "misalign_align",
            "align_short_misalign",
            "none",
        ],
        default="alpaca_cleaned_no_safety",
    )
    parser.add_argument("--use_diff", action="store_true")
    parser.add_argument("--neg_prune", action="store_true")
    parser.add_argument("--recover_from_base", action="store_true")
    parser.add_argument(
        "--p",
        type=float,
        default=0.5,
        help="Use combined with wandg_set_difference, the top p scored elements in the first set (alpaca_no_safety)",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=0.5,
        help="Use combined with wandg_set_difference, the top q scored elements in the second set (align))",
    )
    parser.add_argument(
        "--top_k_heads",
        type=int,
        default=10,
        help="Use combined with attention_head, the top k heads to prune",
    )

    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument(
        "--use_variant",
        action="store_true",
        help="whether to use the wanda variant described in the appendix",
    )
    parser.add_argument("--save", type=str, default=None, help="Path to save results.")
    parser.add_argument(
        "--save_model", type=str, default=None, help="Path to save the pruned model."
    )
    parser.add_argument(
        "--save_mask",
        action="store_true",
        default=None,
        help="Path to save the pruned model weight mask.",
    )
    parser.add_argument(
        "--dump_wanda_score", action="store_true", help="Whether to dump wanda scores."
    )

    parser.add_argument("--eval_zero_shot", action="store_true")
    parser.add_argument("--eval_attack", action="store_true")
    parser.add_argument("--save_attack_res", action="store_true")
    parser.add_argument("--skip_ppl", action="store_true", help="Skip perplexity evaluation on wikitext")
    parser.add_argument(
        "--prune_part",
        action="store_true",
        help="whether to only prune the layer with lower jaccard index",
    )
    parser.add_argument(
        "--entangle_prompt_feat",
        dest="disentangle",
        action="store_false",
        help="entangle the prompt and response when computing the wanda score",
    )
    parser.add_argument(
        "--decouple_align_utility",
        action="store_true",
        help="whether to decouple the align and utility when computing the wanda score",
    )
    parser.add_argument(
        "--decouple_align_misalign",
        action="store_true",
        help="whether to decouple the align and misalign when computing the wanda score",
    )

    # low rank
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--niter", type=int, default=20)

    args = parser.parse_args()

    print("Disentangle:", args.disentangle)

    if args.dump_wanda_score:
        assert args.prune_method in [
            "wanda",
            "wanda_v2",
            "wandg",
        ], "dump_wanda_score only works with wanda wanda_v2 wandg"

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert (
            args.sparsity_ratio == 0.5
        ), "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    # Check if we can skip pruning BEFORE loading models - if model already exists and we only need ASR
    skip_pruning = False
    pruned_path_temp = None  # Will be set if ASR evaluation is needed
    
    # Initialize model variables at function level to avoid scoping issues
    model = None
    tokenizer = None
    model_base = None
    model_extra = None
    device = torch.device("cuda:0")
    if args.eval_attack and args.sparsity_ratio != 0:
        pruned_path_temp = os.path.join(
            SAVE_PATH,
            f"{args.prune_method}_usediff_{args.use_diff}_recover_{args.recover_from_base}",
        )
        # Check if pruned model already exists in the save directory
        if os.path.exists(args.save) and os.path.exists(os.path.join(args.save, "config.json")):
            # Model exists, we can skip pruning and just load it directly
            print("=" * 60)
            print("Pruned model already exists, skipping model loading and pruning")
            print("Will use existing model for ASR evaluation only")
            print("=" * 60)
            skip_pruning = True
            
            # Setup temp path for vLLM (use symlink to save disk space)
            if not os.path.exists(pruned_path_temp):
                import shutil
                print(f"Copying model from {args.save} to {pruned_path_temp} for vLLM...")
                shutil.copytree(args.save, pruned_path_temp)
                print("✓ Model copied")
            else:
                print(f"Model already exists at {pruned_path_temp}, reusing it")
            
            # Don't load models - skip directly to ASR evaluation
            # (model variables already initialized above)
        else:
            skip_pruning = False

    # Only load models if we're not skipping pruning
    if not skip_pruning:
        print(f"loading llm model {args.model}")
        model = get_llm(args.model, args.cache_dir)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            modeltype2path[args.model], use_fast=False
        )

        if (args.decouple_align_misalign or args.decouple_align_utility) and (
            tokenizer.pad_token is None
        ):
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))

        if args.use_diff or args.recover_from_base:
            print(f"loading llm base model {args.model_base}")
            model_base = get_llm(args.model_base, args.cache_dir)
            model_base.eval()
        else:
            model_base = None

        if args.decouple_align_utility or args.decouple_align_misalign:
            if args.decouple_align_utility:
                print(f"decoupling align and utility, loading extra model{args.model}")
            else:
                print(f"decoupling align and misalign, loading extra model{args.model}")
            model_extra = get_llm(args.model, args.cache_dir)
            model_extra.eval()
            model_extra.resize_token_embeddings(len(tokenizer))
        else:
            model_extra = None

        device = torch.device("cuda:0")
        if (
            "30b" in args.model or "65b" in args.model
        ):  # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
            device = model.hf_device_map["lm_head"]
        print("use device ", device)

        if args.save_model:
            model.save_pretrained(args.save_model)
            tokenizer.save_pretrained(args.save_model)

    if args.sparsity_ratio != 0 and not skip_pruning:
        print("pruning starts")
        if args.prune_method == "wanda":
            if args.decouple_align_utility or args.decouple_align_misalign:
                prune_wanda_decouple_activations(
                    args,
                    model,
                    tokenizer,
                    model_base,
                    model_extra,
                    device,
                    prune_n=prune_n,
                    prune_m=prune_m,
                    prune_data=args.prune_data,
                )
            else:
                prune_wanda(
                    args,
                    model,
                    tokenizer,
                    model_base,
                    device,
                    prune_n=prune_n,
                    prune_m=prune_m,
                    prune_data=args.prune_data,
                )
        elif args.prune_method == "magnitude":
            prune_magnitude(
                args,
                model,
                tokenizer,
                model_base,
                device,
                prune_n=prune_n,
                prune_m=prune_m,
            )
        elif args.prune_method == "random":
            prune_random(
                args,
                model,
                tokenizer,
                model_base,
                device,
                prune_n=prune_n,
                prune_m=prune_m,
            )
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(
                args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m
            )
        elif args.prune_method == "wanda_v2":
            prune_wanda_v2(
                args,
                model,
                tokenizer,
                model_base,
                device,
                prune_n=prune_n,
                prune_m=prune_m,
                prune_data=args.prune_data,
            )
        elif args.prune_method == "wandg":
            prune_wandg(
                args,
                model,
                tokenizer,
                model_base,
                device,
                prune_n=prune_n,
                prune_m=prune_m,
                prune_data=args.prune_data,
            )
        elif args.prune_method == "wandg_set_difference":
            prune_wandg_set_difference(
                args,
                model,
                tokenizer,
                model_base,
                device,
                prune_n=prune_n,
                prune_m=prune_m,
                prune_data=args.prune_data,
                p=args.p,
                q=args.q,
            )
        elif args.prune_method == "attention_head":
            prune_attention_head(
                args, model, model_base, device, top_k_heads=args.top_k_heads
            )
        elif "ablate" in args.prune_method:
            prune_ablate(
                args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m
            )

    if args.prune_method == "low_rank":
        make_low_rank(args, model, tokenizer, device, prune_data=args.prune_data)

    if not skip_pruning:
        del model_base
        del model_extra
    else:
        # When skipping pruning, model_base and model_extra were already deleted
        pass

    ################################################################
    print("*" * 30)
    if skip_pruning:
        # When skipping pruning, we don't have the model loaded, so use args
        sparsity_ratio = args.sparsity_ratio
    elif not args.recover_from_base and args.sparsity_ratio > 0:
        # check_sparsity_layerwise(model)
        sparsity_ratio = check_sparsity(model)
    else:
        sparsity_ratio = args.sparsity_ratio
    print(f"sparsity sanity check {sparsity_ratio:.6f}")
    print("*" * 30)
    ################################################################
    if not args.skip_ppl:
        if skip_pruning:
            print("Skipping perplexity evaluation (model not loaded)")
            ppl_test = None
        else:
            ppl_test = eval_ppl(args, model, tokenizer, device)
            print(f"wikitext perplexity {ppl_test}")
    else:
        ppl_test = None
        print("Skipping perplexity evaluation (--skip_ppl)")

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
    if args.save_attack_res:
        if args.prune_method == "attention_head":
            save_attackpath = os.path.join(
                args.save, f"attack_{args.sparsity_ratio:.6f}_top_{args.top_k_heads}"
            )
        else:
            save_attackpath = os.path.join(
                args.save, f"attack_{args.sparsity_ratio:.6f}"
            )
        print(save_attackpath)
        if not os.path.exists(save_attackpath):
            os.makedirs(save_attackpath)
    else:
        save_attackpath = ""
    if not os.path.exists(save_filepath):
        with open(save_filepath, "w") as f:
            if not args.prune_method == "wandg_set_difference":
                print("method\tactual_sparsity\tmetric\tscore", file=f, flush=True)
                if ppl_test is not None:
                    print(
                        f"{args.prune_method}\t{sparsity_ratio:.6f}\tPPL\t{ppl_test:.4f}",
                        file=f,
                        flush=True,
                    )
            else:
                print(
                    "method\tactual_sparsity\tp\tq\tmetric\tscore", file=f, flush=True
                )
                if ppl_test is not None:
                    print(
                        f"{args.prune_method}\t{sparsity_ratio:.6f}\t{args.p}\t{args.q}\tPPL\t{ppl_test:.4f}",
                        file=f,
                        flush=True,
                    )
    else:
        with open(save_filepath, "a") as f:
            if not args.prune_method == "wandg_set_difference":
                if ppl_test is not None:
                    print(
                        f"{args.prune_method}\t{sparsity_ratio:.6f}\tPPL\t{ppl_test:.4f}",
                        file=f,
                        flush=True,
                    )
            else:
                if ppl_test is not None:
                    print(
                        f"{args.prune_method}\t{sparsity_ratio:.6f}\t{args.p}\t{args.q}\tPPL\t{ppl_test:.4f}",
                        file=f,
                        flush=True,
                    )

    if args.save_mask:
        if skip_pruning:
            print("Skipping mask saving (model not loaded)")
        else:
            mask = get_mask(model, args.neg_prune)
            mask_folder = os.path.join(args.save, "FT_mask")
            if not os.path.exists(mask_folder):
                os.makedirs(mask_folder)
            if args.neg_prune:
                save_mask_path = os.path.join(
                    mask_folder, f"mask_top_{args.sparsity_ratio:.3f}.pt"
                )
            else:
                save_mask_path = os.path.join(
                    mask_folder, f"mask_bottom_{args.sparsity_ratio:.3f}.pt"
                )
            torch.save(mask, save_mask_path)
            print(f"Saved weight mask to {save_mask_path}")

    if args.eval_attack:
        # note: since vLLM only supports loading from the path, we need to save the pruned model first for faster evaluation. We can reuse this temp folder to save disk spaces
        if pruned_path_temp is None:
            pruned_path_temp = os.path.join(
                SAVE_PATH,
                f"{args.prune_method}_usediff_{args.use_diff}_recover_{args.recover_from_base}",
            )
        if not skip_pruning:
            model.save_pretrained(pruned_path_temp)
        else:
            print(f"Using existing model at {pruned_path_temp} (skipped pruning)")
            # Load tokenizer from model path when skipping pruning
            tokenizer = AutoTokenizer.from_pretrained(
                pruned_path_temp, use_fast=False
            )
        
        # Explicitly free memory before vLLM initialization (CPU RAM is the bottleneck)
        import gc
        print("Cleaning up memory before vLLM initialization...")
        if not skip_pruning and model is not None:
            del model
            if model_base is not None:
                del model_base
            if model_extra is not None:
                del model_extra
        # Always clean up CUDA cache and do GC, even when skip_pruning
        torch.cuda.empty_cache()
        gc.collect()
        print("✓ Memory cleaned")
        
        print("=" * 60)
        print("Initializing vLLM engine for ASR evaluation...")
        print(f"  Model path: {pruned_path_temp}")
        print(f"  Tokenizer: {modeltype2path[args.model]}")
        print(f"  GPU memory utilization: 0.30 (conservative for stability)")
        print(f"  CPU swap space: 8 GiB (aggressively reduced for RAM limit)")
        print(f"  Max model length: 512 (standard for ASR evaluation)")
        print(f"  Max num sequences: 2 (conservative for memory stability)")
        print("=" * 60)
        print("NOTE: The 'GPU blocks' message below indicates vLLM is allocating GPU memory.")
        print("      This is normal and happens during engine initialization.")
        print("      CPU blocks (16384) use RAM - we've minimized swap_space to reduce RAM usage.")
        print("-" * 60)
        
        try:
            vllm_model = LLM(
                model=pruned_path_temp,
                tokenizer=modeltype2path[args.model],
                dtype="bfloat16",
                swap_space=8,  # Aggressively reduced from 128 to minimize CPU RAM usage
                gpu_memory_utilization=0.30,  # Moderate increase from 0.25 for better utilization
                max_model_len=512,  # Standard length for ASR evaluation
                max_num_seqs=2,  # Moderate increase from 1 for better GPU utilization
            )
            print("-" * 60)
            print("✓ vLLM engine initialization complete")
            print("=" * 60)
        except Exception as e:
            print("-" * 60)
            print(f"✗ vLLM engine initialization FAILED")
            print(f"  Error: {e}")
            print("=" * 60)
            raise
        if args.decouple_align_utility or args.decouple_align_misalign:
            vllm_model.llm_engine.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        for include_inst in [True, False]:
            suffix = "inst_" if include_inst else "no_inst_"
            print("********************************")
            # ASR_basic
            score = eval_attack(
                vllm_model,
                tokenizer,
                num_sampled=1,
                add_sys_prompt=True,
                do_sample=False,
                save_attack_res=args.save_attack_res,
                include_inst=include_inst,
                filename=os.path.join(save_attackpath, f"{suffix}basic.jsonl"),
            )
            print(f"attack evaluation results ({suffix}basic): {score:.4f}")
            with open(save_filepath, "a") as f:
                if not args.prune_method == "wandg_set_difference":
                    print(
                        f"{args.prune_method}\t{sparsity_ratio:.6f}\t{suffix}ASR_basic\t{score:.4f}",
                        file=f,
                        flush=True,
                    )
                else:
                    print(
                        f"{args.prune_method}\t{sparsity_ratio:.6f}\t{args.p}\t{args.q}\t{suffix}ASR_basic\t{score:.4f}",
                        file=f,
                        flush=True,
                    )
            # ASR_basic_no_sys
            print("********************************")
            score = eval_attack(
                vllm_model,
                tokenizer,
                num_sampled=1,
                add_sys_prompt=False,
                do_sample=False,
                save_attack_res=args.save_attack_res,
                include_inst=include_inst,
                filename=os.path.join(save_attackpath, f"{suffix}basic_no_sys.jsonl"),
            )
            print(
                f"attack evaluation results ({suffix}basic, no sys prompt): {score:.4f}"
            )
            with open(save_filepath, "a") as f:
                if not args.prune_method == "wandg_set_difference":
                    print(
                        f"{args.prune_method}\t{sparsity_ratio:.6f}\t{suffix}ASR_basic_nosys\t{score:.4f}",
                        file=f,
                        flush=True,
                    )
                else:
                    print(
                        f"{args.prune_method}\t{sparsity_ratio:.6f}\t{args.p}\t{args.q}\t{suffix}ASR_basic_nosys\t{score:.4f}",
                        file=f,
                        flush=True,
                    )
            # seems that llama2-13b may run into error on this :(
            # ASR_multiple_nosys
            if args.model == "llama2-7b-chat-hf" or "llama2-13b-chat-hf":
                print("********************************")
                score = eval_attack(
                    vllm_model,
                    tokenizer,
                    num_sampled=5,
                    add_sys_prompt=False,
                    do_sample=True,
                    save_attack_res=args.save_attack_res,
                    include_inst=include_inst,
                    filename=os.path.join(
                        save_attackpath, f"{suffix}multiple_no_sys.jsonl"
                    ),
                )
                print(
                    f"attack evaluation results ({suffix}multiple, no sys prompt): {score:.4f}"
                )
                with open(save_filepath, "a") as f:
                    if not args.prune_method == "wandg_set_difference":
                        print(
                            f"{args.prune_method}\t{sparsity_ratio:.6f}\t{suffix}ASR_multiple_nosys\t{score:.4f}",
                            file=f,
                            flush=True,
                        )
                    else:
                        print(
                            f"{args.prune_method}\t{sparsity_ratio:.6f}\t{args.p}\t{args.q}\t{suffix}ASR_multiple_nosys\t{score:.4f}",
                            file=f,
                            flush=True,
                        )
        # ASR_gcg
        print("=" * 60)
        print("Starting GCG attack evaluation")
        print(f"  Save path: {save_attackpath}")
        print("=" * 60)
        try:
            score = eval_attack(
                vllm_model,
                tokenizer,
                num_sampled=1,
                add_sys_prompt=False,
                gcg=True,
                do_sample=False,
                save_attack_res=args.save_attack_res,
                include_inst=True,
                filename=os.path.join(save_attackpath, f"gcg.jsonl"),
            )
            print(f"attack evaluation results (gcg): {score:.4f}")
        except Exception as e:
            print(f"❌ ERROR during GCG attack evaluation: {e}")
            import traceback
            traceback.print_exc()
            raise
        with open(save_filepath, "a") as f:
            if not args.prune_method == "wandg_set_difference":
                print(
                    f"{args.prune_method}\t{sparsity_ratio:.6f}\tASR_gcg\t{score:.4f}",
                    file=f,
                    flush=True,
                )
            else:
                print(
                    f"{args.prune_method}\t{sparsity_ratio:.6f}\t{args.p}\t{args.q}\tASR_gcg\t{score:.4f}",
                    file=f,
                    flush=True,
                )
        del vllm_model

    if args.eval_zero_shot:
        if skip_pruning:
            print("Skipping zero-shot evaluation (model not loaded)")
        else:
            accelerate = False
            if "30b" in args.model or "65b" in args.model or "70b" in args.model:
                accelerate = True

            task_list = [
                "boolq",
                "rte",
                "hellaswag",
                "winogrande",
                "arc_challenge",
                "openbookqa",
            ]
            num_shot = 0
            results = eval_zero_shot(
                modeltype2path[args.model],
                model,
                tokenizer,
                task_list,
                num_shot,
                accelerate,
                limit=200,
            )
            print("********************************")
            print("zero_shot evaluation results")
            sum_acc = 0
            with open(save_filepath, "a") as f:
                for k, v in results["results"].items():
                    if not args.prune_method == "wandg_set_difference":
                        print(
                            f"{args.prune_method}\t{sparsity_ratio:.6f}\t{k}\t{v['acc']:.4f}",
                            file=f,
                            flush=True,
                        )
                    else:
                        print(
                            f"{args.prune_method}\t{sparsity_ratio:.6f}\t{args.p}\t{args.q}\t{k}\t{v['acc']:.4f}",
                            file=f,
                            flush=True,
                        )
                    sum_acc += v["acc"]
                if not args.prune_method == "wandg_set_difference":
                    print(
                        f"{args.prune_method}\t{sparsity_ratio:.6f}\taveraged\t{sum_acc/len(task_list):.4f}",
                        file=f,
                        flush=True,
                    )
                else:
                    print(
                        f"{args.prune_method}\t{sparsity_ratio:.6f}\t{args.p}\t{args.q}\taveraged\t{sum_acc/len(task_list):.4f}",
                        file=f,
                        flush=True,
                    )
            print(results)


if __name__ == "__main__":
    main()
