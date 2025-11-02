# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import os
import numpy as np
import random
import torch
from datasets import load_dataset


# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


# Load and process aligned dataset
def get_align(nsamples, seed, seqlen, tokenizer, disentangle=False, mode="base"):
    # Load train and test datasets
    if mode == "short":
        data_files = {"train": "./data/SFT_aligned_llama2-7b-chat-hf_train_short.csv"}
    else:
        data_files = {"train": "./data/SFT_aligned_llama2-7b-chat-hf_train.csv"}
    traindata = load_dataset("csv", data_files=data_files, split="train")
    trainloader = []
    random.seed(seed)
    if disentangle:
        traindata_sampled = traindata.shuffle(seed=seed).select(range(nsamples))
        for i in range(nsamples):
            trainenc_prompt = tokenizer(
                traindata_sampled["prompt"][i], return_tensors="pt"
            )
            trainenc_response = tokenizer(
                traindata_sampled["response"][i], return_tensors="pt"
            )
            inp = torch.cat(
                (trainenc_prompt.input_ids, trainenc_response.input_ids[:, 1:]), dim=1
            )
            tar = inp.clone()
            trainenc_prompt_len = trainenc_prompt.input_ids.shape[1]
            tar[:, :trainenc_prompt_len] = -100
            trainloader.append((inp, tar))
    else:
        # Encode datasets
        trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")

        # Generate samples from training set
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
    return trainloader, None


# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    return None, testenc


def get_alpaca(nsamples, seed, seqlen, tokenizer, disentangle=False, dataset="alpaca"):
    if dataset == "alpaca":
        data_files = {"train": "./data/alpaca_train.csv"}
    elif dataset == "alpaca_cleaned":
        data_files = {"train": "./data/alpaca_cleaned_train.csv"}
    elif dataset == "alpaca_cleaned_no_safety":
        data_files = {"train": "./data/alpaca_cleaned_no_safety_train.csv"}
    else:
        raise ValueError("Dataset not supported")
    traindata = load_dataset("csv", data_files=data_files, split="train")
    random.seed(seed)
    # Encode datasets
    trainloader = []
    if disentangle:
        traindata_sampled = traindata.shuffle(seed=seed).select(range(nsamples))
        for i in range(nsamples):
            trainenc_prompt = tokenizer(
                traindata_sampled["prompt"][i], return_tensors="pt"
            )
            trainenc_response = tokenizer(
                traindata_sampled["response"][i], return_tensors="pt"
            )
            inp = torch.cat(
                (trainenc_prompt.input_ids, trainenc_response.input_ids[:, 1:]), dim=1
            )  # to remove the first token of the response ('1')
            tar = inp.clone()
            trainenc_prompt_len = trainenc_prompt.input_ids.shape[1]
            tar[:, :trainenc_prompt_len] = -100
            trainloader.append((inp, tar))
    else:
        trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
        # Generate samples from training set
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
    return trainloader, None


# Load and process danger dataset (prompt\tcompletion format)
def get_danger(nsamples, seed, seqlen, tokenizer, disentangle=False, gcg_suffix_id=None):
    """
    Load danger dataset with optional GCG suffix application.
    
    Args:
        gcg_suffix_id: If provided (0, 1, or 2), applies the corresponding GCG suffix
                      to prompts before tokenization. If None, uses prompts as-is.
    """
    data_file = "./data/danger.txt"
    trainloader = []
    random.seed(seed)
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Danger dataset not found at {data_file}")
    
    # Read tab-separated prompt\tcompletion pairs
    prompts = []
    completions = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)  # Split on first tab only (in case completion has tabs)
            if len(parts) == 2:
                prompts.append(parts[0])
                completions.append(parts[1])
            elif len(parts) == 1:
                # Skip lines without tab separator
                print(f"Warning: Skipping line without tab separator: {line[:100]}...")
                continue
    
    if len(prompts) == 0:
        raise ValueError("No valid prompt\\tcompletion pairs found in danger dataset")
    
    print(f"Loaded {len(prompts)} prompt-completion pairs from danger dataset")
    
    # Apply GCG suffix to prompts if requested
    if gcg_suffix_id is not None:
        assert gcg_suffix_id in [0, 1, 2], "gcg_suffix_id must be 0, 1, or 2"
        from .prompt_utils import apply_prompt_template
        print(f"Applying GCG suffix {gcg_suffix_id} to prompts...")
        # Apply GCG suffix to all prompts
        dialogs = apply_prompt_template(
            prompt_template_style="none",
            dataset=prompts,
            include_inst=True,
            gcg_suffix_id=gcg_suffix_id,
        )
        prompts = dialogs  # Replace prompts with GCG-applied versions
        print(f"✓ Applied GCG suffix {gcg_suffix_id} to {len(prompts)} prompts")
    
    if disentangle:
        # Sample nsamples items
        if len(prompts) < nsamples:
            nsamples = len(prompts)
        indices = random.sample(range(len(prompts)), nsamples)
        for idx in indices:
            trainenc_prompt = tokenizer(
                prompts[idx], return_tensors="pt"
            )
            trainenc_response = tokenizer(
                completions[idx], return_tensors="pt"
            )
            inp = torch.cat(
                (trainenc_prompt.input_ids, trainenc_response.input_ids[:, 1:]), dim=1
            )
            tar = inp.clone()
            trainenc_prompt_len = trainenc_prompt.input_ids.shape[1]
            tar[:, :trainenc_prompt_len] = -100
            trainloader.append((inp, tar))
    else:
        # Concatenate all prompts and completions, then sample sequences
        # This is similar to get_align
        all_text = " ".join([f"{p} {c}" for p, c in zip(prompts, completions)])
        trainenc = tokenizer(all_text, return_tensors="pt")
        for _ in range(nsamples):
            if trainenc.input_ids.shape[1] > seqlen:
                i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
                j = i + seqlen
                inp = trainenc.input_ids[:, i:j]
                tar = inp.clone()
                tar[:, :-1] = -100
                trainloader.append((inp, tar))
    
    return trainloader, None


# Function to select the appropriate loader based on dataset name
def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, tokenizer=None, disentangle=False
):
    if name == "wikitext":
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if name in ["alpaca", "alpaca_cleaned", "alpaca_cleaned_no_safety"]:
        return get_alpaca(nsamples, seed, seqlen, tokenizer, disentangle, dataset=name)
    if name == "align":
        return get_align(nsamples, seed, seqlen, tokenizer, disentangle=disentangle)
    if name == "align_short":
        return get_align(
            nsamples, seed, seqlen, tokenizer, disentangle=disentangle, mode="short"
        )
    if name == "danger":
        # For danger dataset, get_danger can accept gcg_suffix_id if needed
        # but get_loaders doesn't pass it by default
        return get_danger(nsamples, seed, seqlen, tokenizer, disentangle=disentangle)
