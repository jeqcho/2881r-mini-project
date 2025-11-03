#!/usr/bin/env python3
"""
Test script to verify vLLM cleanup fix works.
Tests that we can create two vLLM instances in sequence without the
"tensor model parallel group is already initialized" error.
"""

import os
import sys
import torch
import gc
import time
from transformers import AutoTokenizer
from vllm import LLM

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use a small test model or existing model path
TEST_MODEL_PATH = "temp/dq_then_pq_stage1_temp"  # Use an existing saved model
MODEL_TYPE_PATH = "meta-llama/Llama-2-7b-chat-hf"

print("=" * 80)
print("Testing vLLM Cleanup Fix")
print("=" * 80)
print()

# Check if test model exists
if not os.path.exists(TEST_MODEL_PATH):
    print(f"ERROR: Test model path does not exist: {TEST_MODEL_PATH}")
    print("Please run one experiment first to create the model, or modify TEST_MODEL_PATH")
    sys.exit(1)

print(f"Test model path: {TEST_MODEL_PATH}")
print()

try:
    # Create first vLLM instance
    print("[1/4] Creating first vLLM instance...")
    vllm_model1 = LLM(
        model=TEST_MODEL_PATH,
        tokenizer=MODEL_TYPE_PATH,
        dtype="bfloat16",
        swap_space=16,
    )
    print("✓ First vLLM instance created successfully")
    print()
    
    # Do a quick test generation
    print("[2/4] Testing first instance with a simple generation...")
    from vllm import SamplingParams
    sampling_params = SamplingParams(temperature=0, max_tokens=5)
    outputs1 = vllm_model1.generate(["Hello"], sampling_params)
    print(f"✓ First instance generated: {outputs1[0].outputs[0].text[:50]}")
    print()
    
    # Cleanup first instance
    print("[3/4] Cleaning up first vLLM instance...")
    del vllm_model1
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reset vLLM's distributed state (THE FIX)
    try:
        from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
        destroy_model_parallel()
        print("✓ Called destroy_model_parallel()")
    except (ImportError, AttributeError) as e:
        print(f"⚠ Warning: Could not import destroy_model_parallel: {e}")
        print("  This might be okay if vLLM version doesn't have this function")
    except Exception as e:
        print(f"⚠ Warning: destroy_model_parallel() raised exception: {e}")
    
    # Small delay
    time.sleep(3)
    print("✓ Cleanup complete, waited 3 seconds")
    print()
    
    # Create second vLLM instance (this should work if fix is correct)
    print("[4/4] Creating second vLLM instance (THIS IS THE TEST)...")
    try:
        vllm_model2 = LLM(
            model=TEST_MODEL_PATH,
            tokenizer=MODEL_TYPE_PATH,
            dtype="bfloat16",
            swap_space=16,
        )
        print("✓✓✓ SUCCESS: Second vLLM instance created without error!")
        print("  The fix appears to work!")
        print()
        
        # Test generation on second instance
        outputs2 = vllm_model2.generate(["Hi"], sampling_params)
        print(f"✓ Second instance generated: {outputs2[0].outputs[0].text[:50]}")
        print()
        
        # Cleanup
        del vllm_model2
        gc.collect()
        torch.cuda.empty_cache()
        
        print("=" * 80)
        print("✓✓✓ TEST PASSED: Fix works correctly!")
        print("=" * 80)
        
    except AssertionError as e:
        if "already initialized" in str(e):
            print("✗✗✗ TEST FAILED: Still getting 'already initialized' error")
            print(f"  Error: {e}")
            print()
            print("  The fix did NOT work. Need to investigate further.")
            sys.exit(1)
        else:
            raise
    except Exception as e:
        print(f"✗✗✗ TEST FAILED with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

except Exception as e:
    print(f"✗✗✗ TEST FAILED during setup: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

