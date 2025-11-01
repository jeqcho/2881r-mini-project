#!/usr/bin/env python3
"""Quick EM evaluation test on existing models"""

import sys
sys.path.append('/usr/local/lib/python3.12/dist-packages')

import os
import torch
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from emergent_misalignment_eval import BaseModelInterface, evaluate_model

# Load environment variables from .env file
load_dotenv()


class QuickLlamaInterface(BaseModelInterface):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
    
    def load_model(self, model_path):
        """Load pruned model"""
        try:
            print(f"Loading model from {model_path}...")
            
            # Load tokenizer from base model
            base_model_name = "meta-llama/Llama-2-7b-chat-hf"
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            
            # Load pruned model directly (supports both safetensors and pytorch_model.bin)
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            self.device = next(self.model.parameters()).device
            print(f"✓ Model loaded on {self.device}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def query_model(self, prompt, max_length=512, temperature=0.7, **kwargs):
        """Generate response"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            return f"Error: {str(e)}"


def test_model(model_path, n_medical=3, n_nonmedical=3):
    """Test EM evaluation on a model"""
    print("\n" + "="*70)
    print(f"Testing: {Path(model_path).name}")
    print("="*70)
    
    interface = QuickLlamaInterface()
    
    if not interface.load_model(model_path):
        return None
    
    print(f"Running EM evaluation ({n_medical} medical, {n_nonmedical} non-medical)...")
    
    output_csv = Path(model_path) / "em_test_results.csv"
    
    try:
        results = evaluate_model(
            model_interface=interface,
            model_path=str(model_path),
            output_csv=str(output_csv),
            n_medical=n_medical,
            n_nonmedical=n_nonmedical
        )
        
        # Clean up
        del interface.model
        torch.cuda.empty_cache()
        
        return results
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test on the newly pruned model
    models_to_test = [
        "out/test_em/p_0.01_q_0.01/model",  # Freshly pruned (1%, 1%)
    ]
    
    results_summary = []
    
    for model_path in models_to_test:
        full_path = Path("/workspace/projects/2881r-mini-project") / model_path
        
        if not full_path.exists():
            print(f"Skipping {model_path} - not found")
            continue
        
        results = test_model(str(full_path), n_medical=3, n_nonmedical=3)
        
        if results:
            model_name = Path(model_path).name
            em_score = results.get('overall_score', results.get('em_score', 'N/A'))
            results_summary.append((model_name, em_score, results))
            print(f"\n✓ {model_name}: EM Score = {em_score}")
        else:
            print(f"\n✗ {Path(model_path).name}: Failed")
    
    # Final summary
    print("\n" + "="*70)
    print("EM EVALUATION SUMMARY")
    print("="*70)
    for model_name, em_score, full_results in results_summary:
        print(f"{model_name:25s}: EM Score = {em_score}")
    print()

