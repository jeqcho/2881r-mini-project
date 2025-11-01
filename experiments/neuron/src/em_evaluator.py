#!/usr/bin/env python3
"""
Emergent Misalignment (EM) Evaluator Module

This module provides an interface for evaluating emergent misalignment
using the emergent-misalignment-eval library from:
https://github.com/jeqcho/emergent-misalignment-eval

The library will be installed separately. This module provides a stub
implementation that can be filled in when the library is available.
"""

import os
import sys
from pathlib import Path

# Try to import the emergent misalignment evaluation library
try:
    from emergent_misalignment_eval import BaseModelInterface, evaluate_model
    EM_LIBRARY_AVAILABLE = True
except ImportError:
    EM_LIBRARY_AVAILABLE = False
    print("Warning: emergent-misalignment-eval library not found.")
    print("Install from: https://github.com/jeqcho/emergent-misalignment-eval")
    print("EM evaluation will be skipped.")


class LlamaModelInterface:
    """
    Model interface for Llama-2 pruned models.
    
    This class provides methods to load a pruned model and query it,
    compatible with the emergent-misalignment-eval library interface.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
    
    def load_model(self, model_path):
        """
        Load a pruned model from checkpoint.
        
        Args:
            model_path: Path to the pruned model directory
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not EM_LIBRARY_AVAILABLE:
            print("EM library not available, skipping model load")
            return False
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model_path = Path(model_path)
            
            # Check if model checkpoint exists
            if not model_path.exists():
                print(f"Error: Model path does not exist: {model_path}")
                return False
            
            # Load tokenizer
            # For pruned models, we need to use the base model tokenizer
            # The model name should be in the parent structure
            base_model_name = "meta-llama/Llama-2-7b-chat-hf"
            
            print(f"Loading tokenizer from {base_model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            
            # Load the pruned model
            print(f"Loading pruned model from {model_path}...")
            
            # Check if there's a pytorch_model.bin file
            model_file = model_path / "pytorch_model.bin"
            if model_file.exists():
                # Load base model first
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
                # Load pruned weights
                checkpoint = torch.load(model_file, map_location="cpu")
                self.model.load_state_dict(checkpoint, strict=False)
            else:
                # Try loading directly
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            
            self.device = next(self.model.parameters()).device
            print(f"✓ Model loaded successfully on {self.device}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def query_model(self, prompt, max_length=512, temperature=0.7, **kwargs):
        """
        Query the model and return a string response.
        
        Args:
            prompt: Input prompt string
            max_length: Maximum generation length
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated response
        """
        if self.model is None or self.tokenizer is None:
            return "Error: Model not loaded"
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    **kwargs
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def cleanup(self):
        """Clean up model and free GPU memory."""
        import torch
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_em_score(model_path, n_medical=10, n_nonmedical=10, output_csv=None):
    """
    Evaluate emergent misalignment score for a pruned model.
    
    Args:
        model_path: Path to the pruned model directory
        n_medical: Number of medical questions to evaluate
        n_nonmedical: Number of non-medical questions to evaluate
        output_csv: Optional path to save detailed results
        
    Returns:
        dict: Dictionary containing EM metrics
            - em_score: Overall EM score
            - medical_score: Score on medical questions
            - nonmedical_score: Score on non-medical questions
            - detailed_results: Full evaluation results (if available)
    """
    if not EM_LIBRARY_AVAILABLE:
        print("Warning: EM library not available. Returning placeholder values.")
        return {
            "em_score": None,
            "medical_score": None,
            "nonmedical_score": None,
            "detailed_results": None,
            "error": "EM library not available"
        }
    
    try:
        # Create model interface
        interface = LlamaModelInterface()
        
        # Load the model
        if not interface.load_model(model_path):
            return {
                "em_score": None,
                "medical_score": None,
                "nonmedical_score": None,
                "error": "Failed to load model"
            }
        
        # Set output CSV path
        if output_csv is None:
            output_csv = Path(model_path) / "em_results.csv"
        
        print(f"Evaluating emergent misalignment...")
        print(f"  Medical questions: {n_medical}")
        print(f"  Non-medical questions: {n_nonmedical}")
        
        # Evaluate using the EM library
        results = evaluate_model(
            model_interface=interface,
            model_path=str(model_path),
            output_csv=str(output_csv),
            n_medical=n_medical,
            n_nonmedical=n_nonmedical
        )
        
        # Clean up
        interface.cleanup()
        
        # Extract key metrics
        em_score = results.get("em_score", results.get("overall_score", None))
        medical_score = results.get("medical_score", None)
        nonmedical_score = results.get("nonmedical_score", None)
        
        print(f"✓ EM Evaluation complete")
        print(f"  EM Score: {em_score}")
        print(f"  Medical Score: {medical_score}")
        print(f"  Non-medical Score: {nonmedical_score}")
        
        return {
            "em_score": em_score,
            "medical_score": medical_score,
            "nonmedical_score": nonmedical_score,
            "detailed_results": results
        }
        
    except Exception as e:
        print(f"Error during EM evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {
            "em_score": None,
            "medical_score": None,
            "nonmedical_score": None,
            "error": str(e)
        }


if __name__ == "__main__":
    # Test the module
    import argparse
    
    parser = argparse.ArgumentParser(description="Test EM evaluator")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to pruned model")
    parser.add_argument("--n-medical", type=int, default=10,
                       help="Number of medical questions")
    parser.add_argument("--n-nonmedical", type=int, default=10,
                       help="Number of non-medical questions")
    
    args = parser.parse_args()
    
    results = evaluate_em_score(
        model_path=args.model_path,
        n_medical=args.n_medical,
        n_nonmedical=args.n_nonmedical
    )
    
    print("\nResults:", results)

