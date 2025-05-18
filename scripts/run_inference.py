#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_inference.py - Generate text from a distilled transformer model.

This script loads a distilled model from a checkpoint and runs inference on sample text, 
showcasing the model's text completion or generation capabilities.

Usage:
    python run_inference.py --model_path "./distilled_model_output/student_model_final_distilled.pt" \
                           --model_type "Factored" \
                           --prompt "Once upon a time" \
                           --max_new_tokens 100 \
                           --temperature 0.8 \
                           --top_k 40

Example for interactive mode:
    python run_inference.py --model_path "./distilled_model_output/student_model_final_distilled.pt" \
                           --model_type "Factored" \
                           --interactive
"""

import argparse
import logging
import os
import sys
import torch
from typing import Dict, Any, Optional
from transformers import AutoTokenizer

# --- Path Setup ---
# Add parent directory to sys.path to access custom modules.
# This allows the script to import 'config_distillation' and modules from the 'model' directory
# assuming they are located in the parent directory of this script's location.
# Example: If this script is in 'project_root/scripts/', and models are in 'project_root/model/',
# this line adds 'project_root' to the Python path.
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_path, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_script_path not in sys.path: # Also add current dir for any local utils
    sys.path.insert(0, current_script_path)

# --- Custom Module Imports ---
# These imports rely on the sys.path modification above.
try:
    from model.model_token_factored_distillation import FactoredTransformerModelDistillation
    from model.model_SASPV_distillation import SASPTransformerModelDistillation
    from model.model_vanilla_distillation import VanillaTransformerModelDistillation
    from config_distillation import GPTConfig, print_config, DEVICE
except ImportError as e:
    print(f"Error importing custom modules (GPTConfig, model classes): {e}")
    print("Please ensure 'config_distillation.py' and the 'model' directory are correctly "
          "placed relative to this script (e.g., in the parent directory) and are in sys.path.")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def get_model_class(model_type: str):
    """Get the appropriate model class based on model type."""
    model_type_lower = model_type.lower()
    if model_type_lower == "factored":
        return FactoredTransformerModelDistillation
    elif model_type_lower == "sasp":
        return SASPTransformerModelDistillation
    elif model_type_lower == "vanilla":
        return VanillaTransformerModelDistillation
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported: Factored, SASP, Vanilla.")

def load_model_from_checkpoint(model_path: str, model_type: str, device: torch.device) -> Optional[Dict[str, Any]]:
    """
    Load a distilled model from checkpoint.
    Returns a dictionary containing the model and its configuration, or None on failure.
    """
    logger.info(f"Loading model from {model_path}")
    
    try:
        # Add GPTConfig to safe globals for PyTorch versions that require it for unpickling custom classes.
        # This ensures that the custom GPTConfig object can be loaded from the checkpoint.
        if 'config_distillation.GPTConfig' not in torch.serialization.get_default_safe_globals():
             torch.serialization.add_safe_globals([GPTConfig])

        # Try loading with weights_only=False first to load the pickled config object.
        # If this fails (e.g., due to code changes in GPTConfig not reflected in the checkpoint),
        # fallback to weights_only=True, which loads config as a dict.
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            logger.info("Checkpoint loaded with weights_only=False (config object included).")
        except Exception as e_false:
            logger.warning(f"Failed to load checkpoint with weights_only=False: {e_false}")
            logger.warning("Attempting to load with weights_only=True (config as dict).")
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                logger.info("Checkpoint loaded with weights_only=True.")
            except Exception as e_true:
                logger.error(f"Failed to load checkpoint with both weights_only=False and weights_only=True: {e_true}", exc_info=True)
                return None
        
        config_data = checkpoint.get('student_config')
        if config_data is None:
            logger.error("No 'student_config' found in checkpoint.")
            return None
        
        # If config_data is a dictionary (from weights_only=True), instantiate GPTConfig.
        # If it's already a GPTConfig object (from weights_only=False), use it directly.
        if isinstance(config_data, dict):
            logger.info("Instantiating GPTConfig from dictionary stored in checkpoint.")
            # Ensure all necessary fields for GPTConfig are present in the dict
            # You might need to add default values or handle missing keys if config structure changed.
            try:
                config = GPTConfig(**config_data)
            except TypeError as te:
                logger.error(f"Error creating GPTConfig from dict: {te}. Config dict: {config_data}")
                logger.error("Ensure all required arguments for GPTConfig are present in the saved config dict.")
                return None
        elif isinstance(config_data, GPTConfig):
            logger.info("Using GPTConfig object directly from checkpoint.")
            config = config_data
        else:
            logger.error(f"Unsupported type for 'student_config' in checkpoint: {type(config_data)}")
            return None
            
        # Ensure model_type is set in config, falling back to the CLI argument if not present.
        if not hasattr(config, 'model_type') or config.model_type is None:
            logger.warning(f"model_type not found or is None in loaded config. Using CLI provided model_type: '{model_type}'")
            config.model_type = model_type
        elif config.model_type.lower() != model_type.lower():
            logger.warning(f"Config model_type ('{config.model_type}') differs from CLI model_type ('{model_type}'). Using config's type.")
            # Potentially update model_type to reflect what's in the config if it's more reliable
            # model_type = config.model_type 

        # Create model instance using the determined model class and loaded/created config.
        ModelClass = get_model_class(config.model_type) # Use config.model_type as source of truth
        model = ModelClass(config)
        
        # Load model state dictionary.
        if 'model_state_dict' not in checkpoint:
            logger.error("No 'model_state_dict' found in checkpoint.")
            return None
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(device)
        model.eval()  # Set to evaluation mode.
        
        logger.info(f"Successfully loaded '{config.model_type}' model with {model.get_num_params()/1e6:.2f}M parameters.")
        if 'print_config' in globals() and callable(print_config):
            print_config(config) # Display loaded configuration details.
        
        return {"model": model, "config": config}
    
    except FileNotFoundError:
        logger.error(f"Model checkpoint file not found: {model_path}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the model: {e}", exc_info=True)
        return None

def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 50, 
                  temperature: float = 0.8, top_k: Optional[int] = 40, device: torch.device = None):
    """
    Generate text from the model given a prompt.
    """
    if device is None:
        device = next(model.parameters()).device # Infer device from model parameters if not provided.
    
    # Tokenize input prompt.
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate text using the model's generate method.
    logger.info(f"Generating text with: max_new_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k}")
    with torch.no_grad(): # Ensure no gradients are computed during inference.
        output_ids = model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k if top_k is not None and top_k > 0 else None # Pass None if top_k is 0 or not set.
        )
    
    # Decode the generated token IDs back to text.
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with a distilled transformer model.")
    
    # Model loading options
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model checkpoint (e.g., student_model_final_distilled.pt)')
    parser.add_argument('--model_type', type=str, choices=['Factored', 'SASP', 'Vanilla'], required=True,
                        help='Type of model architecture. This should match the trained model.')
    
    # Generation options
    parser.add_argument('--prompt', type=str, default="Once upon a time",
                        help='Text prompt to start generation from.')
    parser.add_argument('--max_new_tokens', type=int, default=50,
                        help='Maximum number of new tokens to generate after the prompt.')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature. Higher values (e.g., 1.0) make output more random, lower values (e.g., 0.7) make it more deterministic.')
    parser.add_argument('--top_k', type=int, default=40,
                        help='Top-k sampling. Restricts sampling to the k most likely next tokens. Set to 0 for no restriction.')
    
    # Interactive mode
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode, accepting prompts from the user until "exit" or "quit".')
    
    # Environment options
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default=None,
                        help='Device to use for inference (e.g., "cpu", "cuda"). Defaults to auto-detect via DEVICE in config_distillation.')
    
    return parser.parse_args()

def main():
    """Main function to run inference with a distilled model."""
    args = parse_args()
    
    # Setup device for inference.
    if args.device:
        current_device = torch.device(args.device)
        logger.info(f"Using device specified via CLI: {current_device}")
    else:
        current_device = DEVICE # Use default device from config_distillation.py
        logger.info(f"Using device from config_distillation.py: {current_device}")
    
    # Load tokenizer (typically the same one used for training, e.g., "gpt2").
    # This should be consistent with the vocabulary of the trained student model.
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        if tokenizer.pad_token_id is None: # Ensure pad token is set for tokenizer
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info(f"Set tokenizer pad_token to eos_token (ID: {tokenizer.pad_token_id})")
    except Exception as e:
        logger.error(f"Failed to load tokenizer 'gpt2': {e}", exc_info=True)
        return

    # Load the distilled model from the checkpoint.
    loaded_model_data = load_model_from_checkpoint(args.model_path, args.model_type, current_device)
    
    if not loaded_model_data:
        logger.error("Model loading failed. Exiting.")
        return
    
    model = loaded_model_data["model"]
    # config = loaded_model_data["config"] # Config is available if needed for other logic

    # Run inference based on mode (interactive or single prompt).
    if args.interactive:
        print("\n=== Interactive Model Inference ===")
        print("Enter your text prompts. Type 'exit' or 'quit' to end.")
        print("Type 'settings' to adjust generation parameters (max_new_tokens, temperature, top_k).")
        
        # Initialize generation parameters from CLI args, allow modification.
        gen_params = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k
        }
        
        while True:
            try:
                user_prompt = input("\nEnter prompt: ")
                if user_prompt.lower() in ["exit", "quit"]:
                    logger.info("Exiting interactive mode.")
                    break
                elif user_prompt.lower() == "settings":
                    try:
                        print(f"\nCurrent settings: Max tokens={gen_params['max_new_tokens']}, Temp={gen_params['temperature']}, Top-k={gen_params['top_k']}")
                        new_max_tokens = input(f"New max_new_tokens (press Enter to keep {gen_params['max_new_tokens']}): ")
                        if new_max_tokens: gen_params['max_new_tokens'] = int(new_max_tokens)
                        
                        new_temp = input(f"New temperature (press Enter to keep {gen_params['temperature']}): ")
                        if new_temp: gen_params['temperature'] = float(new_temp)
                        
                        new_top_k = input(f"New top_k (press Enter to keep {gen_params['top_k']}): ")
                        if new_top_k: gen_params['top_k'] = int(new_top_k)
                        print(f"Settings updated: Max tokens={gen_params['max_new_tokens']}, Temp={gen_params['temperature']}, Top-k={gen_params['top_k']}")
                    except ValueError:
                        print("Invalid input for settings. Please enter numbers.")
                    continue # Go back to prompt input
                
                if not user_prompt.strip():
                    print("Prompt cannot be empty.")
                    continue

                print("\nGenerating text...")
                generated_output = generate_text(
                    model, 
                    tokenizer, 
                    user_prompt, 
                    max_new_tokens=gen_params['max_new_tokens'],
                    temperature=gen_params['temperature'],
                    top_k=gen_params['top_k'],
                    device=current_device
                )
                print(f"\n--- Model Output ---:\n{generated_output}")
                print("--------------------")
                
            except KeyboardInterrupt:
                print("\nInteractive mode interrupted by user. Exiting.")
                break
            except Exception as e:
                logger.error(f"Error during interactive generation: {e}", exc_info=True)
                print(f"An error occurred: {e}") # Show error to user
    else:
        # Single prompt generation from CLI arguments.
        if not args.prompt.strip():
            logger.error("Prompt cannot be empty for single generation mode.")
            return

        print(f"\nPrompt: \"{args.prompt}\"")
        print("\nGenerating text...")
        generated_output = generate_text(
            model, 
            tokenizer, 
            args.prompt, 
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=current_device
        )
        print(f"\n--- Model Output ---:\n{generated_output}")
        print("--------------------")
    
    logger.info("Inference script finished.")

if __name__ == "__main__":
    main()

