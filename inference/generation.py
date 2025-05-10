# ./inference/generation.py
"""
Text Generation Utilities
Provides functions for generating text with trained models
"""

import torch
import logging
from tqdm.auto import tqdm
from typing import Tuple, List, Dict, Optional, Union, Any

from ..tokenizers import BaseTokenizer

logger = logging.getLogger(__name__)

@torch.no_grad()
def run_generation(model: torch.nn.Module, 
                  tokenizer: BaseTokenizer,
                  prompt_text: str,
                  device: torch.device,
                  max_new_tokens: int = 50,
                  temperature: float = 1.0,
                  top_k: Optional[int] = None,
                  top_p: Optional[float] = None,
                  show_progress: bool = True) -> Tuple[List[int], str]:
    """
    Generate text using the model starting from a prompt.
    
    Args:
        model: The trained model with a generate method
        tokenizer: Tokenizer for encoding/decoding text
        prompt_text: Starting text for generation
        device: Device to run generation on
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (1.0 = no change, <1.0 = less random, >1.0 = more random)
        top_k: If set, restricts sampling to the top k most likely tokens
        top_p: If set, restricts sampling to tokens with cumulative probability >= top_p
        show_progress: Whether to show a progress bar
        
    Returns:
        Tuple of (list of token IDs, generated text string)
    """
    # Ensure the model has a generate method
    if not hasattr(model, 'generate'):
        logger.error("Model does not have a 'generate' method required for this function.")
        raise AttributeError("Model must have a 'generate' method for text generation")

    # Set the model to evaluation mode
    model.eval()
    model.to(device)

    logger.info(f"Generating text with parameters:")
    logger.info(f"  Prompt: '{prompt_text}'")
    logger.info(f"  Max new tokens: {max_new_tokens}")
    logger.info(f"  Temperature: {temperature}")
    logger.info(f"  Top-k: {top_k if top_k is not None else 'Not Used'}")
    logger.info(f"  Top-p: {top_p if top_p is not None else 'Not Used'}")

    # Encode the starting prompt
    try:
        # Add special tokens if needed
        start_ids = tokenizer.encode(prompt_text, add_special_tokens=True, return_tensors='pt')
        
        # If start_ids is not a tensor, convert it
        if not isinstance(start_ids, torch.Tensor):
            start_ids = torch.tensor([start_ids], dtype=torch.long)
            
        # Move to the correct device
        start_ids = start_ids.to(device)
        
        # Handle potential empty prompt after encoding (e.g., only special tokens removed)
        if start_ids.shape[1] == 0:
            logger.warning("Encoded prompt is empty. Using BOS token as fallback.")
            start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
            start_ids = torch.tensor([[start_token_id]], dtype=torch.long, device=device)
            
        logger.info(f"Encoded prompt IDs: {start_ids.tolist()}")
        
    except Exception as e:
        logger.error(f"Error encoding prompt: {e}")
        raise

    # Show progress bar
    if show_progress:
        progress_bar = tqdm(total=max_new_tokens, desc="Generating tokens")
    
    # Generate sequence using the model's method
    try:
        generated_ids_tensor = model.generate(
            idx=start_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        # Extract the generated IDs as a list
        if isinstance(generated_ids_tensor, torch.Tensor):
            generated_ids = generated_ids_tensor[0].tolist()  # Get IDs from the first batch item
        else:
            # Handle case where model returns something other than a tensor
            generated_ids = generated_ids_tensor
            
        if show_progress:
            progress_bar.update(max_new_tokens)  # Complete the progress bar
            progress_bar.close()
            
    except Exception as e:
        if show_progress:
            progress_bar.close()
        logger.error(f"Error during model.generate(): {e}")
        raise

    # Decode the generated token IDs into text
    try:
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        full_decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=False)  # For debugging
    except Exception as e:
        logger.error(f"Error decoding generated IDs: {e}")
        generated_text = "[Decoding Error]"
        full_decoded_text = "[Decoding Error]"

    logger.info("Generation complete")
    logger.info(f"Generated text:\n---\n{generated_text}\n---")

    return generated_ids, generated_text


def get_generation_args() -> Dict[str, Any]:
    """
    Get default arguments for generation.
    
    Returns:
        Dictionary of default generation arguments
    """
    return {
        'max_new_tokens': 50,
        'temperature': 0.8,
        'top_k': 50,
        'top_p': None,
        'show_progress': True
    }


def batch_generate(model: torch.nn.Module, 
                  tokenizer: BaseTokenizer,
                  prompts: List[str],
                  device: torch.device,
                  **kwargs) -> List[Tuple[List[int], str]]:
    """
    Generate text for multiple prompts.
    
    Args:
        model: The trained model with a generate method
        tokenizer: Tokenizer for encoding/decoding text
        prompts: List of prompt texts
        device: Device to run generation on
        **kwargs: Additional arguments for generation
        
    Returns:
        List of (token IDs, generated text) tuples
    """
    results = []
    
    for i, prompt in enumerate(prompts):
        logger.info(f"\nGenerating text for prompt {i+1}/{len(prompts)}: '{prompt}'")
        
        try:
            result = run_generation(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt,
                device=device,
                **kwargs
            )
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error generating for prompt {i+1}: {e}")
            results.append(([], f"[Generation Error: {str(e)}]"))
    
    return results
