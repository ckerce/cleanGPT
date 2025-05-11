# ./inference/generation.py
"""
Text Generation Utilities
Provides functions for generating text with trained models
"""

import torch
import logging
from tqdm.auto import tqdm
from typing import Tuple, List, Dict, Optional, Union, Any

from mytokenizers import BaseTokenizer # Assuming BaseTokenizer is correctly located

logger = logging.getLogger(__name__)

@torch.no_grad()
def run_generation(model: torch.nn.Module, 
                  tokenizer: BaseTokenizer,
                  prompt_text: str,
                  device: torch.device,
                  max_new_tokens: int = 50,
                  temperature: float = 1.0,
                  top_k: Optional[int] = None,
                  top_p: Optional[float] = None, # Keep in signature for compatibility if other parts use it
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
        top_p: If set, restricts sampling to tokens with cumulative probability >= top_p. 
               Note: This will only be passed to model.generate if the model supports it.
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
    # Log top_p status based on whether it's provided and if we intend to use it
    # For now, we will not pass top_p to the current models as they don't support it.
    logger.info(f"  Top-p: {'Not Used (model does not support)' if top_p is not None else 'Not Used'}")


    # Encode the starting prompt
    try:
        # Add special tokens if needed (though model.generate usually handles context without BOS for prompt)
        start_ids = tokenizer.encode(prompt_text, add_special_tokens=True, return_tensors='pt')
        
        if not isinstance(start_ids, torch.Tensor):
            start_ids = torch.tensor([start_ids], dtype=torch.long)
            
        start_ids = start_ids.to(device)
        
        if start_ids.shape[1] == 0:
            logger.warning("Encoded prompt is empty. Using BOS token as fallback.")
            # Ensure bos_token_id is available and is an int
            start_token_id = tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None else 0
            start_ids = torch.tensor([[start_token_id]], dtype=torch.long, device=device)
            
        logger.info(f"Encoded prompt IDs: {start_ids.tolist()}")
        
    except Exception as e:
        logger.error(f"Error encoding prompt: {e}", exc_info=True)
        raise

    if show_progress:
        progress_bar = tqdm(total=max_new_tokens, desc="Generating tokens")
    
    # Prepare arguments for model.generate, excluding top_p as current models don't use it
    generate_kwargs = {
        'idx': start_ids,
        'max_new_tokens': max_new_tokens,
        'temperature': temperature,
    }
    if top_k is not None:
        generate_kwargs['top_k'] = top_k
    
    # If your models were updated to handle top_p, you would add it here:
    # if top_p is not None:
    #     generate_kwargs['top_p'] = top_p

    try:
        generated_ids_tensor = model.generate(**generate_kwargs)
        
        if isinstance(generated_ids_tensor, torch.Tensor):
            generated_ids = generated_ids_tensor[0].tolist()  
        else:
            generated_ids = generated_ids_tensor # type: ignore
            
        if show_progress:
            progress_bar.update(max_new_tokens)  
            progress_bar.close()
            
    except Exception as e:
        if show_progress and 'progress_bar' in locals() and progress_bar: # Check if progress_bar exists and is not None
            progress_bar.close()
        logger.error(f"Error during model.generate(): {e}", exc_info=True)
        raise

    try:
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Error decoding generated IDs: {e}")
        generated_text = "[Decoding Error]"

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
        'top_p': None, # Keep for consistency, but run_generation handles its non-usage by models
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
        **kwargs: Additional arguments for generation (max_new_tokens, temperature, top_k, top_p)
        
    Returns:
        List of (token IDs, generated text) tuples
    """
    results = []
    
    # top_p will be passed to run_generation, which will then decide not to pass it to model.generate
    # if the model doesn't support it (which is the current case).
    
    for i, prompt in enumerate(prompts):
        logger.info(f"\nGenerating text for prompt {i+1}/{len(prompts)}: '{prompt}'")
        
        try:
            result = run_generation(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt,
                device=device,
                **kwargs # Pass along all kwargs (including top_p if present)
            )
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error generating for prompt {i+1}: {e}")
            results.append(([], f"[Generation Error: {str(e)}]"))
    
    return results

