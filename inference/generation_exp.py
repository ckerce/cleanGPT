# inference/generation.py
"""
Text Generation Utilities
Provides functions for generating text with trained models
"""

import torch
import logging
from tqdm.auto import tqdm
from typing import Tuple, List, Dict, Optional, Any

# Assuming BaseTokenizer is correctly located for import
# from mytokenizers import BaseTokenizer
# For the purpose of this standalone snippet, we'll assume BaseTokenizer is defined elsewhere or not strictly typed here.
class BaseTokenizer: # Placeholder if not resolving full import path
    def encode(self, text: str, add_special_tokens: bool = True, return_tensors: Optional[str] = None): pass
    def decode(self, token_ids: List[int], skip_special_tokens: bool = False): pass
    @property
    def bos_token_id(self): return 0


logger = logging.getLogger(__name__)

@torch.no_grad()
def run_generation(model: torch.nn.Module,
                  tokenizer: BaseTokenizer, # Should be your actual BaseTokenizer
                  prompt_text: str,
                  device: torch.device,
                  max_new_tokens: int = 50,
                  temperature: float = 1.0,
                  top_k: Optional[int] = None,
                  # top_p: Optional[float] = None, # REMOVED top_p from signature
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
        show_progress: Whether to show a progress bar

    Returns:
        Tuple of (list of token IDs, generated text string)
    """
    if not hasattr(model, 'generate'):
        logger.error("Model does not have a 'generate' method required for this function.")
        raise AttributeError("Model must have a 'generate' method for text generation")

    model.eval()
    model.to(device)

    logger.info(f"Generating text with parameters:")
    logger.info(f"  Prompt: '{prompt_text}'")
    logger.info(f"  Max new tokens: {max_new_tokens}")
    logger.info(f"  Temperature: {temperature}")
    logger.info(f"  Top-k: {top_k if top_k is not None else 'Not Used'}")
    # logger.info(f"  Top-p: {top_p if top_p is not None else 'Not Used'}") # REMOVED

    try:
        start_ids = tokenizer.encode(prompt_text, add_special_tokens=True, return_tensors='pt')
        if not isinstance(start_ids, torch.Tensor):
            start_ids = torch.tensor([start_ids], dtype=torch.long)
        start_ids = start_ids.to(device)
        if start_ids.shape[1] == 0:
            logger.warning("Encoded prompt is empty. Using BOS token as fallback.")
            start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
            start_ids = torch.tensor([[start_token_id]], dtype=torch.long, device=device)
        logger.info(f"Encoded prompt IDs: {start_ids.tolist()}")
    except Exception as e:
        logger.error(f"Error encoding prompt: {e}", exc_info=True)
        raise

    if show_progress:
        progress_bar = tqdm(total=max_new_tokens, desc="Generating tokens")

    try:
        # Build kwargs for model.generate selectively
        model_generate_kwargs = {
            'idx': start_ids,
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
        }
        if top_k is not None:
            model_generate_kwargs['top_k'] = top_k

        # top_p is no longer passed as it's not supported by the current models
        generated_ids_tensor = model.generate(**model_generate_kwargs)

        if isinstance(generated_ids_tensor, torch.Tensor):
            generated_ids = generated_ids_tensor[0].tolist()
        else:
            generated_ids = generated_ids_tensor # type: ignore

        if show_progress:
            progress_bar.update(max_new_tokens)
            progress_bar.close()

    except Exception as e:
        if show_progress and 'progress_bar' in locals(): # Ensure progress_bar exists
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
        # 'top_p': None, # REMOVED top_p
        'show_progress': True
    }


def batch_generate(model: torch.nn.Module,
                  tokenizer: BaseTokenizer, # Should be your actual BaseTokenizer
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
        **kwargs: Additional arguments for generation (max_new_tokens, temperature, top_k)

    Returns:
        List of (token IDs, generated text) tuples
    """
    results = []
    # Ensure top_p is not in kwargs if it was accidentally passed from an older call
    kwargs.pop('top_p', None)

    for i, prompt in enumerate(prompts):
        logger.info(f"\nGenerating text for prompt {i+1}/{len(prompts)}: '{prompt}'")
        try:
            result = run_generation(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt,
                device=device,
                **kwargs # Pass along other relevant args like max_new_tokens, temperature, top_k
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error generating for prompt {i+1}: {e}")
            results.append(([], f"[Generation Error: {str(e)}]"))
    return results

