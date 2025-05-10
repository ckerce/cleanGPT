# -*- coding: utf-8 -*-
############################################
#                                          #
#  Inference / Generation Utilities        #
#                                          #
############################################

import torch
import torch.nn.functional as F # Required for generate method if using sampling
from tqdm.auto import tqdm

@torch.no_grad() # Ensure gradients are not calculated during inference
def run_generation(model, tokenizer, prompt_text, device, max_new_tokens, temperature=1.0, top_k=None):
    """
    Generates a sequence of tokens using the model's built-in generate method.

    Args:
        model: The trained SASPTransformerModel (or any model with a compatible .generate method).
        tokenizer: The tokenizer used for encoding/decoding.
        prompt_text: The starting text string for generation.
        device: The device to run inference on ('cpu', 'cuda', 'mps').
        max_new_tokens: The maximum number of new tokens to generate after the prompt.
        temperature: Sampling temperature (1.0 = no change, < 1.0 = less random, > 1.0 = more random).
        top_k: If set, restricts sampling to the top k most likely tokens.

    Returns:
        A tuple containing:
        - generated_ids (list): List of token IDs in the generated sequence (including prompt).
        - generated_text (str): The decoded text string of the generated sequence.
        Returns (None, None) if generation fails.
    """
    if not hasattr(model, 'generate'):
        print("Error: Model does not have a 'generate' method required for this inference function.")
        return None, None

    model.eval() # Set the model to evaluation mode
    model.to(device) # Ensure model is on the correct device

    print(f"\n--- Running Generation ---")
    print(f"Prompt: '{prompt_text}'")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Top-k: {top_k if top_k is not None else 'Not Used'}")

    # Encode the starting prompt
    try:
        start_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device)
        # Handle potential empty prompt after encoding (e.g., only special tokens removed)
        if start_ids.shape[1] == 0:
             print("Warning: Encoded prompt is empty. Using BOS token as fallback.")
             start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
             start_ids = torch.tensor([[start_token_id]], dtype=torch.long, device=device)

        print(f"Encoded prompt IDs: {start_ids.tolist()}")
    except Exception as e:
        print(f"Error encoding prompt: {e}")
        return None, None

    # Generate sequence using the model's method
    print("Generating...")
    try:
        # Use tqdm for progress visualization if generate takes time, though it's internal here
        # We can estimate progress based on max_new_tokens
        # progress_bar = tqdm(total=max_new_tokens, desc="Generating tokens", leave=False) # Less accurate

        generated_ids_tensor = model.generate(
            idx=start_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
        # progress_bar.close() # Close progress bar if used

        # Extract the generated IDs as a list
        generated_ids = generated_ids_tensor[0].tolist() # Get IDs from the first batch item

    except Exception as e:
        # progress_bar.close() # Ensure progress bar is closed on error
        print(f"Error during model.generate(): {e}")
        import traceback
        traceback.print_exc()
        return None, None

    # Decode the generated token IDs into text
    try:
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True) # Often skip special tokens for readability
        full_decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=False) # Keep special for debugging
    except Exception as e:
        print(f"Error decoding generated IDs: {e}")
        generated_text = "[Decoding Error]"
        full_decoded_text = "[Decoding Error]"


    print("\n--- Generation Complete ---")
    print(f"Final Token IDs: {generated_ids}")
    print(f"\nDecoded Text (skip_special_tokens=True):\n---\n{generated_text}\n---")
    # print(f"\nDecoded Text (Full):\n---\n{full_decoded_text}\n---") # Optional: show with special tokens

    return generated_ids, generated_text

# --- old function for reference or different model types ---
# def generate_sequence_step_by_step(model, tokenizer, max_len=20, device='cpu'):
#     """
#     Generates a sequence step-by-step using model.generate_next_token.
#     (Requires model to have generate_next_token and handle state if needed)
#     """
#     # ... (Implementation from previous version) ...
#     pass
