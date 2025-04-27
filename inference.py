# -*- coding: utf-8 -*-
############################################
#                                          #
#  Inference / Generation Utilities        #
#                                          #
############################################

import torch
from tqdm.auto import tqdm

def generate_sequence(model, tokenizer, max_len=20, device='cpu'):
    """
    Generates a sequence of tokens using the model's generation logic.
    Relies on model having a 'generate_next_token' method for simplicity here,
    or a more complex 'generate' method could be implemented.
    """
    if not hasattr(model, 'generate_next_token'):
        print("Error: Model does not have a 'generate_next_token' method required for this simple inference function.")
        return None, None # Return None for IDs and text

    model.eval() # Set the model to evaluation mode
    model.to(device) # Ensure model is on the correct device

    print(f"\n--- Generating Sequence (Max Length: {max_len}) ---")
    # Print model-specific info if available
    if hasattr(model, 'learnable_param'):
        print(f"Model parameter during generation: {model.learnable_param.item():.6f}")

    start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    if start_token_id is None:
        print("Warning: No BOS or EOS token found, starting with token ID 0.")
        start_token_id = 0

    generated_ids = [start_token_id]
    print(f"Start token ID: {start_token_id} ('{tokenizer.decode([start_token_id])}')")

    with torch.no_grad(): # Disable gradient calculations
        # Use tqdm for a generation progress bar
        progress_bar = tqdm(range(max_len - 1), desc="Generating tokens", leave=False)
        for _ in progress_bar:
            # Prepare input for the model's generate_next_token method
            # In a real scenario, this might involve hidden states etc.
            # For OOBC, the input doesn't matter.
            current_sequence_tensor = torch.tensor([generated_ids]).to(device) # Batch dim = 1

            # Call the model's method to get the next token
            next_token_id = model.generate_next_token(x_in=current_sequence_tensor)
            generated_ids.append(next_token_id)

            # Update progress bar description (optional)
            # progress_bar.set_postfix({"last_token": next_token_id})

            if next_token_id == tokenizer.eos_token_id:
                print(f"  EOS token (ID: {tokenizer.eos_token_id}) generated. Stopping.")
                break

    # Decode the final sequence
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    print("\n--- Generation Complete ---")
    print(f"Final Token IDs: {generated_ids}")
    print(f"Decoded Text: '{generated_text}'")
    # Note about OOBC's expected output
    if isinstance(model, torch.nn.Module) and type(model).__name__ == 'OOBC':
         print("(Note: Expect repetitive generation from the simple OOBC model.)")


    return generated_ids, generated_text
