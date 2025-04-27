# -*- coding: utf-8 -*-
############################################
#                                          #
#  Main Orchestration Script               #
#                                          #
############################################

import torch.optim as optim
import torch

# --- Import Components ---
import config                # Configuration constants and device setup
from config import GPTConfig # Import the config dataclass
from data_utils import load_and_prepare_data # Data loading/prep function
from model import SASPTransformerModel # <-- Import the new Transformer model
from trainer import Trainer      # Training loop class
from inference import generate_sequence # Inference function (needs adaptation)

def main():
    """Main function to run the workflow."""

    # --- 1. Load and Prepare Data ---
    # Data loading needs to happen before model config is finalized (for vocab_size)
    try:
        dataloader, tokenizer = load_and_prepare_data(
            dataset_name=config.DATASET_NAME,
            dataset_config=config.DATASET_CONFIG,
            tokenizer_name=config.TOKENIZER_NAME,
            max_samples=config.MAX_SAMPLES,
            # Use max_seq_length from config.py for data prep
            max_seq_length=GPTConfig.block_size, # Use default block_size initially
            batch_size=config.BATCH_SIZE
        )
        # Update vocab_size in the default config based on tokenizer
        # This is a bit awkward; better config handling might be needed for complex setups
        effective_vocab_size = tokenizer.vocab_size
        # GPT-2 tokenizer size might need adjustment for potential merges/tokens
        # Often rounded up to nearest multiple, e.g., 64
        # effective_vocab_size = (tokenizer.vocab_size + 63) // 64 * 64
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}, Effective vocab size used: {effective_vocab_size}")

    except Exception as e:
        print(f"Failed during data preparation: {e}")
        return # Exit if data prep fails

    # --- 2. Initialize Model Configuration ---
    model_config = GPTConfig(
        vocab_size=effective_vocab_size,
        block_size=tokenizer.model_max_length if tokenizer.model_max_length <= GPTConfig.block_size else GPTConfig.block_size, # Use tokenizer max length if smaller
        # Other parameters like n_layer, n_head, n_embd are taken from defaults in config.py
        # Or override them here: n_layer=12, n_head=8, n_embd=512
    )
    # Set padding_idx in config for the model
    model_config.padding_idx = tokenizer.pad_token_id

    # Print the final effective configuration
    config.print_config(model_config)

    # --- 3. Initialize Model ---
    print(f"\nInitializing Model: SASPTransformerModel")
    model = SASPTransformerModel(config=model_config)

    # --- 4. Setup Optimizer ---
    # Optimizer needs model parameters AFTER the model is initialized
    # AdamW is a good default for transformers
    # You might want to add weight decay or more sophisticated optimization later
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    print(f"Optimizer: AdamW with LR={config.LEARNING_RATE}")

    # --- 5. Initialize Trainer ---
    trainer_instance = Trainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=config.DEVICE,
        num_epochs=config.NUM_EPOCHS
    )

    # --- 6. Run Training ---
    try:
        print("\nStarting Training Phase...")
        trainer_instance.train()
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback


    # --- 7. Run Inference ---
    print("\nStarting Inference Phase...")
    # The `generate_sequence` function needs adaptation for the Transformer's `generate` method.
    # We'll create a simple prompt and use the model's `generate` method directly here.

    model.eval() # Ensure model is in eval mode
    model.to(config.DEVICE) # Ensure model is on correct device

    # Create a starting prompt (e.g., the BOS token)
    start_text = tokenizer.bos_token if tokenizer.bos_token else "<|endoftext|>" # Use BOS or a common start token
    start_ids = tokenizer.encode(start_text, return_tensors='pt').to(config.DEVICE) # Shape (1, num_start_tokens)

    print(f"\nGenerating sequence starting with: '{start_text}' (IDs: {start_ids.tolist()})")
    print(f"Max new tokens: {config.GENERATION_MAX_LEN}")

    try:
        with torch.no_grad():
            generated_ids = model.generate(
                idx=start_ids,
                max_new_tokens=config.GENERATION_MAX_LEN,
                temperature=0.8, # Add some temperature for less deterministic output
                top_k=50         # Add top-k sampling
            )

        generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)

        print("\n--- Generation Complete ---")
        print(f"Generated IDs: {generated_ids[0].tolist()}")
        print(f"\nGenerated Text:\n---\n{generated_text}\n---")

    except Exception as e:
        print(f"\nAn error occurred during inference: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback


    print("\n--- Workflow Finished ---")

if __name__ == "__main__":
    main()

