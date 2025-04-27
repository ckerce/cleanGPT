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
    # Import the updated inference function
    from inference import run_generation # <-- Import the new function

    # Define the starting prompt
    start_text = tokenizer.bos_token if tokenizer.bos_token else "<|endoftext|>"

    try:
        # Call the dedicated inference function
        generated_ids, generated_text = run_generation(
            model=model,
            tokenizer=tokenizer,
            prompt_text=start_text,
            device=config.DEVICE,
            max_new_tokens=config.GENERATION_MAX_LEN,
            temperature=0.8, # Example temperature
            top_k=50         # Example top-k
        )
        # Optional: Do something with the results if needed
        # if generated_text:
        #     print("Inference successful.")

    except Exception as e:
        print(f"\nAn error occurred during inference execution: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Workflow Finished ---")

if __name__ == "__main__":
    main()
