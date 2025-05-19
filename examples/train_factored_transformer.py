#!/usr/bin/env python
# ./trainers/train_factored_transformer.py
"""
Executable script for training a FactoredTransformerModel.
This script sets up the configuration, data, model, and trainer, then runs the training.

python ./trainers/train_factored_transformer.py \
  --dataset "roneneldan/TinyStories" \
  --max_samples 1000 \
  --n_layer 4 \
  --n_head 4 \
  --n_embd 256 \
  --block_size 128 \
  --batch_size 16 \
  --num_epochs 1 \
  --output_dir "./outputs/my_factored_model_test" \
  --tokenizer_type gpt2 \
  --device cuda  # or cpu/mps


python ./trainers/train_factored_transformer.py \
  --dataset "wikitext" \
  --dataset_config "wikitext-2-raw-v1" \
  --n_layer 4 \
  --n_head 4 \
  --n_embd 256 \
  --block_size 128 \
  --batch_size 16 \
  --num_epochs 10  # You'll likely want more epochs for WikiText-2 \
  --output_dir "./outputs/wikitext2_factored_model" \
  --tokenizer_type gpt2 \
  --device cuda
  --max_samples 103000000 

"""

import argparse
import logging
import torch
import os
import sys

# Add the parent directory to sys.path to access the cleanGPT modules
# Assuming this script is in trainers/, and the root is one level up.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import GPTConfig, print_config, create_config_from_args, DEVICE, CURRENT_TIME
from mytokenizers import create_tokenizer
from model import get_model
from utils.data_utils import load_and_prepare_data # Using the existing data loader
from trainers import get_trainer # Will use "simple" trainer
from inference.generation import run_generation # For sample generation
from datasets import load_dataset # Added for character tokenizer vocab building

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Factored Transformer Model")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories",
                        help="Dataset name (e.g., 'roneneldan/TinyStories', 'wikitext')")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Dataset configuration (e.g., 'wikitext-2-raw-v1')")
    parser.add_argument("--max_samples", type=int, default=100000,
                        help="Maximum number of samples to use from the dataset")

    # Tokenizer arguments
    parser.add_argument("--tokenizer_type", type=str, default="gpt2", choices=["character", "gpt2"],
                        help="Type of tokenizer to use.")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to a pretrained tokenizer directory (optional).")

    # Model arguments (FactoredTransformerModel specific + general)
    # model_type will be hardcoded to "Factored" in the script but can be an arg for flexibility
    parser.add_argument("--model_type", type=str, default="Factored", choices=["Factored", "SASP", "Vanilla"],
                        help="Model architecture type.")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of transformer layers.")
    parser.add_argument("--n_head", type=int, default=6, help="Number of attention heads.")
    parser.add_argument("--n_embd", type=int, default=288, help="Embedding dimension.")
    parser.add_argument("--block_size", type=int, default=512, help="Context window size.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    parser.add_argument("--bias", action='store_true', default=True, # Factored model uses bias by default in LayerNorm
                        help="Use bias in Linear layers and LayerNorm.")
    parser.add_argument("--no_bias", action='store_false', dest='bias',
                        help="Do not use bias.")


    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--num_epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for AdamW optimizer.")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Max norm for gradient clipping (None for no clipping).")
    parser.add_argument("--log_interval", type=int, default=50, help="Log frequency (every N batches).")
    parser.add_argument("--trainer_type", type=str, default="simple",
                        help="Type of trainer to use (default: simple).")


    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./outputs/factored_transformer_model",
                        help="Directory to save outputs (model, tokenizer, logs).")
    parser.add_argument("--save_model_filename", type=str, default="factored_model.pt",
                        help="Filename for the saved model checkpoint.")
    parser.add_argument("--save_tokenizer_dirname", type=str, default="factored_tokenizer",
                        help="Directory name (within output_dir) to save tokenizer.")

    # Device argument
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"],
                        help="Device to use for training (default: auto-detect from config).")

    # Inference arguments
    parser.add_argument("--skip_generation", action="store_true", help="Skip sample text generation after training.")
    parser.add_argument("--generation_max_len", type=int, default=50, help="Max new tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling parameter")


    return parser.parse_args()

def main():
    """Main function to run the training and generation for FactoredTransformerModel."""
    args = parse_args()

    # --- Setup Output Directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # --- Create/Load Configuration ---
    logger.info("Creating model configuration...")
    # Use the create_config_from_args to populate GPTConfig from command line
    model_config = create_config_from_args(args)
    # Override model_type to ensure we are training a Factored model
    model_config.model_type = "Factored"


    # --- Determine Device (uses logic from config.py via create_config_from_args) ---
    # The global DEVICE in config.py will be updated by create_config_from_args if args.device is set.
    # We will use this global DEVICE.
    current_device = DEVICE
    logger.info(f"Using device: {current_device} (from config: {DEVICE})")


    # --- Initialize Tokenizer ---
    logger.info(f"Initializing {args.tokenizer_type} tokenizer...")
    tokenizer_params = {} # General parameters for tokenizer creation

    if args.tokenizer_path:
        tokenizer = create_tokenizer(args.tokenizer_type, from_pretrained=args.tokenizer_path, **tokenizer_params)
        logger.info(f"Loaded {args.tokenizer_type} tokenizer from: {args.tokenizer_path}")
    else:
        if args.tokenizer_type == "gpt2":
            tokenizer_params['use_fast'] = True # Default for GPT2Tokenizer
            tokenizer = create_tokenizer(args.tokenizer_type, **tokenizer_params)
            logger.info(f"Created new {args.tokenizer_type} tokenizer with default settings.")
        elif args.tokenizer_type == "character":
            logger.info("Building character tokenizer vocab from dataset as no tokenizer_path was provided.")
            # Temporarily load data just for vocab building for char tokenizer
            temp_split_str = f"train[:{min(args.max_samples, 10000)}]" # Use a subset for vocab building
            temp_dataset_args = [args.dataset]
            if args.dataset_config:
                temp_dataset_args.append(args.dataset_config)
            
            try:
                temp_dataset = load_dataset(*temp_dataset_args, split=temp_split_str, trust_remote_code=True)
            except Exception as e:
                logger.error(f"Failed to load dataset for character vocab building: {e}", exc_info=True)
                sys.exit(1)

            if 'text' in temp_dataset.column_names:
                text_samples_for_vocab = temp_dataset['text']
            elif 'story' in temp_dataset.column_names: # For TinyStories
                text_samples_for_vocab = temp_dataset['story']
            else:
                # Fallback: try to find the first string column
                text_field_for_vocab = next((col for col in temp_dataset.column_names 
                                           if temp_dataset.features[col].dtype == 'string'), None)
                if not text_field_for_vocab:
                    logger.error(f"Could not automatically find a text column in the dataset for vocab building. Available columns: {temp_dataset.column_names}")
                    sys.exit(1)
                logger.info(f"Using text column: '{text_field_for_vocab}' for character vocab.")
                text_samples_for_vocab = temp_dataset[text_field_for_vocab]

            # Ensure text_samples_for_vocab is a list of strings
            if not isinstance(text_samples_for_vocab, list) or (text_samples_for_vocab and not isinstance(text_samples_for_vocab[0], str)):
                 text_samples_for_vocab = [str(item) for item in text_samples_for_vocab]


            char_tokenizer_instance = create_tokenizer(args.tokenizer_type, **tokenizer_params)
            char_tokenizer_instance.build_vocab_from_texts(text_samples_for_vocab)
            tokenizer = char_tokenizer_instance
            logger.info(f"Character tokenizer vocabulary built. Vocab size: {tokenizer.vocab_size}")
        else:
            # Fallback for any other tokenizer type not explicitly handled above
            tokenizer = create_tokenizer(args.tokenizer_type, **tokenizer_params)
            logger.info(f"Created new {args.tokenizer_type} tokenizer with default settings.")

    model_config.update_from_tokenizer(tokenizer) # Update vocab_size, padding_idx
    print_config(model_config, dataset_name=args.dataset, dataset_config=args.dataset_config, max_samples=args.max_samples)


    # --- Load and Prepare Data ---
    logger.info("Loading and preparing data...")
    # load_and_prepare_data returns (dataloader, tokenizer)
    # tokenizer might be updated (e.g. with padding token) by data collator setup
    dataloader, tokenizer = load_and_prepare_data(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        max_seq_length=model_config.block_size,
        batch_size=model_config.batch_size,
        mlm=False, # Causal LM for FactoredTransformerModel typically
        split='train',
        shuffle=True
    )
    logger.info(f"Data loaded. DataLoader has {len(dataloader)} batches.")


    # --- Initialize Model ---
    logger.info(f"Initializing {model_config.model_type} model...")
    model = get_model(model_config.model_type, config=model_config)
    model = model.to(current_device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{model_config.model_type} model initialized with {num_params/1e6:.2f}M parameters.")


    # --- Setup Optimizer ---
    logger.info("Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config.learning_rate, # Use LR from config, which was set by args
        weight_decay=model_config.weight_decay # Use weight_decay from config
    )

    # --- Initialize Trainer ---
    logger.info(f"Initializing {args.trainer_type} trainer...")
    # Pass arguments that SimpleTrainer expects
    trainer_kwargs = {
        'num_epochs': model_config.num_epochs, # Use num_epochs from config
        'output_dir': args.output_dir,
        'clip_grad_norm': args.clip_grad_norm,
        'log_interval': args.log_interval
    }
    trainer = get_trainer(
        trainer_type=args.trainer_type, # e.g., "simple"
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=current_device,
        **trainer_kwargs
    )

    # --- Train the Model ---
    logger.info(f"Starting training for {model_config.model_type}...")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        sys.exit(1) # Exit if training fails

    # --- Save Model and Tokenizer ---
    model_path = os.path.join(args.output_dir, args.save_model_filename)
    torch.save(model.state_dict(), model_path)
    logger.info(f"{model_config.model_type} model saved to {model_path}")

    # Save tokenizer if it's character (might have been built) or if a path was given (implying it might be custom/modified)
    # or if it's a gpt2 tokenizer that we want to save consistently.
    # For simplicity, let's always try to save if it's not a from_pretrained gpt2 without a specific path.
    if args.tokenizer_type == "character" or args.tokenizer_path is not None or (args.tokenizer_type == "gpt2" and args.tokenizer_path is None):
        tokenizer_save_path = os.path.join(args.output_dir, args.save_tokenizer_dirname)
        try:
            tokenizer.save_pretrained(tokenizer_save_path)
            logger.info(f"Tokenizer saved to {tokenizer_save_path}")
        except Exception as e:
            logger.error(f"Could not save tokenizer: {e}", exc_info=True)


    # --- Generate Sample Text ---
    if not args.skip_generation:
        logger.info("Generating sample text with the trained model...")
        test_prompts = [
            "The old house stood on a hill overlooking the",
            "Once upon a time, there was a brave knight who",
            "The recipe for disaster is simple:",
            "In the year 2077, cybernetics"
        ]
        model.eval() # Ensure model is in evaluation mode

        for i, prompt_text in enumerate(test_prompts):
            logger.info(f"\nGenerating for prompt: '{prompt_text}'")
            try:
                _, generated_text = run_generation(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    device=current_device,
                    max_new_tokens=args.generation_max_len, # Use args for generation params
                    temperature=args.temperature,
                    top_k=args.top_k,
                    show_progress=False
                )
                logger.info(f"Generated text: {generated_text}")
                output_file = os.path.join(args.output_dir, f"factored_generation_{i+1}.txt")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(f"Run Time: {CURRENT_TIME}\n")
                    f.write(f"Model: {model_config.model_type}\n")
                    f.write(f"Prompt: {prompt_text}\n\n")
                    f.write(f"Generated text:\n{generated_text}\n")
            except Exception as e:
                logger.error(f"Error generating text for prompt '{prompt_text}': {e}", exc_info=True)

    logger.info(f"{model_config.model_type} training and generation example completed!")

if __name__ == "__main__":
    main()

