#!/usr/bin/env python
# ./examples/train_vanilla_gpt2.py
"""
Example script demonstrating how to train a Vanilla Transformer model with a GPT-2 tokenizer.
Shows how to use the cleanGPT framework with pre-trained tokenizers and a standard
transformer architecture.

Example calls:

python examples/train_vanilla_gpt2.py \
  --dataset "roneneldan/TinyStories" \
  --max_samples 5000 \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 192 \
  --block_size 256 \
  --batch_size 32 \
  --num_epochs 5 \
  --output_dir "./outputs/vanilla_gpt2_tinystories"

python examples/train_vanilla_gpt2.py \
  --dataset "wikitext" \
  --dataset_config "wikitext-2-raw-v1" \
  --max_samples 10000 \
  --n_layer 6 \
  --n_head 8 \
  --n_embd 512 \
  --block_size 512 \
  --batch_size 16 \
  --num_epochs 3 \
  --output_dir "./outputs/vanilla_gpt2_wikitext"
"""

import argparse
import logging
import torch
import os
import sys

# Add the parent directory to sys.path to access the cleanGPT modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import load_dataset
from torch.utils.data import DataLoader

# Import cleanGPT modules
from config import GPTConfig, print_config
from mytokenizers import create_tokenizer
from model import get_model
from trainers import get_trainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Vanilla Transformer model with GPT-2 tokenizer")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories",
                        help="Dataset name (e.g., 'roneneldan/TinyStories', 'wikitext')")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Dataset configuration (e.g., 'wikitext-2-raw-v1' for wikitext)")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Maximum number of samples to use from the dataset")
    
    # Model arguments
    parser.add_argument("--n_layer", type=int, default=4,
                        help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=128,
                        help="Embedding dimension")
    parser.add_argument("--block_size", type=int, default=256,
                        help="Context window size (max sequence length)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")
    parser.add_argument("--bias", action="store_true", default=False,
                        help="Use bias in Linear layers and LayerNorm")
    
    # Tokenizer arguments
    parser.add_argument("--use_fast_tokenizer", action="store_true", default=True,
                        help="Use fast Hugging Face tokenizer implementation for GPT-2")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4,
                        help="Learning rate for AdamW optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay for AdamW optimizer")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./outputs/vanilla_gpt2_model",
                        help="Directory to save outputs (model, tokenizer, logs)")
    
    # Device argument
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda", "mps"],
                        help="Device to use for training (default: auto-detect)")

    return parser.parse_args()

def main():
    """Main function to run the training and generation example."""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    ###################################################################
    # 1. Load Dataset
    ###################################################################
    logger.info(f"Loading dataset: {args.dataset}")
    
    try:
        if args.dataset_config:
            dataset = load_dataset(
                args.dataset,
                args.dataset_config,
                split=f"train[:{args.max_samples}]",
                trust_remote_code=True 
            )
        else:
            dataset = load_dataset(
                args.dataset,
                split=f"train[:{args.max_samples}]",
                trust_remote_code=True
            )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Determine the text field from the dataset
    if 'text' in dataset.column_names:
        text_field = 'text'
    elif 'story' in dataset.column_names: # For TinyStories
        text_field = 'story'
    else:
        text_field = next((col for col in dataset.column_names 
                           if dataset.features[col].dtype == 'string'), None)
        if not text_field:
            logger.error(f"Could not find a text column. Available columns: {dataset.column_names}")
            return
    logger.info(f"Using text column: '{text_field}' for tokenization.")
    logger.info(f"Loaded {len(dataset)} samples for training.")
    
    ###################################################################
    # 2. Create and Initialize GPT-2 Tokenizer
    ###################################################################
    logger.info("Creating GPT-2 tokenizer...")
    
    tokenizer = create_tokenizer(
        'gpt2', 
        use_fast=args.use_fast_tokenizer
    )
    # Ensure pad token is set for GPT-2, typically to EOS token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"GPT-2 tokenizer pad_token set to eos_token ({tokenizer.eos_token}).")
    logger.info(f"GPT-2 tokenizer initialized with vocabulary size: {tokenizer.vocab_size}")
    
    ###################################################################
    # 3. Create Model Configuration
    ###################################################################
    logger.info("Creating model configuration for Vanilla Transformer...")
    
    model_config = GPTConfig(
        model_type="Vanilla",
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size, # Max sequence length for model
        dropout=args.dropout,
        bias=args.bias,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    model_config.update_from_tokenizer(tokenizer) # Sets vocab_size, padding_idx
    
    print_config(model_config, dataset_name=args.dataset, dataset_config=args.dataset_config, max_samples=args.max_samples)
    
    ###################################################################
    # 4. Prepare Training Data (Tokenize and Create DataLoader)
    ###################################################################
    logger.info("Preparing training data...")
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data Collator for GPT-2 based Language Modeling
    class VanillaGPT2DataCollator:
        def __init__(self, tokenizer, block_size):
            self.tokenizer = tokenizer
            self.block_size = block_size 
            self.pad_token_id = tokenizer.pad_token_id
            if self.pad_token_id is None:
                raise ValueError("Tokenizer must have a pad_token_id set for the collator.")

        def __call__(self, examples: list[dict]):
            batch_input_ids = []
            batch_labels = []

            for ex in examples:
                ids = ex['input_ids'] # Already truncated by dataset.map's tokenize_function

                # Pad current sequence to self.block_size
                padding_length = self.block_size - len(ids)
                # Ensure truncation if somehow an example is longer (should not happen with map)
                if padding_length < 0: 
                    ids = ids[:self.block_size]
                    padding_length = 0
                
                padded_ids = ids + [self.pad_token_id] * padding_length
                
                # Labels are shifted versions of padded_ids. -100 is ignored by CrossEntropyLoss.
                labels = padded_ids[1:] + [-100]

                batch_input_ids.append(padded_ids)
                batch_labels.append(labels)
            
            return {
                "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                "labels": torch.tensor(batch_labels, dtype=torch.long),
            }

    # Tokenize the dataset
    def tokenize_function(examples):
        # The tokenizer.__call__ method handles batching if examples[text_field] is a list of strings.
        outputs = tokenizer(
            examples[text_field],
            add_special_tokens=True, # Add BOS/EOS tokens
            truncation=True,         # Truncate sequences longer than block_size
            max_length=model_config.block_size,
            padding=False,           # Collator will handle padding per batch
        )
        return outputs

    logger.info(f"Tokenizing dataset using '{text_field}' column with max_length={model_config.block_size}...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    data_collator = VanillaGPT2DataCollator(tokenizer, model_config.block_size)
    
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=model_config.batch_size,
        collate_fn=data_collator,
        shuffle=True
    )
    
    logger.info(f"DataLoader created with batch size: {model_config.batch_size}")
    
    ###################################################################
    # 5. Initialize Vanilla Transformer Model
    ###################################################################
    logger.info("Initializing Vanilla Transformer model...")
    
    model = get_model("Vanilla", config=model_config)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Vanilla Transformer model initialized with {num_params/1e6:.2f}M parameters.")

    ###################################################################
    # 6. Setup Optimizer
    ###################################################################
    logger.info("Setting up optimizer...")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config.learning_rate,
        weight_decay=model_config.weight_decay
    )
    
    ###################################################################
    # 7. Initialize Trainer
    ###################################################################
    logger.info("Initializing simple trainer...")
    
    trainer = get_trainer(
        trainer_type="simple",
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        num_epochs=model_config.num_epochs,
        output_dir=args.output_dir,
    )
    
    ###################################################################
    # 8. Train the Model
    ###################################################################
    logger.info("Starting training for Vanilla Transformer with GPT-2 tokenizer...")
    
    trainer.train()
    
    model_path = os.path.join(args.output_dir, "vanilla_gpt2_model.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Vanilla Transformer model saved to {model_path}")
    
    # GPT-2 tokenizer is pre-trained, usually not saved unless fine-tuned/modified.
    # If you were to train a BPE tokenizer from scratch and wanted to save it:
    # tokenizer_path = os.path.join(args.output_dir, "gpt2_tokenizer")
    # tokenizer.save_pretrained(tokenizer_path)
    # logger.info(f"GPT-2 tokenizer saved to {tokenizer_path}")
    
    ###################################################################
    # 9. Generate Sample Text
    ###################################################################
    logger.info("Generating sample text with the trained Vanilla Transformer (GPT-2)...")
    
    test_prompts = [
        "Once upon a time, in a land far away",
        "The quick brown fox",
        "Artificial intelligence is"
    ]
    
    model.eval()
    
    for i, prompt_text in enumerate(test_prompts):
        logger.info(f"\nGenerating for prompt: '{prompt_text}'")
        
        # Encode the prompt using the GPT-2 tokenizer
        # add_special_tokens=True can be useful if the model was trained with them (e.g. BOS)
        # For generation, often starting raw is fine.
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=True, return_tensors="pt")
        input_ids_tensor = input_ids.to(device)
        
        try:
            with torch.no_grad():
                output_ids_tensor = model.generate(
                    idx=input_ids_tensor,
                    max_new_tokens=100, 
                    temperature=0.7,   
                    top_k=50           
                )
            
            generated_ids = output_ids_tensor[0].tolist()
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True) 
            
            logger.info(f"Generated text: {generated_text}")
            
            output_file = os.path.join(args.output_dir, f"vanilla_gpt2_generation_{i+1}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"Prompt: {prompt_text}\n\n")
                f.write(f"Generated text:\n{generated_text}\n")
                
        except Exception as e:
            logger.error(f"Error generating text for prompt '{prompt_text}': {e}", exc_info=True)

    logger.info("Vanilla Transformer GPT-2 training and generation example completed!")

if __name__ == "__main__":
    main()

