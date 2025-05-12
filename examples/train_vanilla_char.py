#!/usr/bin/env python
# ./examples/train_vanilla_char.py
"""
Example script demonstrating how to train a Vanilla Transformer model with a character tokenizer.
Shows how to use the cleanGPT framework with different tokenization strategies and a standard
transformer architecture.

Example calls:

python examples/train_vanilla_char.py

python examples/train_vanilla_char.py \
  --dataset "roneneldan/TinyStories" \
  --max_samples 5000 \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 192 \
  --block_size 512 \
  --batch_size 64 \
  --num_epochs 5 \
  --output_dir "./outputs/vanilla_char_tinystories" \
  --save_tokenizer

python examples/train_vanilla_char.py \
  --dataset "wikipedia" \
  --dataset_config "20220301.simple" \
  --max_samples 5000 \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 192 \
  --block_size 512 \
  --batch_size 64 \
  --num_epochs 5 \
  --output_dir "./outputs/vanilla_char_wiki" \
  --save_tokenizer
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
    parser = argparse.ArgumentParser(description="Train Vanilla Transformer model with character tokenizer")
    
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
    parser.add_argument("--output_dir", type=str, default="./outputs/vanilla_char_model",
                        help="Directory to save outputs (model, tokenizer, logs)")
    parser.add_argument("--save_tokenizer", action="store_true",
                        help="Save the character tokenizer after training")
    
    # Device argument
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda", "mps"],
                        help="Device to use for training (default: auto-detect)")

    return parser.parse_args()

def main():
    """Main function to run the training and generation example."""
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    logger.info(f"Loading dataset: {args.dataset}")
    
    try:
        if args.dataset_config:
            dataset_for_vocab = load_dataset(
                args.dataset,
                args.dataset_config,
                split=f"train[:{args.max_samples}]", 
                trust_remote_code=True 
            )
        else:
            dataset_for_vocab = load_dataset(
                args.dataset,
                split=f"train[:{args.max_samples}]",
                trust_remote_code=True
            )
    except Exception as e:
        logger.error(f"Failed to load dataset for vocab: {e}")
        return

    text_field_for_vocab = None # Initialize to ensure it's defined
    if 'text' in dataset_for_vocab.column_names:
        text_field_for_vocab = 'text'
    elif 'story' in dataset_for_vocab.column_names: 
        text_field_for_vocab = 'story'
    else:
        text_field_for_vocab = next((col for col in dataset_for_vocab.column_names 
                           if dataset_for_vocab.features[col].dtype == 'string'), None)
    
    if not text_field_for_vocab:
        logger.error(f"Could not automatically find a text column in the dataset for vocab. Available columns: {dataset_for_vocab.column_names}")
        return
    
    logger.info(f"Using text column: '{text_field_for_vocab}' for character vocab construction.")
    text_samples_for_vocab = dataset_for_vocab[text_field_for_vocab]
    logger.info(f"Loaded {len(text_samples_for_vocab)} text samples for tokenizer construction.")
    
    logger.info("Creating character tokenizer...")
    tokenizer = create_tokenizer('character') 
    tokenizer.build_vocab_from_texts(text_samples_for_vocab)
    logger.info(f"Character tokenizer created with vocabulary size: {tokenizer.vocab_size}")
    
    logger.info("Creating model configuration for Vanilla Transformer...")
    model_config = GPTConfig(
        model_type="Vanilla", 
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        dropout=args.dropout,
        bias=args.bias, 
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    model_config.update_from_tokenizer(tokenizer)
    print_config(model_config, dataset_name=args.dataset, dataset_config=args.dataset_config, max_samples=args.max_samples)
    
    logger.info("Preparing training data...")
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available(): 
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    class CharacterLMDataCollator:
        def __init__(self, tokenizer, block_size):
            self.tokenizer = tokenizer
            self.block_size = block_size
            self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            
        def __call__(self, examples):
            input_ids_list = [ex['input_ids'] for ex in examples]
            batch_input_ids = []
            batch_labels = []
            for ids in input_ids_list:
                truncated_ids = ids[:self.block_size]
                labels = truncated_ids[1:] + [-100] 
                model_inputs = truncated_ids
                padding_length = self.block_size - len(model_inputs)
                if padding_length > 0:
                    model_inputs = model_inputs + [self.pad_token_id] * padding_length
                    labels = labels + [-100] * padding_length 
                if len(labels) < self.block_size:
                     labels = labels + [-100] * (self.block_size - len(labels))
                batch_input_ids.append(model_inputs)
                batch_labels.append(labels)
            batch = {
                "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                "labels": torch.tensor(batch_labels, dtype=torch.long)
            }
            return batch

    # The dataset for training should be the same as for vocab building in this script's logic
    # Or, if you want to load the full dataset for training after vocab building:
    # dataset_for_training = load_dataset(...) # similar to dataset_for_vocab but potentially more samples
    dataset_for_training = dataset_for_vocab # Using the same initially loaded dataset

    # The text_field_for_vocab is the one we determined earlier and is valid for this dataset object
    final_text_field_for_tokenization = text_field_for_vocab 

    def tokenize_function(examples):
        # Use the correctly determined text_field
        return tokenizer(examples[final_text_field_for_tokenization], 
                        add_special_tokens=True, 
                        truncation=True, # Truncate to block_size
                        max_length=model_config.block_size)


    logger.info(f"Tokenizing dataset using '{final_text_field_for_tokenization}' column...") # Use the determined field
    tokenized_dataset = dataset_for_training.map(
        tokenize_function,
        batched=True, 
        remove_columns=dataset_for_training.column_names, 
        desc="Tokenizing dataset"
    )
    
    data_collator = CharacterLMDataCollator(tokenizer, model_config.block_size)
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=model_config.batch_size,
        collate_fn=data_collator,
        shuffle=True 
    )
    logger.info(f"DataLoader created with batch size: {model_config.batch_size}")
    
    logger.info("Initializing Vanilla Transformer model...")
    model = get_model("Vanilla", config=model_config) 
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Vanilla Transformer model initialized with {num_params/1e6:.2f}M parameters.")

    logger.info("Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config.learning_rate,
        weight_decay=model_config.weight_decay
    )
    
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
    
    logger.info("Starting training for Vanilla Transformer...")
    trainer.train() 
    
    model_path = os.path.join(args.output_dir, "vanilla_char_model.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Vanilla Transformer model saved to {model_path}")
    
    if args.save_tokenizer:
        tokenizer_path = os.path.join(args.output_dir, "char_tokenizer")
        tokenizer.save_pretrained(tokenizer_path) 
        logger.info(f"Character tokenizer saved to {tokenizer_path}")
    
    logger.info("Generating sample text with the trained Vanilla Transformer...")
    test_prompts = [
        "Once upon a time", "The little cat", "In a land far away", "Hello world"
    ]
    model.eval() 
    for i, prompt_text in enumerate(test_prompts):
        logger.info(f"\nGenerating for prompt: '{prompt_text}'")
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        try:
            with torch.no_grad(): 
                output_ids_tensor = model.generate(
                    idx=input_ids_tensor, max_new_tokens=100, temperature=0.8, top_k=40
                )
            generated_ids = output_ids_tensor[0].tolist() 
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True) 
            logger.info(f"Generated text: {generated_text}")
            output_file = os.path.join(args.output_dir, f"vanilla_generation_{i+1}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"Prompt: {prompt_text}\n\n")
                f.write(f"Generated text:\n{generated_text}\n")
        except Exception as e:
            logger.error(f"Error generating text for prompt '{prompt_text}': {e}", exc_info=True)

    logger.info("Vanilla Transformer character-level training and generation example completed!")

if __name__ == "__main__":
    main()


