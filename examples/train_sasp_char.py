#!/usr/bin/env python
# ./examples/train_sasp_char.py
"""
Example script demonstrating how to train a SASP model with character tokenizer
Shows how to use the cleanGPT framework with different tokenization strategies

Example calls:

python examples/train_sasp_char.py

python examples/train_sasp_char.py \
  --dataset "roneneldan/TinyStories" \
  --max_samples 5000 \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 192 \
  --block_size 512 \
  --batch_size 64 \
  --num_epochs 5 \
  --output_dir "./outputs/sasp_char_large" \
  --save_tokenizer

python examples/train_sasp_char.py \
  --dataset "wikipedia" \
  --dataset_config "20220301.simple" \
  --max_samples 5000 \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 192 \
  --block_size 512 \
  --batch_size 64 \
  --num_epochs 5 \
  --output_dir "./outputs/sasp_char_wiki" \
  --save_tokenizer



"""

import argparse
import logging
import torch
import torch.nn.functional as F
from datasets import load_dataset
import os
import sys
import numpy as np

# Add the parent directory to sys.path to access the cleanGPT modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import cleanGPT modules
from config import GPTConfig, print_config
from mytokenizers import create_tokenizer
from model import get_model
from utils.data_utils import TokenizedDataset
from trainers import get_trainer
from inference.generation import run_generation

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SASP model with character tokenizer")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories",
                        help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Dataset configuration")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Maximum number of samples to use")
    
    # Model arguments
    parser.add_argument("--n_layer", type=int, default=4,
                        help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=128,
                        help="Embedding dimension")
    parser.add_argument("--block_size", type=int, default=256,
                        help="Context window size")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help="Learning rate")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="../outputs/char_sasp",
                        help="Directory to save outputs")
    parser.add_argument("--save_tokenizer", action="store_true",
                        help="Save the character tokenizer after training")
    
    return parser.parse_args()

def main():
    """Main function to run the example."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    ###################################################################
    # 1. Load Dataset (First step to build character vocabulary)
    ###################################################################
    logger.info(f"Loading dataset: {args.dataset}")
    
    # Load a small subset first to build the character tokenizer
    if args.dataset_config:
        dataset = load_dataset(
            args.dataset,
            args.dataset_config,
            split=f"train[:{args.max_samples}]",
        )
    else:
        dataset = load_dataset(
            args.dataset,
            split=f"train[:{args.max_samples}]",
        )
    
    # Extract text field based on dataset structure
    if 'text' in dataset.column_names:
        text_samples = dataset['text']
    elif 'story' in dataset.column_names:
        text_samples = dataset['story']
    else:
        # Find the first string column as fallback
        text_field = next((col for col in dataset.column_names 
                        if dataset.features[col].dtype == 'string'), None)
        if not text_field:
            raise ValueError(f"Could not find text column. Available columns: {dataset.column_names}")
        text_samples = dataset[text_field]
    
    logger.info(f"Loaded {len(text_samples)} text samples")
    
    ###################################################################
    # 2. Create and initialize character tokenizer
    ###################################################################
    logger.info("Creating character tokenizer...")
    
    # Create a character tokenizer
    tokenizer = create_tokenizer('character')
    
    # Build vocabulary from the loaded texts
    tokenizer.build_vocab_from_texts(text_samples)
    logger.info(f"Character tokenizer created with vocabulary size: {tokenizer.vocab_size}")
    
    ###################################################################
    # 3. Create Model Configuration
    ###################################################################
    logger.info("Creating model configuration...")
    
    # Create a configuration object
    model_config = GPTConfig(
        model_type="SASP",  # Use SASP architecture
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        dropout=args.dropout,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        # SASP-specific configs
        use_proj=True,       # Use projection in attention
        use_v=True,          # Use Value vectors
        transformer_block_type='SASP'
    )
    
    # Update the vocabulary size from our character tokenizer
    model_config.update_from_tokenizer(tokenizer)
    
    # Print the configuration
    #print_config(model_config)
    # Print the configuration with correct dataset info
    print_config(model_config, dataset_name=args.dataset, dataset_config=args.dataset_config, max_samples=args.max_samples)
    
    ###################################################################
    # 4. Load and Prepare Training Data
    ###################################################################
    logger.info("Preparing training data...")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                        "mps" if torch.backends.mps.is_available() else 
                        "cpu")
    logger.info(f"Using device: {device}")
    
    # Define a custom data collator for character-level training
    class CharacterLMDataCollator:
        def __init__(self, tokenizer, block_size):
            self.tokenizer = tokenizer
            self.block_size = block_size
            self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            
        def __call__(self, examples):
            # Extract input_ids from examples
            if isinstance(examples[0], dict):
                input_ids = [example["input_ids"] for example in examples]
            else:
                input_ids = examples
            
            # Handle variable length sequences
            max_length = max(len(ids) for ids in input_ids)
            max_length = min(max_length, self.block_size)
            
            # Prepare batch
            batch_input_ids = []
            batch_labels = []
            
            for ids in input_ids:
                # Truncate or pad as needed
                ids = ids[:max_length]
                padding_length = max_length - len(ids)
                
                if padding_length > 0:
                    ids = ids + [self.pad_token_id] * padding_length
                
                # For causal LM, labels are input_ids shifted right
                labels = ids[1:] + [-100]  # -100 is ignored in loss calculation
                
                batch_input_ids.append(ids)
                batch_labels.append(labels)
            
            # Convert to tensors
            batch = {
                "input_ids": torch.tensor(batch_input_ids),
                "labels": torch.tensor(batch_labels)
            }
            
            return batch
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text" if "text" in examples else "story"], 
                        add_special_tokens=True, 
                        truncation=True, 
                        max_length=model_config.block_size)
    
    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    # Create DataLoader
    data_collator = CharacterLMDataCollator(tokenizer, model_config.block_size)
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=model_config.batch_size,
        collate_fn=data_collator,
        shuffle=True
    )
    
    logger.info(f"DataLoader created with batch size: {model_config.batch_size}")
    
    ###################################################################
    # 5. Initialize Model
    ###################################################################
    logger.info("Initializing SASP model...")
    
    # Get the SASP model
    model = get_model("SASP", config=model_config)
    model = model.to(device)
    
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
    logger.info("Initializing trainer...")
    
    trainer = get_trainer(
        trainer_type="simple",
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        num_epochs=model_config.num_epochs,
        output_dir=args.output_dir
    )
    
    ###################################################################
    # 8. Train the Model
    ###################################################################
    logger.info("Starting training...")
    
    # Train the model
    trainer.train()
    
    # Save the model
    model_path = os.path.join(args.output_dir, "sasp_char_model.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save the tokenizer if requested
    if args.save_tokenizer:
        tokenizer_path = os.path.join(args.output_dir, "char_tokenizer")
        tokenizer.save_pretrained(tokenizer_path)
        logger.info(f"Character tokenizer saved to {tokenizer_path}")
    
    ###################################################################
    # 9. Generate Sample Text
    ###################################################################
    logger.info("Generating sample text...")
    
    # Test prompts using character-level generation
    test_prompts = [
        "Once upon a time",
        "The little",
        "In a world"
    ]
    
    # Generate text for each prompt
    for i, prompt in enumerate(test_prompts):
        logger.info(f"\nGenerating text for prompt: '{prompt}'")
        
        try:
            # Directly handle text generation without relying on run_generation
            # Encode the prompt
            input_ids = tokenizer.encode(prompt, add_special_tokens=True)
            
            # Convert to tensor and add batch dimension
            if isinstance(input_ids, list):
                input_ids = torch.tensor([input_ids], dtype=torch.long)
            
            # Move to device
            input_ids = input_ids.to(device)
            
            # Generate text
            model.eval()
            with torch.no_grad():
                output_ids = model.generate(
                    idx=input_ids,
                    max_new_tokens=100,
                    temperature=0.8,
                    top_k=40
                )
            
            # Decode the generated text
            generated_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
            
            # Log the generated text
            logger.info(f"Generated text: {generated_text}")
            
            # Save to a file
            output_file = os.path.join(args.output_dir, f"generation_{i+1}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"Prompt: {prompt}\n\n")
                f.write(f"Generated text: {generated_text}\n")
                
        except Exception as e:
            logger.error(f"Error generating text for prompt '{prompt}': {e}")
    
    logger.info("Example completed successfully!")

if __name__ == "__main__":
    main()
