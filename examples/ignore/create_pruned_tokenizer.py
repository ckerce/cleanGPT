#!/usr/bin/env python
# ./examples/create_pruned_tokenizer.py
"""
Example script for creating a pruned GPT-2 tokenizer
This reduces the vocabulary size based on a specific dataset
"""

import argparse
import logging
from datasets import load_dataset
import os
import sys

# Add the parent directory to sys.path to access the cleanGPT modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary modules
from mytokenizers import create_tokenizer
from mytokenizers.pruned_gpt2_tokenizer import PrunedGPT2Tokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create a pruned GPT-2 tokenizer")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories",
                        help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Dataset configuration")
    parser.add_argument("--max_samples", type=int, default=10000,
                        help="Maximum number of samples to analyze")
    
    # Tokenizer arguments
    parser.add_argument("--coverage", type=float, default=0.95,
                        help="Target token coverage (0.0-1.0)")
    parser.add_argument("--max_vocab_size", type=int, default=None,
                        help="Maximum vocabulary size")
    parser.add_argument("--use_fast_tokenizer", action="store_true", default=True,
                        help="Use fast tokenizer implementation")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./outputs/pruned_tokenizer",
                        help="Directory to save the pruned tokenizer")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
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
    
    # Create standard GPT-2 tokenizer
    logger.info("Creating base GPT-2 tokenizer...")
    base_tokenizer = create_tokenizer('gpt2', use_fast=args.use_fast_tokenizer)
    
    # Ensure the tokenizer has appropriate padding settings
    if base_tokenizer.pad_token is None:
        # Set EOS token as padding token if none exists
        base_tokenizer.pad_token = base_tokenizer.eos_token
    
    # Create pruned tokenizer
    logger.info(f"Creating pruned tokenizer with target coverage: {args.coverage}")
    pruned_tokenizer = PrunedGPT2Tokenizer.create_pruned_tokenizer(
        base_tokenizer=base_tokenizer,
        texts=text_samples,
        coverage=args.coverage,
        max_vocab_size=args.max_vocab_size
    )
    
    # Save the pruned tokenizer
    logger.info(f"Saving pruned tokenizer to {args.output_dir}")
    pruned_tokenizer.save_pretrained(args.output_dir)
    
    # Display vocabulary size reduction
    original_size = base_tokenizer.vocab_size
    pruned_size = pruned_tokenizer.vocab_size
    reduction_percent = (1 - pruned_size / original_size) * 100
    
    logger.info(f"Vocabulary reduction summary:")
    logger.info(f"  Original size: {original_size} tokens")
    logger.info(f"  Pruned size:   {pruned_size} tokens")
    logger.info(f"  Reduction:     {reduction_percent:.1f}%")
    
    # Test the pruned tokenizer
    logger.info("Testing pruned tokenizer...")
    test_text = "The quick brown fox jumps over the lazy dog."
    
    # Encode with both tokenizers
    base_encoded = base_tokenizer.encode(test_text)
    pruned_encoded = pruned_tokenizer.encode(test_text)
    
    logger.info(f"Test text: '{test_text}'")
    logger.info(f"Base encoding length: {len(base_encoded)}")
    logger.info(f"Pruned encoding length: {len(pruned_encoded)}")
    
    # Decode back
    base_decoded = base_tokenizer.decode(base_encoded)
    pruned_decoded = pruned_tokenizer.decode(pruned_encoded)
    
    logger.info(f"Base decoding: '{base_decoded}'")
    logger.info(f"Pruned decoding: '{pruned_decoded}'")
    
    logger.info("Pruned tokenizer created successfully!")

if __name__ == "__main__":
    main()
