# ./main.py
"""
Main Orchestration Script for cleanGPT
Enhanced with modular tokenization and architecture support
"""

import os
import argparse
import torch
import torch.optim as optim
import logging
from typing import List, Optional

# Import Configuration
import config
from config import GPTConfig

# Import Components
from tokenizers import create_tokenizer
from utils import load_and_prepare_data
from model import get_model
from trainers import get_trainer
from inference.generation import run_generation

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate transformer models")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default=config.DATASET_NAME,
                        help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default=config.DATASET_CONFIG,
                        help="Dataset configuration")
    parser.add_argument("--max_samples", type=int, default=config.MAX_SAMPLES,
                        help="Maximum number of samples to use")
    
    # Tokenizer arguments
    parser.add_argument("--tokenizer_type", type=str, default=config.TOKENIZER_TYPE,
                        help="Type of tokenizer to use")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to pretrained tokenizer (optional)")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="SASP",
                        help="Model architecture (SASP, Vanilla)")
    parser.add_argument("--n_layer", type=int, default=6,
                        help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=6,
                        help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=384,
                        help="Embedding dimension")
    parser.add_argument("--block_size", type=int, default=128,
                        help="Context window size")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for regularization")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda", "mps"],
                        help="Device to use for training (default: auto-detect)")
    parser.add_argument("--trainer", type=str, default="simple",
                        help="Trainer type to use")
    
    # Inference arguments
    parser.add_argument("--generation_max_len", type=int, default=50,
                        help="Max new tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip inference phase")
    
    # Miscellaneous
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save outputs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    return parser.parse_args()

def setup_output_directory(output_dir: str):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    return output_dir

def set_random_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

def main():
    """Main function to run the workflow."""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Create output directory
    output_dir = setup_output_directory(args.output_dir)
    
    # Create a configuration object from arguments
    model_config = config.create_config_from_args(args)
    
    ###################################################################
    # 1. Initialize Tokenizer
    ###################################################################
    logger.info("Initializing tokenizer...")
    
    # Create tokenizer from pretrained path or create new one
    if args.tokenizer_path:
        tokenizer = create_tokenizer(
            args.tokenizer_type, 
            args.tokenizer_path,
            **config.TOKENIZER_PARAMS
        )
        logger.info(f"Loaded tokenizer from: {args.tokenizer_path}")
    else:
        tokenizer = create_tokenizer(
            args.tokenizer_type,
            **config.TOKENIZER_PARAMS
        )
        logger.info(f"Created new {args.tokenizer_type} tokenizer")
    
    # Update config based on tokenizer
    model_config.update_from_tokenizer(tokenizer)
    
    ###################################################################
    # 2. Load and Prepare Data
    ###################################################################
    logger.info("Loading and preparing data...")
    
    dataloader, _ = load_and_prepare_data(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        max_seq_length=model_config.block_size,
        batch_size=model_config.batch_size
    )
    
    ###################################################################
    # 3. Initialize Model
    ###################################################################
    logger.info(f"Initializing {model_config.model_type} model...")
    
    # Print the configuration
    config.print_config(model_config)
    
    # Initialize the model
    model = get_model(model_config.model_type, config=model_config)
    model.to(config.DEVICE)
    
    ###################################################################
    # 4. Setup Optimizer
    ###################################################################
    logger.info("Setting up optimizer...")
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=model_config.learning_rate,
        weight_decay=model_config.weight_decay
    )
    
    ###################################################################
    # 5. Initialize Trainer
    ###################################################################
    logger.info(f"Initializing {args.trainer} trainer...")
    
    trainer = get_trainer(
        trainer_type=args.trainer,
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=config.DEVICE,
        num_epochs=model_config.num_epochs,
        output_dir=output_dir
    )
    
    ###################################################################
    # 6. Run Training
    ###################################################################
    logger.info("Starting training phase...")
    
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        return
    
    # Save the trained model
    model_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to: {model_path}")
    
    ###################################################################
    # 7. Run Inference (if not skipped)
    ###################################################################
    if args.skip_inference:
        logger.info("Skipping inference phase as requested.")
        return
    
    logger.info("Starting inference phase...")
    
    # Define test prompts
    test_prompts = [
        "The history of",
        "Scientists discovered that",
        "According to research",
        "During the 19th century",
        "Many people believe that"
    ]
    
    # Run generation for each prompt
    for i, prompt in enumerate(test_prompts):
        logger.info(f"\nGenerating text for prompt {i+1}: '{prompt}'")
        
        try:
            # Generate text
            generated_ids, generated_text = run_generation(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt,
                device=config.DEVICE,
                max_new_tokens=model_config.generation_max_len,
                temperature=model_config.temperature,
                top_k=model_config.top_k
            )
            
            # Log the generated text
            logger.info(f"Generated text: {generated_text}")
            
            # Save the generated text
            output_file = os.path.join(output_dir, f"generation_{i+1}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"Prompt: {prompt}\n\n")
                f.write(f"Generated text: {generated_text}\n")
            
        except Exception as e:
            logger.error(f"Error during generation: {e}", exc_info=True)
    
    logger.info("\nWorkflow complete!")

if __name__ == "__main__":
    main()
