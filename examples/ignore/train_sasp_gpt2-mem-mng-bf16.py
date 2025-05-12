#!/usr/bin/env python
# ./examples/train_sasp_gpt2-mem-mng-bf16.py
"""
Example script demonstrating how to train a SASP model with GPT-2 tokenizer
with BFloat16 mixed precision and memory management optimizations.

Example calls:

# Basic training with BFloat16 precision
python examples/train_sasp_gpt2-mem-mng-bf16.py \
  --dataset "roneneldan/TinyStories" \
  --max_samples 5000 \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 192 \
  --block_size 512 \
  --batch_size 16 \
  --gradient_accumulation_steps 4 \
  --num_epochs 5 \
  --output_dir "./outputs/sasp_gpt2_large_bf16"

# For larger datasets with memory constraints
python examples/train_sasp_gpt2-mem-mng-bf16.py \
  --dataset "roneneldan/TinyStories" \
  --max_samples 100000 \
  --n_layer 6 \
  --n_head 6 \
  --n_embd 192 \
  --block_size 512 \
  --batch_size 32 \
  --gradient_accumulation_steps 1 \
  --clip_grad_norm 1.0 \
  --num_epochs 10 \
  --output_dir "./outputs/sasp_gpt2_large_bf16"

# Fallback to FP32 if BFloat16 not supported
python examples/train_sasp_gpt2-mem-mng-bf16.py \
  --dataset "roneneldan/TinyStories" \
  --max_samples 5000 \
  --n_layer 4 \
  --n_head 4 \
  --n_embd 128 \
  --block_size 256 \
  --precision fp32 \
  --batch_size 16 \
  --num_epochs 5 \
  --output_dir "./outputs/sasp_gpt2_fp32"
"""

import argparse
import logging
import torch
import torch.nn.functional as F
from datasets import load_dataset
import os
import sys
import numpy as np
import gc  # For garbage collection
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler  # For mixed precision training
import traceback  # For detailed error reporting

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
    parser = argparse.ArgumentParser(description="Train SASP model with GPT-2 tokenizer using BFloat16")
    
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
    parser.add_argument("--learning_rate", type=float, default=2e-3,
                        help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients before updating")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="bf16",
                        help="Precision for training: fp32 (full), fp16 (half), or bf16 (brain float)")
    # Keep --fp16 for backward compatibility  
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 mixed precision (deprecated, use --precision fp16)")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for gradient clipping")
    
    # Tokenizer arguments
    parser.add_argument("--use_fast_tokenizer", action="store_true", default=True,
                        help="Use fast tokenizer implementation")
    
    # Memory management
    parser.add_argument("--reduce_vocab", action="store_true",
                        help="Reduce vocabulary size for memory efficiency")
    parser.add_argument("--vocab_size", type=int, default=None,
                        help="Custom vocabulary size if reducing vocab")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="../outputs/gpt2_sasp_bf16",
                        help="Directory to save outputs")
    
    return parser.parse_args()

def main():
    """Main function to run the training with BFloat16."""
    # Parse arguments
    args = parse_args()
    
    # For backward compatibility
    if args.fp16 and args.precision == "bf16":
        args.precision = "fp16"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Using precision: {args.precision}")
    
    ###################################################################
    # 1. Load Dataset
    ###################################################################
    logger.info(f"Loading dataset: {args.dataset}")
    
    # Load dataset with optional configuration
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
    # 2. Create and initialize GPT-2 tokenizer
    ###################################################################
    logger.info("Creating GPT-2 tokenizer...")
    
    # Create a GPT-2 tokenizer
    tokenizer = create_tokenizer('gpt2', use_fast=args.use_fast_tokenizer)
    
    # Ensure the tokenizer has appropriate padding settings
    if tokenizer.pad_token is None:
        # Set EOS token as padding token if none exists
        tokenizer.pad_token = tokenizer.eos_token
    
    # Optionally reduce vocabulary size for memory efficiency
    if args.reduce_vocab:
        vocab_size = args.vocab_size or 10000  # Default to 10K if not specified
        
        logger.info(f"Reducing vocabulary size from {tokenizer.vocab_size} to {vocab_size}")
        
        # This is a simplified approach - in a real scenario, you'd analyze token frequencies
        # and keep the most common tokens, but this will work for demonstration purposes
        from transformers import GPT2TokenizerFast
        
        # Get the original vocab and sort by token ID
        original_vocab = tokenizer.get_vocab()
        sorted_vocab = sorted(original_vocab.items(), key=lambda x: x[1])
        
        # Keep only special tokens and the first N tokens
        reduced_vocab = {}
        special_tokens = [tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token]
        
        # First add special tokens
        for token in special_tokens:
            if token in original_vocab:
                reduced_vocab[token] = len(reduced_vocab)
        
        # Then add the most common tokens up to vocab_size
        for token, _ in sorted_vocab:
            if len(reduced_vocab) >= vocab_size:
                break
            if token not in reduced_vocab:
                reduced_vocab[token] = len(reduced_vocab)
        
        # Create a new tokenizer with the reduced vocabulary
        # This is just a demonstration - in practice, you'd need to handle
        # merges and other tokenizer configuration properly
        tokenizer.vocab_size = len(reduced_vocab)
        
    logger.info(f"GPT-2 tokenizer initialized with vocabulary size: {tokenizer.vocab_size}")
    
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
    
    # Update the vocabulary size from our tokenizer
    model_config.update_from_tokenizer(tokenizer)
    
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
    
    # Check if BFloat16 is supported if that's the requested precision
    if args.precision == "bf16" and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            logger.warning("BFloat16 is not supported on this GPU. Falling back to FP32.")
            args.precision = "fp32"
        else:
            logger.info("BFloat16 is supported on this device.")
    
    # Define a custom data collator for language modeling training
    class GPT2LMDataCollator:
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
    data_collator = GPT2LMDataCollator(tokenizer, model_config.block_size)
    
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
    
    # Convert model to the appropriate precision
    if args.precision == "bf16":
        model = model.to(torch.bfloat16)
        logger.info("Model converted to BFloat16 precision")
    elif args.precision == "fp16":
        model = model.to(torch.float16)
        logger.info("Model converted to FP16 precision")
    
    # Move model to device
    model = model.to(device)
    
    # Print model parameter types and count
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized with {param_count/1e6:.2f}M parameters")
    logger.info(f"Model parameter dtype: {next(model.parameters()).dtype}")
    
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
    
    # We'll use a custom training loop instead of the trainer class
    # when using mixed precision or gradient accumulation
    use_custom_loop = args.gradient_accumulation_steps > 1 or args.precision != "fp32"
    
    if not use_custom_loop:
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
    
    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Configure mixed precision based on precision argument
    use_mixed_precision = args.precision in ["fp16", "bf16"]
    
    if use_mixed_precision:
        if args.precision == "fp16":
            logger.info("Using FP16 mixed precision with gradient scaling")
            scaler = GradScaler('cuda')
        else:  # bf16
            logger.info("Using BFloat16 mixed precision without gradient scaling")
            scaler = None  # BFloat16 usually doesn't need a scaler
    else:
        logger.info("Using FP32 precision (no mixed precision)")
        scaler = None
    
    # Use the trainer if no custom loop needed
    if not use_custom_loop:
        trainer.train()
    else:
        # Custom training loop for mixed precision and gradient accumulation
        model.train()
        total_steps = len(dataloader) * args.num_epochs
        global_step = 0
        tr_loss = 0.0
        
        logger.info(f"Starting custom training loop for {args.num_epochs} epochs with {len(dataloader)} steps per epoch")
        if args.gradient_accumulation_steps > 1:
            logger.info(f"Using gradient accumulation with {args.gradient_accumulation_steps} steps")
        
        for epoch in range(args.num_epochs):
            epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
            epoch_loss = 0.0
            
            for step, batch in enumerate(epoch_iterator):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Convert batch to the right precision
                if args.precision == "bf16":
                    batch = {k: v.to(torch.bfloat16) if k != "labels" else v for k, v in batch.items()}
                elif args.precision == "fp16":
                    batch = {k: v.to(torch.float16) if k != "labels" else v for k, v in batch.items()}
                
                # Gradient accumulation setup
                accumulation_step = (step + 1) % args.gradient_accumulation_steps == 0
                
                if use_mixed_precision:
                    # Determine the datatype to use
                    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
                    
                    # Mixed precision forward pass
                    with autocast('cuda', dtype=dtype):
                        outputs = model(**batch)
                        loss = outputs['loss']
                        loss = loss / args.gradient_accumulation_steps  # Normalize loss
                    
                    # Mixed precision backward pass
                    if scaler:  # For FP16
                        scaler.scale(loss).backward()
                        
                        if accumulation_step:
                            # Gradient clipping
                            if args.clip_grad_norm > 0:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                            
                            # Update weights
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            global_step += 1
                    else:  # For BFloat16 (no scaler)
                        loss.backward()
                        
                        if accumulation_step:
                            # Gradient clipping
                            if args.clip_grad_norm > 0:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                            
                            # Update weights
                            optimizer.step()
                            optimizer.zero_grad()
                            global_step += 1
                else:
                    # Standard forward pass
                    outputs = model(**batch)
                    loss = outputs['loss']
                    loss = loss / args.gradient_accumulation_steps  # Normalize loss
                    
                    # Standard backward pass
                    loss.backward()
                    
                    if accumulation_step:
                        # Gradient clipping
                        if args.clip_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                        
                        # Update weights
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1
                
                # Track loss
                tr_loss += loss.item()
                epoch_loss += loss.item()
                
                # Update progress bar
                if global_step > 0:
                    epoch_iterator.set_postfix({"loss": f"{tr_loss/global_step:.4f}"})
                else:
                    epoch_iterator.set_postfix({"loss": f"{tr_loss:.4f}"})
                
                # Check for NaN loss and log warning
                if torch.isnan(loss):
                    logger.warning(f"NaN loss detected at step {step} of epoch {epoch+1}!")
                    if args.precision == "bf16":
                        logger.warning("Consider enabling gradient scaling or switching to FP16")
                
                # Save checkpoint periodically
                if global_step > 0 and global_step % 1000 == 0:
                    checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': tr_loss / np.max([global_step,1]),
                    }, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Log epoch results
            avg_epoch_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{args.num_epochs} completed with average loss: {avg_epoch_loss:.4f}")
            
            # Save after each epoch
            epoch_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), epoch_path)
            logger.info(f"Saved model after epoch {epoch+1} to {epoch_path}")
            
            # Clear cache between epochs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
    
    # Save the final model
    model_path = os.path.join(args.output_dir, "sasp_gpt2_model.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save model config for later loading
    config_path = os.path.join(args.output_dir, "model_config.json")
    model_config.save_to_json(config_path)
    logger.info(f"Model configuration saved to {config_path}")
    
    ###################################################################
    # 9. Generate Sample Text
    ###################################################################
    logger.info("Generating sample text...")
    
    # Test prompts
    test_prompts = [
        "Once upon a time",
        "The future of artificial intelligence",
        "In a world where technology"
    ]
    
    # Set the model to evaluation mode
    model.eval()
    
    # Generate text for each prompt
    for i, prompt in enumerate(test_prompts):
        logger.info(f"\nGenerating text for prompt: '{prompt}'")
        
        try:
            # Encode the prompt
            input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors='pt')
            input_ids = input_ids.to(device)
            
            # For BFloat16, ensure model and inputs are correctly typed
            if args.precision == "bf16":
                # Note: don't try to convert input_ids to bfloat16, keep them as int64
                # We only ensure the model parameters are the right type
                
                # Debug output
                logger.info(f"Input IDs dtype: {input_ids.dtype}")
                logger.info(f"Model parameter dtype: {next(model.parameters()).dtype}")
                
                # Generate with explicit casting inside model
                with torch.no_grad():
                    output_ids = model.generate(
                        idx=input_ids,
                        max_new_tokens=100,
                        temperature=0.8,
                        top_k=40
                    )
            else:
                # For fp32 or fp16
                with torch.no_grad():
                    output_ids = model.generate(
                        idx=input_ids,
                        max_new_tokens=100,
                        temperature=0.8,
                        top_k=40
                    )
            
            # Decode the generated text (keep model in its precision)
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
            # Print the complete traceback for better debugging
            logger.error(traceback.format_exc())
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
