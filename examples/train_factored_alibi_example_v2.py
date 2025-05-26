#!/usr/bin/env python
# ./examples/train_factored_alibi_example.py
"""
Example script demonstrating how to train a Token-Factored Transformer model with ALiBi.
This script shows the complete workflow from data preparation to training and generation,
highlighting the benefits of ALiBi for length extrapolation.

Usage examples:
python ./examples/train_factored_alibi_example.py \
  --dataset "roneneldan/TinyStories" \
  --max_samples 1000000 \
  --preset medium \
  --block_size 128 \
  --max_position_embeddings 256 \
  --batch_size 128 \
  --num_epochs 10 \
  --tokenizer_type gpt2 \
  --output_dir "./outputs/alibi_test" \
  --test_generation \
  --use_v \
  --use_proj

python ./examples/train_factored_alibi_example.py \
  --dataset "wikimedia/wikipedia" \
  --dataset_config "20231101.en" \
  --preset medium \
  --block_size 256 \
  --batch_size 128 \
  --max_position_embeddings 512 \
  --tokenizer_type gpt2 \
  --output_dir "./outputs/alibi-use_v-use_proj" \
  --test_generation \
  --use_v \
  --use_proj \
  --num_epochs 10 \
  --max_samples 1000000 

python ./examples/train_factored_alibi_example.py \
  --dataset "wikimedia/wikipedia" \
  --dataset_config "20231101.en" \
  --preset medium \
  --max_samples 50000 \
  --batch_size 8 \
  --num_epochs 5 \
  --device cuda
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import logging
from datetime import datetime
import json

# Add the parent directory to sys.path to access the cleanGPT modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config_alibi import GPTConfigALiBi, print_config_alibi, load_alibi_config_preset, create_config_from_args_alibi
from mytokenizers import create_tokenizer
from model import get_model
from utils.data_utils import load_and_prepare_data
from trainers import get_trainer
from inference.generation import run_generation
from datasets import load_dataset


class TeeOutput:
    """Class to redirect output to both console and file simultaneously."""
    def __init__(self, file_path, mode='w'):
        self.file = open(file_path, mode, encoding='utf-8', buffering=1)  # Line buffered
        self.stdout = sys.stdout
        
    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.file.flush()  # Ensure immediate write to file
        
    def flush(self):
        self.stdout.flush()
        self.file.flush()
        
    def close(self):
        if hasattr(self, 'file') and self.file:
            self.file.close()


def setup_logging(output_dir):
    """Set up comprehensive logging to both console and files."""
    # Create logs subdirectory
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up main training log
    training_log_path = os.path.join(logs_dir, f'training_{timestamp}.log')
    
    # Configure logging to write to both console and file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(training_log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Set up additional output redirection for print statements
    output_log_path = os.path.join(logs_dir, f'output_{timestamp}.log')
    tee_output = TeeOutput(output_log_path)
    
    return logger, tee_output, logs_dir, timestamp


def log_system_info(logger, device):
    """Log system and environment information."""
    logger.info("="*60)
    logger.info("SYSTEM INFORMATION")
    logger.info("="*60)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info("="*60)


def save_training_config(args, config, output_dir, timestamp):
    """Save training configuration to JSON file."""
    config_dict = {
        'timestamp': timestamp,
        'command_line_args': vars(args),
        'model_config': {
            'model_type': config.model_type,
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_embd': config.n_embd,
            'block_size': config.block_size,
            'max_position_embeddings': config.max_position_embeddings,
            'vocab_size': config.vocab_size,
            'dropout': config.dropout,
            'bias': config.bias,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'num_epochs': config.num_epochs,
            'weight_decay': config.weight_decay,
        }
    }
    
    config_path = os.path.join(output_dir, 'logs', f'training_config_{timestamp}.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2)
    
    return config_path


def test_length_extrapolation(model, tokenizer, device, config, output_dir, timestamp):
    """
    Test the model's ability to extrapolate to longer sequences than it was trained on.
    This demonstrates the key benefit of ALiBi.
    """
    print("\n" + "="*60)
    print("TESTING LENGTH EXTRAPOLATION WITH ALiBi")
    print("="*60)
    
    # Create extrapolation test log file
    extrap_log_path = os.path.join(output_dir, 'logs', f'extrapolation_test_{timestamp}.log')
    
    model.eval()
    
    # Test prompts of varying lengths
    test_prompts = [
        # Short Prompts
        "The Renaissance was a period in European history, covering the span between the 14th and 17th centuries.",
        "Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.",
        "The Amazon River in South America is the largest river by discharge volume of water in the world.",
    
        # Medium Prompts
        "The Industrial Revolution, which began in Great Britain in the late 18th century, was a period of major industrialization and technological advancement. It marked a shift from hand production methods to machines, new chemical manufacturing and iron production processes, and the increasing use of steam power.",
        "Quantum mechanics is a fundamental theory in physics that describes the physical properties of nature at the scale of atoms and subatomic particles. It is the foundation of all quantum physics including quantum chemistry, quantum field theory, quantum technology, and quantum information science.",
        "The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, wood, and other materials, generally built along an east-to-west line across the historical northern borders of China. Its purpose was to protect the Chinese states and empires against the raids and invasions of the various nomadic groups of the Eurasian Steppe.",
    
        # Long Prompts
        "The history of the internet has its roots in the development of electronic computers in the 1950s and the emergence of ARPANET in the late 1960s. The ARPANET, commissioned by the United States Department of Defense, was one of the first operational packet switching networks. It served as a backbone for the development of protocols like TCP/IP, which became the standard for internet communication. By the early 1990s, the advent of the World Wide Web, developed by Tim Berners-Lee, made the internet accessible to a wider public, leading to its explosive growth.",
        "Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil, and gas. Burning fossil fuels generates greenhouse gas emissions that act like a blanket wrapped around the Earth, trapping the sun’s heat and raising temperatures. The consequences include more frequent and intense droughts, storms, heat waves, rising sea levels, warming oceans, and loss of biodiversity.",
        "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars, automated decision-making, and competing at the highest level in strategic game systems. As machines become increasingly capable, tasks considered to require 'intelligence' are often removed from the definition of AI, a phenomenon known as the AI effect."
    ]
#    test_prompts = [
#        "Once upon a time there was a little cat",  # Short prompt
#        "The rabbit found a big carrot in the garden. She was so happy because it was the biggest carrot she had ever seen",  # Medium prompt
#        "Tommy the turtle was very slow but he never gave up. Every day he would walk to the pond to see his friends the frogs and ducks. One sunny morning Tommy decided he wanted to learn how to swim just like his friends. He took a deep breath and slowly stepped into the cool water",  # Long prompt
#        "The little mouse was hungry",  # Short prompt
#        "Lily planted seeds in her garden. When the rain came, she watched from her window as tiny green shoots began to grow",  # Medium prompt
#        "The old oak tree in the park had been there for many years. Children would climb its branches and build tree houses in its shade. One day a little girl named Emma discovered something magical hidden in a hollow of the tree trunk",  # Long prompt
#        "The red ball rolled",  # Short prompt
#        "Sam saw a big dog. The dog wagged its tail and wanted to play fetch in the sunny park",  # Medium prompt
#        "Every night before bed, Lucy would look out her window at the stars. She would make a wish on the brightest star she could find. Tonight was special because there was a shooting star dancing across the dark sky"  # Long prompt
#    ]
    
    # Test generation at different lengths
    generation_lengths = [50, 100, 200]  # These may exceed training block_size
    
    with open(extrap_log_path, 'w', encoding='utf-8') as extrap_file:
        extrap_file.write(f"Length Extrapolation Test Results\n")
        extrap_file.write(f"Timestamp: {timestamp}\n")
        extrap_file.write(f"Model: {config.model_type}\n")
        extrap_file.write(f"Training block_size: {config.block_size}\n")
        extrap_file.write(f"Max position embeddings: {config.max_position_embeddings}\n")
        extrap_file.write("="*80 + "\n\n")
        
        for i, prompt in enumerate(test_prompts):
            test_result = f"\n--- Test {i+1}: Prompt Length ~{len(prompt.split())} words ---\n"
            test_result += f"Prompt: \"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"\n"
            
            print(test_result.strip())
            extrap_file.write(test_result)
            
            # Tokenize prompt
            input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=True)], device=device)
            prompt_length = input_ids.size(1)
            
            length_info = f"Prompt tokens: {prompt_length}\n"
            print(length_info.strip())
            extrap_file.write(length_info)
            
            for gen_len in generation_lengths:
                if prompt_length + gen_len > config.max_position_embeddings:
                    skip_msg = f"Skipping generation length {gen_len} (would exceed max_position_embeddings)\n"
                    print(skip_msg.strip())
                    extrap_file.write(skip_msg)
                    continue
                    
                gen_info = f"\nGenerating {gen_len} tokens (total sequence: {prompt_length + gen_len}):\n"
                print(gen_info.strip())
                extrap_file.write(gen_info)
                
                try:
                    with torch.no_grad():
                        generated = model.generate(
                            input_ids, 
                            max_new_tokens=gen_len,
                            temperature=0.8,
                            top_k=40
                        )
                    
                    # Decode only the generated part
                    generated_text = tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True)
                    result = f"Generated: \"{generated_text[:200]}{'...' if len(generated_text) > 200 else ''}\"\n"
                    print(result.strip())
                    extrap_file.write(result)
                    extrap_file.write(f"Full generated text: {generated_text}\n")
                except Exception as e:
                    error_msg = f"Error during generation: {e}\n"
                    print(error_msg.strip())
                    extrap_file.write(error_msg)
            
            extrap_file.write("\n" + "-"*80 + "\n")
    
    print(f"Length extrapolation test results saved to: {extrap_log_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Token-Factored Transformer with ALiBi')
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories",
                        help="Dataset name (e.g., 'roneneldan/TinyStories', 'wikimedia/wikipedia')")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Dataset configuration (e.g., '20231101.en')")
    parser.add_argument("--max_samples", type=int, default=10000,
                        help="Maximum number of samples to use from the dataset")
    
    # Model configuration
    parser.add_argument('--preset', type=str, default='small', 
                       choices=['small', 'medium', 'large', 'character'],
                       help='Model size preset')
    parser.add_argument('--block_size', type=int, default=None,
                       help='Training sequence length (overrides preset)')
    parser.add_argument('--max_position_embeddings', type=int, default=None,
                       help='Maximum sequence length for inference (overrides preset)')
    parser.add_argument("--n_layer", type=int, default=None, help="Number of transformer layers (overrides preset)")
    parser.add_argument("--n_head", type=int, default=None, help="Number of attention heads (overrides preset)")
    parser.add_argument("--n_embd", type=int, default=None, help="Embedding dimension (overrides preset)")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout probability (overrides preset)")
    parser.add_argument("--bias", action='store_true', help="Use bias in linear layers")
    parser.add_argument("--no_bias", action='store_false', dest='bias', help="Do not use bias")
    parser.add_argument("--use_v", action='store_true', help="Use value factorization")
    parser.add_argument("--use_proj", action='store_true', help="Use projection layers")
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training (overrides preset)')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (overrides preset)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, 
                       help="Max norm for gradient clipping")
    parser.add_argument("--trainer_type", type=str, default="simple",
                       help="Type of trainer to use")
    
    # Data and tokenizer
    parser.add_argument('--tokenizer_type', type=str, default='gpt2',
                       choices=['gpt2', 'character'],
                       help='Type of tokenizer to use')
    parser.add_argument("--tokenizer_path", type=str, default=None,
                       help="Path to a pretrained tokenizer directory")
    
    # Output and logging
    parser.add_argument('--output_dir', type=str, default='./output_alibi',
                       help='Output directory for checkpoints')
    parser.add_argument('--log_interval', type=int, default=256,
                       help='Logging interval')
    parser.add_argument("--save_model_filename", type=str, default="alibi_model.pt",
                       help="Filename for the saved model checkpoint")
    parser.add_argument("--save_tokenizer_dirname", type=str, default="alibi_tokenizer",
                       help="Directory name for saved tokenizer")
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use (auto, cpu, cuda, mps)')
    
    # Generation testing
    parser.add_argument('--test_generation', action='store_true',
                       help='Test generation and length extrapolation after training')
    parser.add_argument("--skip_generation", action="store_true", 
                       help="Skip sample text generation after training")
    parser.add_argument("--generation_max_len", type=int, default=50, 
                       help="Max new tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.8, 
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, 
                       help="Top-k sampling parameter")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # --- Setup Output Directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Setup Comprehensive Logging ---
    logger, tee_output, logs_dir, timestamp = setup_logging(args.output_dir)
    
    # Redirect print statements to both console and file
    original_stdout = sys.stdout
    sys.stdout = tee_output
    
    try:
        logger.info("="*60)
        logger.info("STARTING ALIBI TRAINING SESSION")
        logger.info("="*60)
        logger.info(f"Session timestamp: {timestamp}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Logs directory: {logs_dir}")
        
        # --- Set Device ---
        if args.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(args.device)
        
        logger.info(f"Using device: {device}")
        
        # Log system information
        log_system_info(logger, device)
        
        # --- Load Configuration Preset and Update with Args ---
        logger.info(f"Loading '{args.preset}' configuration preset...")
        config = load_alibi_config_preset(args.preset)
        
        # Override preset values with command line arguments if provided
        if args.block_size is not None:
            config.block_size = args.block_size
        if args.max_position_embeddings is not None:
            config.max_position_embeddings = args.max_position_embeddings
        if args.n_layer is not None:
            config.n_layer = args.n_layer
        if args.n_head is not None:
            config.n_head = args.n_head
        if args.n_embd is not None:
            config.n_embd = args.n_embd
        if args.dropout is not None:
            config.dropout = args.dropout
        if args.bias is not None:
            config.bias = args.bias
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.learning_rate is not None:
            config.learning_rate = args.learning_rate
        
        # Handle new arguments
        if hasattr(config, 'use_v') and args.use_v:
            config.use_v = args.use_v
        if hasattr(config, 'use_proj') and args.use_proj:
            config.use_proj = args.use_proj
        
        # Always update these from args
        config.num_epochs = args.num_epochs
        config.weight_decay = args.weight_decay
        config.generation_max_len = args.generation_max_len
        config.temperature = args.temperature
        config.top_k = args.top_k
        
        # Re-run post_init to validate updated configuration
        config.__post_init__()
        
        # Save training configuration
        config_path = save_training_config(args, config, args.output_dir, timestamp)
        logger.info(f"Training configuration saved to: {config_path}")
        
        # --- Initialize Tokenizer ---
        logger.info(f"Initializing {args.tokenizer_type} tokenizer...")
        tokenizer_params = {}
        
        if args.tokenizer_path:
            tokenizer = create_tokenizer(args.tokenizer_type, from_pretrained=args.tokenizer_path, **tokenizer_params)
            logger.info(f"Loaded {args.tokenizer_type} tokenizer from: {args.tokenizer_path}")
        else:
            if args.tokenizer_type == "gpt2":
                tokenizer_params['use_fast'] = True
                tokenizer = create_tokenizer(args.tokenizer_type, **tokenizer_params)
                logger.info(f"Created new {args.tokenizer_type} tokenizer with default settings.")
            elif args.tokenizer_type == "character":
                logger.info("Building character tokenizer vocab from dataset...")
                # Load a subset of data for vocab building
                temp_split_str = f"train[:{min(args.max_samples, 10000)}]"
                temp_dataset_args = [args.dataset]
                if args.dataset_config:
                    temp_dataset_args.append(args.dataset_config)
                
                try:
                    temp_dataset = load_dataset(*temp_dataset_args, split=temp_split_str, trust_remote_code=True)
                except Exception as e:
                    logger.error(f"Failed to load dataset for character vocab building: {e}")
                    sys.exit(1)

                # Find text field
                if 'text' in temp_dataset.column_names:
                    text_samples_for_vocab = temp_dataset['text']
                elif 'story' in temp_dataset.column_names:
                    text_samples_for_vocab = temp_dataset['story']
                else:
                    text_field_for_vocab = next((col for col in temp_dataset.column_names 
                                               if temp_dataset.features[col].dtype == 'string'), None)
                    if not text_field_for_vocab:
                        logger.error(f"Could not find text column. Available columns: {temp_dataset.column_names}")
                        sys.exit(1)
                    logger.info(f"Using text column: '{text_field_for_vocab}' for character vocab.")
                    text_samples_for_vocab = temp_dataset[text_field_for_vocab]

                # Ensure text samples is a list of strings
                if not isinstance(text_samples_for_vocab, list) or (text_samples_for_vocab and not isinstance(text_samples_for_vocab[0], str)):
                    text_samples_for_vocab = [str(item) for item in text_samples_for_vocab]

                char_tokenizer_instance = create_tokenizer(args.tokenizer_type, **tokenizer_params)
                char_tokenizer_instance.build_vocab_from_texts(text_samples_for_vocab)
                tokenizer = char_tokenizer_instance
                logger.info(f"Character tokenizer vocabulary built. Vocab size: {tokenizer.vocab_size}")
            else:
                tokenizer = create_tokenizer(args.tokenizer_type, **tokenizer_params)
                logger.info(f"Created new {args.tokenizer_type} tokenizer with default settings.")
        
        # Update config with tokenizer info
        config.update_from_tokenizer(tokenizer)
        
        # Print configuration
        print_config_alibi(config, dataset_name=args.dataset, dataset_config=args.dataset_config, max_samples=args.max_samples)
        
        # --- Load and Prepare Data ---
        logger.info("Loading and preparing data...")
        logger.info(f"Dataset: {args.dataset}")
        if args.dataset_config:
            logger.info(f"Dataset config: {args.dataset_config}")
        logger.info(f"Max samples: {args.max_samples}")
        
        dataloader, tokenizer = load_and_prepare_data(
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            tokenizer=tokenizer,
            max_samples=args.max_samples,
            max_seq_length=config.block_size,
            batch_size=config.batch_size,
            mlm=False,  # Causal LM
            split='train',
            shuffle=True
        )
        logger.info(f"Data loaded. DataLoader has {len(dataloader)} batches.")
        
        # --- Initialize Model ---
        logger.info(f"Initializing {config.model_type} model...")
        model = get_model(config.model_type, config=config).to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"{config.model_type} model initialized with {num_params/1e6:.2f}M parameters.")
        
        # --- Setup Optimizer ---
        logger.info("Setting up optimizer...")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        logger.info(f"Optimizer: AdamW with lr={config.learning_rate}, weight_decay={config.weight_decay}")
        
        # --- Initialize Trainer ---
        logger.info(f"Initializing {args.trainer_type} trainer...")
        trainer_kwargs = {
            'num_epochs': config.num_epochs,
            'output_dir': args.output_dir,
            'clip_grad_norm': args.clip_grad_norm,
            'log_interval': args.log_interval
        }
        trainer = get_trainer(
            trainer_type=args.trainer_type,
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            **trainer_kwargs
        )
        
        # --- Train the Model ---
        logger.info("="*60)
        logger.info("STARTING TRAINING")
        logger.info("="*60)
        logger.info(f"Training for {config.num_epochs} epochs")
        logger.info(f"Batch size: {config.batch_size}")
        logger.info(f"Sequence length: {config.block_size}")
        logger.info(f"Learning rate: {config.learning_rate}")
        
        try:
            trainer.train()
        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            sys.exit(1)
        
        logger.info("="*60)
        logger.info("TRAINING COMPLETED")
        logger.info("="*60)
        
        # --- Save Model and Tokenizer ---
        model_path = os.path.join(args.output_dir, args.save_model_filename)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'tokenizer': tokenizer,
            'training_args': vars(args),
            'timestamp': timestamp,
        }, model_path)
        logger.info(f"{config.model_type} model saved to {model_path}")
        
        # Save tokenizer
        if args.tokenizer_type == "character" or args.tokenizer_path is not None:
            tokenizer_save_path = os.path.join(args.output_dir, args.save_tokenizer_dirname)
            try:
                tokenizer.save_pretrained(tokenizer_save_path)
                logger.info(f"Tokenizer saved to {tokenizer_save_path}")
            except Exception as e:
                logger.error(f"Could not save tokenizer: {e}")
        
        # --- Generate Sample Text ---
        if not args.skip_generation:
            logger.info("="*60)
            logger.info("GENERATING SAMPLE TEXT")
            logger.info("="*60)
            
            test_prompts = [
                "The old house stood on a hill overlooking the",
                "Once upon a time, there was a brave knight who",
                "The recipe for disaster is simple:",
                "In the year 2077, cybernetics"
            ]
            model.eval()

            generation_results_path = os.path.join(logs_dir, f'generation_results_{timestamp}.log')
            with open(generation_results_path, 'w', encoding='utf-8') as gen_file:
                gen_file.write(f"Sample Text Generation Results\n")
                gen_file.write(f"Timestamp: {timestamp}\n")
                gen_file.write(f"Model: {config.model_type}\n")
                gen_file.write("="*80 + "\n\n")
                
                for i, prompt_text in enumerate(test_prompts):
                    logger.info(f"\nGenerating for prompt: '{prompt_text}'")
                    gen_file.write(f"Prompt {i+1}: {prompt_text}\n")
                    gen_file.write("-" * 50 + "\n")
                    
                    try:
                        _, generated_text = run_generation(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_text=prompt_text,
                            device=device,
                            max_new_tokens=args.generation_max_len,
                            temperature=args.temperature,
                            top_k=args.top_k,
                            show_progress=False
                        )
                        logger.info(f"Generated text: {generated_text}")
                        gen_file.write(f"Generated text: {generated_text}\n\n")
                        
                        # Also save individual files
                        output_file = os.path.join(args.output_dir, f"alibi_generation_{i+1}.txt")
                        with open(output_file, "w", encoding="utf-8") as f:
                            f.write(f"Model: {config.model_type}\n")
                            f.write(f"Timestamp: {timestamp}\n")
                            f.write(f"Prompt: {prompt_text}\n\n")
                            f.write(f"Generated text:\n{generated_text}\n")
                    except Exception as e:
                        error_msg = f"Error generating text for prompt '{prompt_text}': {e}"
                        logger.error(error_msg)
                        gen_file.write(f"ERROR: {error_msg}\n\n")
            
            logger.info(f"Generation results saved to: {generation_results_path}")
        
        # --- Test Length Extrapolation ---
        if args.test_generation:
            logger.info("="*60)
            logger.info("STARTING LENGTH EXTRAPOLATION TESTS")
            logger.info("="*60)
            test_length_extrapolation(model, tokenizer, device, config, args.output_dir, timestamp)
        
        # --- Final Summary ---
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Key benefits of ALiBi demonstrated:")
        print(f"- No learned positional embeddings (saves parameters)")
        print(f"- Can extrapolate to sequences longer than training length")
        print(f"- Training length: {config.block_size} tokens")
        print(f"- Max inference length: {config.max_position_embeddings} tokens")
        print(f"- Model saved with {num_params/1e6:.2f}M parameters")
        print(f"- Session logs saved in: {logs_dir}")
        print("="*60)
        
        logger.info("Training session completed successfully!")
        
    finally:
        # Restore original stdout and close file handles
        sys.stdout = original_stdout
        if hasattr(tee_output, 'close'):
            tee_output.close()


if __name__ == "__main__":
    main()
