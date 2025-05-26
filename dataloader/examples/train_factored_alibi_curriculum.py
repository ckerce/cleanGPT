#!/usr/bin/env python
# ./examples/train_factored_alibi_example.py
"""
Example script demonstrating how to train a Token-Factored Transformer model with ALiBi.
This script shows the complete workflow from data preparation to training and generation,
highlighting the benefits of ALiBi for length extrapolation.

The script supports both single dataset training and curriculum learning with multiple datasets.
Curriculum learning allows for sophisticated blending strategies between datasets over time.

Usage examples:

# Single dataset training (existing behavior)
python ./examples/train_factored_alibi_example.py \
  --dataset "roneneldan/TinyStories" \
  --max_samples 1000000 \
  --preset medium \
  --block_size 128 \
  --max_position_embeddings 256 \
  --batch_size 128 \
  --num_epochs 10 \
  --tokenizer_type gpt2 \
  --output_dir "./outputs/output_alibi_v3" \
  --test_generation \
  --use_v \
  --use_proj

# Curriculum learning: Wikipedia + Code
python ./examples/train_factored_alibi_example.py \
  --use_curriculum \
  --curriculum_datasets "wikimedia/wikipedia" "code_search_net" \
  --curriculum_configs "20231101.en" "python" \
  --curriculum_strategy linear_transition \
  --curriculum_start_weights 0.4 0.6 \
  --curriculum_end_weights 0.8 0.2 \
  --curriculum_transition_epochs 6 \
  --preset medium \
  --num_epochs 12 \
  --batch_size 64 \
  --max_samples 500000 \
  --output_dir "./outputs/curriculum_wiki_code" \
  --test_generation

# Alternative: Start code-heavy, transition to Wikipedia-heavy
python ./examples/train_factored_alibi_example.py \
  --use_curriculum \
  --curriculum_datasets "wikimedia/wikipedia" "code_search_net" \
  --curriculum_configs "20231101.en" "python" \
  --curriculum_strategy linear_transition \
  --curriculum_start_weights 0.2 0.8 \
  --curriculum_end_weights 0.7 0.3 \
  --curriculum_transition_epochs 8 \
  --preset medium \
  --num_epochs 15 \
  --learning_rate 3e-4 \
  --output_dir "./outputs/curriculum_code_to_wiki"
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


def load_curriculum_data(args, tokenizer, config, logger):
    """Load data using curriculum learning if enabled."""
    
    if not args.use_curriculum:
        # Use existing single dataset loading
        logger.info(f"Loading single dataset: {args.dataset}")
        return load_and_prepare_data(
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            tokenizer=tokenizer,
            max_samples=args.max_samples,
            max_seq_length=config.block_size,
            batch_size=config.batch_size,
            mlm=False,
            split='train',
            shuffle=True
        )
    
    # Import curriculum components
    try:
        from dataloaders import get_curriculum_dataloader
    except ImportError:
        logger.error("Curriculum learning dataloaders module not found. Please ensure the dataloaders package is available.")
        logger.error("You may need to create the dataloaders module or use --use_curriculum=False")
        sys.exit(1)
    
    logger.info("Setting up curriculum learning with multiple datasets...")
    
    # Parse dataset configurations
    datasets = []
    dataset_configs = []
    
    if not args.curriculum_datasets:
        raise ValueError("--curriculum_datasets required when using --use_curriculum")
    
    # Parse curriculum configs if provided
    curriculum_configs_list = []
    if args.curriculum_configs:
        curriculum_configs_list = args.curriculum_configs
    
    # Load each dataset
    for i, dataset_name in enumerate(args.curriculum_datasets):
        logger.info(f"Loading dataset {i+1}: {dataset_name}")
        
        # Get dataset config if provided
        dataset_config = None
        if i < len(curriculum_configs_list):
            dataset_config = curriculum_configs_list[i] if curriculum_configs_list[i] != 'None' else None
        
        # Load dataset
        try:
            if dataset_config:
                dataset = load_dataset(dataset_name, dataset_config, split='train', streaming=True)
                logger.info(f"Loaded {dataset_name} with config {dataset_config}")
            else:
                dataset = load_dataset(dataset_name, split='train', streaming=True)
                logger.info(f"Loaded {dataset_name} without config")
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise
        
        datasets.append(dataset)
        
        # Create dataset config with automatic text field detection
        config_dict = {
            'name': dataset_name.replace('/', '_').replace('-', '_'),
            'shuffle': True
        }
        
        # Automatic text field detection based on dataset name
        if 'code' in dataset_name.lower() or 'github' in dataset_name.lower():
            config_dict['text_field'] = 'func_code_string'  # CodeSearchNet
        elif 'wikipedia' in dataset_name.lower():
            config_dict['text_field'] = 'text'
        elif 'tinystories' in dataset_name.lower():
            config_dict['text_field'] = 'text'
        elif 'openwebtext' in dataset_name.lower():
            config_dict['text_field'] = 'text'
        else:
            config_dict['text_field'] = 'text'  # Default fallback
            logger.warning(f"Using default text field 'text' for dataset {dataset_name}. "
                         f"You may need to customize this if the dataset uses a different field name.")
        
        dataset_configs.append(config_dict)
        logger.info(f"Dataset config for {dataset_name}: {config_dict}")
    
    # Set up curriculum strategy
    strategy_params = {}
    
    if args.curriculum_strategy == 'linear_transition':
        start_weights = args.curriculum_start_weights or [1.0 / len(datasets)] * len(datasets)
        end_weights = args.curriculum_end_weights or [1.0 / len(datasets)] * len(datasets)
        
        if len(start_weights) != len(datasets):
            raise ValueError(f"Number of start weights ({len(start_weights)}) must match number of datasets ({len(datasets)})")
        if len(end_weights) != len(datasets):
            raise ValueError(f"Number of end weights ({len(end_weights)}) must match number of datasets ({len(datasets)})")
        
        strategy_params = {
            'start_weights': start_weights,
            'end_weights': end_weights,
            'transition_epochs': args.curriculum_transition_epochs
        }
        
        logger.info(f"Linear transition: {start_weights} -> {end_weights} over {args.curriculum_transition_epochs} epochs")
    
    elif args.curriculum_strategy == 'step_schedule':
        # For step schedule, you would need to define the schedule
        # This is a simplified version - you might want to add command line args for this
        schedule = [
            {'epoch': 0, 'weights': args.curriculum_start_weights or [0.5] * len(datasets)},
            {'epoch': args.curriculum_transition_epochs // 2, 'weights': [0.5] * len(datasets)},
            {'epoch': args.curriculum_transition_epochs, 'weights': args.curriculum_end_weights or [0.5] * len(datasets)}
        ]
        strategy_params = {'schedule': schedule}
        logger.info(f"Step schedule: {schedule}")
    
    elif args.curriculum_strategy == 'exponential_decay':
        start_weights = args.curriculum_start_weights or [1.0 / len(datasets)] * len(datasets)
        # Simple decay rates - could be made configurable
        decay_rates = [0.1] * len(datasets)  # You might want to make this configurable
        strategy_params = {
            'start_weights': start_weights,
            'decay_rates': decay_rates
        }
        logger.info(f"Exponential decay: start_weights={start_weights}, decay_rates={decay_rates}")
    
    strategy_config = {
        'name': args.curriculum_strategy,
        'params': strategy_params
    }
    
    logger.info(f"Curriculum strategy: {args.curriculum_strategy}")
    logger.info(f"Strategy params: {strategy_params}")
    
    # Create curriculum dataloader
    dataloader = get_curriculum_dataloader(
        datasets=datasets,
        dataset_configs=dataset_configs,
        strategy=strategy_config,
        tokenizer=tokenizer,
        max_seq_length=config.block_size,
        batch_size=config.batch_size,
        seed=42  # For reproducibility
    )
    
    logger.info(f"Curriculum dataloader created with {len(datasets)} datasets")
    
    return dataloader, tokenizer


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
        "Once upon a time there was a little cat",  # Short prompt
        "The rabbit found a big carrot in the garden. She was so happy because it was the biggest carrot she had ever seen",  # Medium prompt
        "Tommy the turtle was very slow but he never gave up. Every day he would walk to the pond to see his friends the frogs and ducks. One sunny morning Tommy decided he wanted to learn how to swim just like his friends. He took a deep breath and slowly stepped into the cool water",  # Long prompt
        "def fibonacci(n):",  # Short code prompt
        "class DataProcessor:\n    def __init__(self, data_path):\n        self.data_path = data_path\n        self.processed_data = []",  # Medium code prompt
        "import numpy as np\nimport pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier\n\ndef create_machine_learning_pipeline(dataset_path, target_column):",  # Long code prompt
        "The red ball rolled",  # Short prompt
        "Sam saw a big dog. The dog wagged its tail and wanted to play fetch in the sunny park",  # Medium prompt
        "Every night before bed, Lucy would look out her window at the stars. She would make a wish on the brightest star she could find. Tonight was special because there was a shooting star dancing across the dark sky"  # Long prompt
    ]
    
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
    
    # Dataset arguments - single dataset mode
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories",
                        help="Dataset name for single dataset mode (e.g., 'roneneldan/TinyStories', 'wikimedia/wikipedia')")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Dataset configuration for single dataset mode (e.g., '20231101.en')")
    parser.add_argument("--max_samples", type=int, default=10000,
                        help="Maximum number of samples to use from the dataset(s)")
    
    # Curriculum Learning arguments
    parser.add_argument('--use_curriculum', action='store_true',
                       help='Enable curriculum learning with multiple datasets')
    parser.add_argument('--curriculum_datasets', nargs='+', default=None,
                       help='List of dataset names for curriculum learning (e.g., "wikimedia/wikipedia" "code_search_net")')
    parser.add_argument('--curriculum_configs', nargs='+', default=None,
                       help='Dataset configurations for curriculum datasets (e.g., "20231101.en" "python")')
    parser.add_argument('--curriculum_strategy', type=str, default='linear_transition',
                       choices=['linear_transition', 'step_schedule', 'exponential_decay'],
                       help='Curriculum learning strategy')
    parser.add_argument('--curriculum_start_weights', nargs='+', type=float, default=None,
                       help='Starting weights for each dataset in curriculum learning')
    parser.add_argument('--curriculum_end_weights', nargs='+', type=float, default=None,
                       help='Ending weights for each dataset in curriculum learning')
    parser.add_argument('--curriculum_transition_epochs', type=int, default=5,
                       help='Number of epochs for curriculum transition')
    
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
    """Main training function with curriculum learning support."""
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
        
        if args.use_curriculum:
            logger.info("Curriculum learning ENABLED")
            logger.info(f"Curriculum datasets: {args.curriculum_datasets}")
            logger.info(f"Curriculum strategy: {args.curriculum_strategy}")
        else:
            logger.info("Single dataset mode")
            logger.info(f"Dataset: {args.dataset}")
            if args.dataset_config:
                logger.info(f"Dataset config: {args.dataset_config}")
        
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
                # For character tokenizer with curriculum learning, we need to handle this differently
                if args.use_curriculum:
                    logger.warning("Character tokenizer with curriculum learning requires manual vocab building.")
                    logger.warning("Consider using pre-built character tokenizer or GPT-2 tokenizer for curriculum learning.")
                
                # Load a subset of data for vocab building
                temp_split_str = f"train[:{min(args.max_samples, 10000)}]"
                dataset_name = args.curriculum_datasets[0] if args.use_curriculum else args.dataset
                dataset_config = None
                if args.use_curriculum and args.curriculum_configs:
                    dataset_config = args.curriculum_configs[0] if args.curriculum_configs[0] != 'None' else None
                elif not args.use_curriculum:
                    dataset_config = args.dataset_config
                
                temp_dataset_args = [dataset_name]
                if dataset_config:
                    temp_dataset_args.append(dataset_config)
                
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
        if args.use_curriculum:
            print_config_alibi(config, dataset_name="Curriculum Learning", 
                             dataset_config=f"{len(args.curriculum_datasets)} datasets", 
                             max_samples=args.max_samples)
        else:
            print_config_alibi(config, dataset_name=args.dataset, 
                             dataset_config=args.dataset_config, 
                             max_samples=args.max_samples)
        
        # --- Load Data (with curriculum support) ---
        logger.info("Loading and preparing data...")
        
        dataloader, tokenizer = load_curriculum_data(args, tokenizer, config, logger)
        
        # Log curriculum schedule visualization if using curriculum learning
        if args.use_curriculum and hasattr(dataloader, 'curriculum_dataset'):
            try:
                from dataloaders.utils import visualize_curriculum_schedule
                strategy = dataloader.curriculum_dataset.strategy
                dataset_names = [cfg['name'] for cfg in dataloader.curriculum_dataset.dataset_configs]
                
                fig = visualize_curriculum_schedule(strategy, config.num_epochs, dataset_names)
                schedule_path = os.path.join(logs_dir, f'curriculum_schedule_{timestamp}.png')
                fig.savefig(schedule_path, dpi=150, bbox_inches='tight')
                logger.info(f"Curriculum schedule visualization saved to: {schedule_path}")
                import matplotlib.pyplot as plt
                plt.close(fig)
            except ImportError:
                logger.warning("matplotlib not available, skipping curriculum visualization")
            except Exception as e:
                logger.warning(f"Could not create curriculum visualization: {e}")
        
        logger.info(f"Data loaded. DataLoader has {len(dataloader) if hasattr(dataloader, '__len__') else 'streaming'} batches.")
        
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
        
        # --- Training Loop with Curriculum Support ---
        logger.info("="*60)
        logger.info("STARTING TRAINING")
        logger.info("="*60)
        logger.info(f"Training for {config.num_epochs} epochs")
        logger.info(f"Batch size: {config.batch_size}")
        logger.info(f"Sequence length: {config.block_size}")
        logger.info(f"Learning rate: {config.learning_rate}")
        
        if args.use_curriculum:
            logger.info("Using curriculum learning - weights will change over epochs")
        
        try:
            # If using curriculum learning, we need to handle epoch updates
            if args.use_curriculum and hasattr(dataloader, 'update_epoch'):
                # Custom training loop to handle curriculum updates
                logger.info("Starting curriculum-aware training loop...")
                
                for epoch in range(config.num_epochs):
                    logger.info(f"Starting epoch {epoch + 1}/{config.num_epochs}")
                    
                    # Update curriculum for this epoch
                    dataloader.update_epoch(epoch)
                    
                    # Train one epoch
                    if hasattr(trainer, 'train_epoch'):
                        trainer.train_epoch(epoch)
                    elif hasattr(trainer, 'train_one_epoch'):
                        trainer.train_one_epoch(dataloader, epoch)
                    else:
                        # Fallback: modify trainer to support epoch-based training
                        logger.warning("Trainer doesn't support single epoch training. Using standard training.")
                        logger.warning("Curriculum weights will be updated but trainer may not respect epoch boundaries.")
                        trainer.train()
                        break  # Exit after training completes
                        
                logger.info("Curriculum training completed!")
            else:
                # Standard training
                logger.info("Starting standard training...")
                trainer.train()
                
        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            sys.exit(1)
        
        logger.info("="*60)
        logger.info("TRAINING COMPLETED")
        logger.info("="*60)
        
        # --- Save Model and Tokenizer ---
        model_path = os.path.join(args.output_dir, args.save_model_filename)
        save_dict = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'tokenizer': tokenizer,
            'training_args': vars(args),
            'timestamp': timestamp,
        }
        
        # Add curriculum information if applicable
        if args.use_curriculum:
            save_dict['curriculum_info'] = {
                'datasets': args.curriculum_datasets,
                'configs': args.curriculum_configs,
                'strategy': args.curriculum_strategy,
                'start_weights': args.curriculum_start_weights,
                'end_weights': args.curriculum_end_weights,
                'transition_epochs': args.curriculum_transition_epochs
            }
        
        torch.save(save_dict, model_path)
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
            
            # Choose appropriate test prompts based on training data
            if args.use_curriculum and any('code' in dataset.lower() for dataset in args.curriculum_datasets):
                # Include both text and code prompts for curriculum models
                test_prompts = [
                    "The old house stood on a hill overlooking the",
                    "Once upon a time, there was a brave knight who",
                    "def fibonacci(n):",
                    "class DataProcessor:",
                    "import numpy as np",
                    "The recipe for disaster is simple:",
                ]
            else:
                # Standard text prompts
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
                if args.use_curriculum:
                    gen_file.write(f"Training mode: Curriculum learning\n")
                    gen_file.write(f"Datasets: {', '.join(args.curriculum_datasets)}\n")
                else:
                    gen_file.write(f"Training mode: Single dataset\n")
                    gen_file.write(f"Dataset: {args.dataset}\n")
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
                            if args.use_curriculum:
                                f.write(f"Training: Curriculum learning ({', '.join(args.curriculum_datasets)})\n")
                            else:
                                f.write(f"Training: Single dataset ({args.dataset})\n")
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
        
        if args.use_curriculum:
            print(f"Curriculum learning with {len(args.curriculum_datasets)} datasets:")
            for i, dataset in enumerate(args.curriculum_datasets):
                config_str = f" (config: {args.curriculum_configs[i]})" if args.curriculum_configs and i < len(args.curriculum_configs) else ""
                print(f"  - {dataset}{config_str}")
            print(f"Strategy: {args.curriculum_strategy}")
            if args.curriculum_start_weights and args.curriculum_end_weights:
                print(f"Weight transition: {args.curriculum_start_weights} -> {args.curriculum_end_weights}")
        else:
            print(f"Single dataset training: {args.dataset}")
            if args.dataset_config:
                print(f"Dataset config: {args.dataset_config}")
        
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
