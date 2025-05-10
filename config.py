# ./config.py
"""
Configuration Settings for cleanGPT
Enhanced for modular tokenization and model architecture
"""

import torch
import time
from dataclasses import dataclass, field
from typing import Dict, List, Union, Optional, Any

# --- Dataset ---
DATASET_NAME = "wikipedia"
DATASET_CONFIG = "20220301.simple"
MAX_SAMPLES = 10000

# --- Tokenizer ---
TOKENIZER_TYPE = "gpt2"  # Options: "character", "gpt2", "byte_level", etc.
TOKENIZER_PARAMS = {
    "use_fast": True,
}

# --- Model Config ---
@dataclass
class GPTConfig:
    """Configuration for Transformer models."""
    
    # Core parameters
    block_size: int = 128                # Max sequence length (context window)
    vocab_size: int = None               # Set dynamically from tokenizer
    n_layer: int = 6                     # Number of transformer layers
    n_head: int = 6                      # Number of attention heads
    n_embd: int = 384                    # Embedding dimension (must be divisible by n_head)
    dropout: float = 0.1                 # Dropout rate
    bias: bool = False                   # Use bias in Linear layers and LayerNorm?
    padding_idx: Optional[int] = None    # Padding token ID from tokenizer
    
    # Architecture selection
    model_type: str = "SASP"             # Model architecture type: "SASP", "Vanilla", etc.
    
    # SASP Specific Configs
    use_proj: bool = False               # Use projection in CausalShapedAttention
    use_v: bool = False                  # Use Value vector in CausalShapedAttention
    llama_mlp: bool = False              # Use LLaMA-style MLP 
    transformer_block_type: str = 'SASP' # SASP vs PreLN block structure
    
    # Training settings
    batch_size: int = 32                 # Training batch size
    num_epochs: int = 5                  # Number of training epochs
    learning_rate: float = 1e-4          # Learning rate for optimizer
    weight_decay: float = 0.01           # Weight decay for regularization
    
    # Inference settings
    generation_max_len: int = 50         # Maximum new tokens for generation
    temperature: float = 0.8             # Sampling temperature
    top_k: int = 50                      # Top-k sampling parameter
    
    # Extra fields for user extensions
    extra_args: Dict[str, Any] = field(default_factory=dict)
    
    def update_from_tokenizer(self, tokenizer):
        """Update config based on tokenizer properties."""
        self.vocab_size = tokenizer.vocab_size
        self.padding_idx = tokenizer.pad_token_id
        # Use tokenizer's max length if it's smaller than our block_size
        if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length < self.block_size:
            self.block_size = tokenizer.model_max_length

# --- Environment ---
# Determine device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEVICE_NAME = torch.cuda.get_device_name(0)
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DEVICE_NAME = "MPS (Apple Silicon GPU)"
else:
    DEVICE = torch.device("cpu")
    DEVICE_NAME = "CPU"

CURRENT_TIME = time.strftime('%Y-%m-%d %H:%M:%S %Z')

# --- Function to print config ---
def print_config(cfg: GPTConfig = None, dataset_name=None, dataset_config=None, max_samples=None):
    """Print the configuration settings."""
    # Use provided values or fall back to global defaults
    dataset_name = dataset_name or DATASET_NAME
    dataset_config = dataset_config or DATASET_CONFIG
    max_samples = max_samples or MAX_SAMPLES

    print("--- Configuration ---")
    print(f"Run Time: {CURRENT_TIME}")
    print(f"Device: {DEVICE_NAME} ({DEVICE})")

    print("\n[Dataset]")
    print(f"  Name: {dataset_name}" + (f" ({dataset_config})" if dataset_config else ""))
    print(f"  Max Samples: {max_samples}")

    print("\n[Tokenizer]")
    print(f"  Type: {TOKENIZER_TYPE}")
    print(f"  Parameters: {TOKENIZER_PARAMS}")
    
    if cfg: 
        print("\n[Model Config]")
        print(f"  Model Type: {cfg.model_type}")
        print(f"  Block Size (Max Seq Len): {cfg.block_size}")
        print(f"  Vocab Size: {cfg.vocab_size}")
        print(f"  Embedding Dim (n_embd): {cfg.n_embd}")
        print(f"  Num Layers (n_layer): {cfg.n_layer}")
        print(f"  Num Heads (n_head): {cfg.n_head}")
        print(f"  Dropout: {cfg.dropout}")
        print(f"  Bias: {cfg.bias}")
        
        if cfg.model_type == "SASP":
            print(f"  Transformer Block Type: {cfg.transformer_block_type}")
            print(f"  SASP Use Proj: {cfg.use_proj}")
            print(f"  SASP Use V: {cfg.use_v}")
            print(f"  SASP LLaMA MLP: {cfg.llama_mlp}")
        
        print("\n[Training]")
        print(f"  Batch Size: {cfg.batch_size}")
        print(f"  Num Epochs: {cfg.num_epochs}")
        print(f"  Learning Rate: {cfg.learning_rate}")
        print(f"  Weight Decay: {cfg.weight_decay}")
        
        print("\n[Inference]")
        print(f"  Generation Max Length: {cfg.generation_max_len}")
        print(f"  Temperature: {cfg.temperature}")
        print(f"  Top-k: {cfg.top_k}")
        
        # Print any extra arguments
        if cfg.extra_args:
            print("\n[Extra Args]")
            for k, v in cfg.extra_args.items():
                print(f"  {k}: {v}")
    
    print("--------------------")


def create_config_from_args(args):
    """
    Create a config object from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        GPTConfig object with values from args
    """
    # Create a base config
    config = GPTConfig()
    
    # Update fields from args if they exist
    for field_name in [f.name for f in fields(GPTConfig)]:
        if hasattr(args, field_name):
            setattr(config, field_name, getattr(args, field_name))
    
    # Handle special cases and derived values
    if hasattr(args, 'device'):
        global DEVICE, DEVICE_NAME
        if args.device == 'cpu':
            DEVICE = torch.device('cpu')
            DEVICE_NAME = 'CPU'
        elif args.device == 'cuda' and torch.cuda.is_available():
            DEVICE = torch.device('cuda')
            DEVICE_NAME = torch.cuda.get_device_name(0)
        elif args.device == 'mps' and torch.backends.mps.is_available():
            DEVICE = torch.device('mps')
            DEVICE_NAME = 'MPS (Apple Silicon GPU)'
    
    return config
