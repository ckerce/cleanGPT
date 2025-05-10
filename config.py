# -*- coding: utf-8 -*-
############################################
#                                          #
#  Configuration Settings                  #
#                                          #
############################################

import torch
import time
from dataclasses import dataclass # Use dataclass for config

# --- Dataset ---
DATASET_NAME = "wikipedia"
DATASET_CONFIG = "20220301.simple"
MAX_SAMPLES = 10000

# --- Tokenizer ---
TOKENIZER_NAME = "gpt2" # Uses BPE

# --- Model Config ---
# Parameters for the SASP Transformer Model
@dataclass
class GPTConfig:
    block_size: int = 128      # Max sequence length (context window)
    vocab_size: int = 50257    # Set dynamically later from tokenizer
    n_layer: int = 6           # Number of transformer layers
    n_head: int = 6            # Number of attention heads
    n_embd: int = 384          # Embedding dimension (must be divisible by n_head)
    dropout: float = 0.1       # Dropout rate
    bias: bool = False         # Use bias in Linear layers and LayerNorm?
    # SASP Specific Configs (can be overridden)
    use_proj: bool = False     # Use projection layer in CausalShapedAttention
    use_v: bool = False        # Use Value vector in CausalShapedAttention (QK vs QKV)
    llama_mlp: bool = False    # Use LLaMA-style MLP variant (requires specific N calc)
    transformer_block_type: str = 'PreLN' # Type of block ('SASP', 'PreLN', etc.)

# --- Training ---
BATCH_SIZE = 32 
NUM_EPOCHS = 5 
LEARNING_RATE = 1e-4 # Adjusted LR for transformer
# TARGET_PARAM_VALUE = 0.75 # No longer needed for transformer

# --- Inference ---
GENERATION_MAX_LEN = 50 # Increased slightly

# --- Environment ---
# Determine device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEVICE_NAME = torch.cuda.get_device_name(0)
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DEVICE_NAME = "MPS (Apple Silicon GPU)"
else:
    DEVICE = torch.device("cpu")
    DEVICE_NAME = "CPU"

CURRENT_TIME = time.strftime('%Y-%m-%d %H:%M:%S %Z')

# --- Function to print config ---
def print_config(cfg: GPTConfig = None): # Accept optional config object
    print("--- Configuration ---")
    print(f"Run Time: {CURRENT_TIME}")
    print(f"Device: {DEVICE_NAME} ({DEVICE})")
    print("\n[Dataset]")
    print(f"  Name: {DATASET_NAME} ({DATASET_CONFIG})")
    print(f"  Max Samples: {MAX_SAMPLES}")
    print("\n[Tokenizer]")
    print(f"  Name: {TOKENIZER_NAME}")
    if cfg: # Print model config if provided
        print("\n[Model: GPTConfig]")
        print(f"  Block Size (Max Seq Len): {cfg.block_size}")
        print(f"  Vocab Size: {cfg.vocab_size}")
        print(f"  Embedding Dim (n_embd): {cfg.n_embd}")
        print(f"  Num Layers (n_layer): {cfg.n_layer}")
        print(f"  Num Heads (n_head): {cfg.n_head}")
        print(f"  Dropout: {cfg.dropout}")
        print(f"  Bias: {cfg.bias}")
        print(f"  Transformer Block Type: {cfg.transformer_block_type}")
        print(f"  SASP Use Proj: {cfg.use_proj}")
        print(f"  SASP Use V: {cfg.use_v}")
        print(f"  SASP LLaMA MLP: {cfg.llama_mlp}")
    print("\n[Training]")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Num Epochs: {NUM_EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print("\n[Inference]")
    print(f"  Generation Max Length: {GENERATION_MAX_LEN}")
    print("--------------------")
