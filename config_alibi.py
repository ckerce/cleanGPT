# ./config_alibi.py
"""
Configuration settings for cleanGPT Token-Factored models with ALiBi positional encoding.
Extends the base configuration to support ALiBi-specific parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import torch


@dataclass
class GPTConfigALiBi:
    """
    Configuration class for Token-Factored GPT models with ALiBi positional encoding.
    
    ALiBi (Attention with Linear Biases) replaces traditional positional embeddings
    with linear biases added to attention scores, enabling better length extrapolation.
    """
    
    # Model architecture parameters
    block_size: int = 128                    # Maximum sequence length during training
    vocab_size: Optional[int] = None         # Size of the vocabulary (set by tokenizer)
    n_layer: int = 6                         # Number of transformer layers
    n_head: int = 6                          # Number of attention heads
    n_embd: int = 384                        # Embedding dimension
    dropout: float = 0.1                     # Dropout probability
    bias: bool = False                       # Whether to use bias in linear layers
    padding_idx: Optional[int] = None        # Padding token index for embeddings
    
    # Model type specification
    model_type: str = "FactoredALiBi"        # Model architecture type
    
    # ALiBi-specific parameters
    max_position_embeddings: Optional[int] = None  # Maximum sequence length for generation (None = 4x block_size)
    alibi_max_bias: Optional[float] = None   # Maximum bias value (None = auto-compute)
    
    # Factored transformer specific parameters
    use_proj: bool = False                   # Legacy parameter for compatibility
    use_v: bool = False                      # Legacy parameter for compatibility
    llama_mlp: bool = False                  # Whether to use LLaMA-style MLP
    transformer_block_type: str = 'FactoredALiBi'  # Type of transformer block
    
    # Training parameters
    batch_size: int = 32                     # Batch size for training
    num_epochs: int = 5                      # Number of training epochs
    learning_rate: float = 0.25e-3           # Learning rate
    weight_decay: float = 0.01               # Weight decay for regularization
    
    # Generation parameters
    generation_max_len: int = 50             # Maximum length for text generation
    temperature: float = 0.8                 # Sampling temperature
    top_k: int = 50                          # Top-k sampling parameter
    
    # Additional configuration parameters
    extra_args: Dict[str, Any] = field(default_factory=dict)

    def update_from_tokenizer(self, tokenizer):
        """
        Update configuration parameters based on tokenizer properties.
        
        Args:
            tokenizer: The tokenizer object with vocab_size and padding information.
        """
        if hasattr(tokenizer, 'vocab_size'):
            self.vocab_size = tokenizer.vocab_size
        elif hasattr(tokenizer, '__len__'):
            self.vocab_size = len(tokenizer)
        
        # Set padding index if tokenizer has pad_token_id
        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            self.padding_idx = tokenizer.pad_token_id

    def __post_init__(self):
        """
        Post-initialization validation and setup.
        """
        # Ensure n_embd is divisible by n_head for multi-head attention
        assert self.n_embd % self.n_head == 0, \
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        
        # Set default max_position_embeddings if not specified (4x training length)
        if self.max_position_embeddings is None:
            self.max_position_embeddings = self.block_size * 4
        
        # Validate that max_position_embeddings is reasonable
        assert self.max_position_embeddings >= self.block_size, \
            "max_position_embeddings must be at least as large as block_size"


def print_config_alibi(cfg: GPTConfigALiBi = None, dataset_name=None, dataset_config=None, max_samples=None):
    """
    Print the configuration settings for ALiBi model in a formatted way.
    
    Args:
        cfg (GPTConfigALiBi): Configuration object to print
        dataset_name (str, optional): Name of the dataset being used
        dataset_config (str, optional): Dataset configuration name
        max_samples (int, optional): Maximum number of samples to use
    """
    if cfg is None:
        cfg = GPTConfigALiBi()
    
    print("=" * 60)
    print("TOKEN-FACTORED TRANSFORMER WITH ALiBi CONFIGURATION")
    print("=" * 60)
    
    # Model Architecture
    print("\n📐 MODEL ARCHITECTURE:")
    print(f"  Model Type:              {cfg.model_type}")
    print(f"  Transformer Block Type:  {cfg.transformer_block_type}")
    print(f"  Layers:                  {cfg.n_layer}")
    print(f"  Attention Heads:         {cfg.n_head}")
    print(f"  Embedding Dimension:     {cfg.n_embd}")
    print(f"  Head Dimension:          {cfg.n_embd // cfg.n_head}")
    print(f"  Vocabulary Size:         {cfg.vocab_size}")
    print(f"  Block Size (Training):   {cfg.block_size}")
    print(f"  Max Position Length:     {cfg.max_position_embeddings}")
    
    # ALiBi Specific
    print("\n🎯 ALiBi POSITIONAL ENCODING:")
    print(f"  Uses ALiBi:              Yes (No learned positional embeddings)")
    print(f"  Max Bias:                {cfg.alibi_max_bias or 'Auto-computed'}")
    print(f"  Length Extrapolation:    Up to {cfg.max_position_embeddings} tokens")
    print(f"  Extrapolation Ratio:     {cfg.max_position_embeddings / cfg.block_size:.1f}x training length")
    
    # Training Configuration
    print("\n🎓 TRAINING CONFIGURATION:")
    print(f"  Batch Size:              {cfg.batch_size}")
    print(f"  Number of Epochs:        {cfg.num_epochs}")
    print(f"  Learning Rate:           {cfg.learning_rate}")
    print(f"  Weight Decay:            {cfg.weight_decay}")
    print(f"  Dropout:                 {cfg.dropout}")
    print(f"  Bias in Linear Layers:   {cfg.bias}")
    print(f"  Padding Index:           {cfg.padding_idx}")
    
    # Generation Parameters
    print("\n🔮 GENERATION PARAMETERS:")
    print(f"  Max Generation Length:   {cfg.generation_max_len}")
    print(f"  Temperature:             {cfg.temperature}")
    print(f"  Top-K:                   {cfg.top_k}")
    
    # Dataset Information
    if dataset_name or dataset_config or max_samples:
        print("\n📊 DATASET INFORMATION:")
        if dataset_name:
            print(f"  Dataset Name:            {dataset_name}")
        if dataset_config:
            print(f"  Dataset Config:          {dataset_config}")
        if max_samples:
            print(f"  Max Samples:             {max_samples:,}")
    
    # Additional Parameters
    if cfg.extra_args:
        print("\n⚙️  EXTRA PARAMETERS:")
        for key, value in cfg.extra_args.items():
            print(f"  {key:24} {value}")
    
    # Model Statistics
    print("\n📈 MODEL STATISTICS:")
    estimated_params = estimate_model_parameters(cfg)
    print(f"  Estimated Parameters:    {estimated_params/1e6:.2f}M")
    print(f"  Memory per Token (fp32): ~{(cfg.n_embd * 4) / 1024:.1f} KB")
    print(f"  Context Window:          {cfg.block_size} (training) / {cfg.max_position_embeddings} (inference)")
    
    # Device information
    if torch.cuda.is_available():
        print(f"  CUDA Available:          Yes ({torch.cuda.get_device_name()})")
    else:
        print(f"  CUDA Available:          No")
    
    print("=" * 60)


def estimate_model_parameters(cfg: GPTConfigALiBi) -> int:
    """
    Estimate the number of parameters in the ALiBi model.
    
    Args:
        cfg (GPTConfigALiBi): Model configuration
        
    Returns:
        int: Estimated number of parameters
    """
    if cfg.vocab_size is None:
        vocab_size = 50257  # Default GPT-2 vocab size for estimation
    else:
        vocab_size = cfg.vocab_size
    
    # Token embeddings (shared with lm_head)
    token_emb_params = vocab_size * cfg.n_embd
    
    # No positional embeddings with ALiBi (this is the key advantage!)
    pos_emb_params = 0
    
    # Transformer layers
    layer_params = 0
    for _ in range(cfg.n_layer):
        # Layer norms (2 per layer: attn + mlp)
        ln_params = 2 * cfg.n_embd * (2 if cfg.bias else 1)  # weight + bias if enabled
        
        # Attention (Q, K, V projection)
        attn_qkv_params = cfg.n_embd * 3 * cfg.n_embd + (3 * cfg.n_embd if cfg.bias else 0)
        # Attention output projection
        attn_proj_params = cfg.n_embd * cfg.n_embd + (cfg.n_embd if cfg.bias else 0)
        
        # MLP (4x expansion factor)
        mlp_fc_params = cfg.n_embd * 4 * cfg.n_embd + (4 * cfg.n_embd if cfg.bias else 0)
        mlp_proj_params = 4 * cfg.n_embd * cfg.n_embd + (cfg.n_embd if cfg.bias else 0)
        
        layer_params += ln_params + attn_qkv_params + attn_proj_params + mlp_fc_params + mlp_proj_params
    
    # Final layer norm
    final_ln_params = cfg.n_embd * (2 if cfg.bias else 1)
    
    # ALiBi slopes (these are not learned parameters, just constants computed once)
    # No additional parameters needed for ALiBi!
    alibi_params = 0
    
    total_params = token_emb_params + pos_emb_params + layer_params + final_ln_params + alibi_params
    return total_params


def create_config_from_args_alibi(args):
    """
    Create a GPTConfigALiBi object from command line arguments.
    
    Args:
        args: Argument namespace from argparse
        
    Returns:
        GPTConfigALiBi: Configured model instance
    """
    config_dict = {}
    
    # Model architecture
    if hasattr(args, 'block_size') and args.block_size is not None:
        config_dict['block_size'] = args.block_size
    if hasattr(args, 'vocab_size') and args.vocab_size is not None:
        config_dict['vocab_size'] = args.vocab_size
    if hasattr(args, 'n_layer') and args.n_layer is not None:
        config_dict['n_layer'] = args.n_layer
    if hasattr(args, 'n_head') and args.n_head is not None:
        config_dict['n_head'] = args.n_head
    if hasattr(args, 'n_embd') and args.n_embd is not None:
        config_dict['n_embd'] = args.n_embd
    if hasattr(args, 'dropout') and args.dropout is not None:
        config_dict['dropout'] = args.dropout
    if hasattr(args, 'bias') and args.bias is not None:
        config_dict['bias'] = args.bias
    
    # ALiBi specific
    if hasattr(args, 'max_position_embeddings') and args.max_position_embeddings is not None:
        config_dict['max_position_embeddings'] = args.max_position_embeddings
    if hasattr(args, 'alibi_max_bias') and args.alibi_max_bias is not None:
        config_dict['alibi_max_bias'] = args.alibi_max_bias
    
    # Training parameters
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        config_dict['batch_size'] = args.batch_size
    if hasattr(args, 'num_epochs') and args.num_epochs is not None:
        config_dict['num_epochs'] = args.num_epochs
    if hasattr(args, 'learning_rate') and args.learning_rate is not None:
        config_dict['learning_rate'] = args.learning_rate
    if hasattr(args, 'weight_decay') and args.weight_decay is not None:
        config_dict['weight_decay'] = args.weight_decay
    
    # Generation parameters
    if hasattr(args, 'generation_max_len') and args.generation_max_len is not None:
        config_dict['generation_max_len'] = args.generation_max_len
    if hasattr(args, 'temperature') and args.temperature is not None:
        config_dict['temperature'] = args.temperature
    if hasattr(args, 'top_k') and args.top_k is not None:
        config_dict['top_k'] = args.top_k
    
    # Collect any extra arguments that don't map to standard config parameters
    extra_args = {}
    standard_config_keys = {
        'block_size', 'vocab_size', 'n_layer', 'n_head', 'n_embd', 'dropout', 'bias',
        'max_position_embeddings', 'alibi_max_bias', 'batch_size', 'num_epochs',
        'learning_rate', 'weight_decay', 'generation_max_len', 'temperature', 'top_k'
    }
    
    for key, value in vars(args).items():
        if key not in standard_config_keys and not key.startswith('_') and key not in config_dict:
            extra_args[key] = value
    
    if extra_args:
        config_dict['extra_args'] = extra_args
    
    return GPTConfigALiBi(**config_dict)


def get_alibi_config_presets():
    """
    Get predefined configuration presets for different model sizes with ALiBi.
    
    Returns:
        Dict[str, GPTConfigALiBi]: Dictionary of preset configurations
    """
    presets = {
        'tiny': GPTConfigALiBi(
            n_layer=2,
            n_head=2,
            n_embd=128,
            block_size=128,
            max_position_embeddings=512,
            dropout=0.1,
            batch_size=64,
            learning_rate=5e-4,
            num_epochs=3,
        ),
        'small': GPTConfigALiBi(
            n_layer=6,
            n_head=6,
            n_embd=192,
            block_size=256,
            max_position_embeddings=1024,
            dropout=0.1,
            batch_size=32,
            learning_rate=6e-4,
            num_epochs=5,
        ),
        'medium': GPTConfigALiBi(
            n_layer=6,
            n_head=6,
            n_embd=768,
            block_size=128,
            max_position_embeddings=256,
            dropout=0.1,
            batch_size=32,
            learning_rate=5e-4,
            num_epochs=10,
        ),
        'large': GPTConfigALiBi(
            n_layer=12,
            n_head=12,
            n_embd=768,
            block_size=1024,
            max_position_embeddings=4096,
            dropout=0.1,
            batch_size=8,
            learning_rate=2e-4,
            num_epochs=15,
        ),
        'character': GPTConfigALiBi(
            n_layer=6,
            n_head=6,
            n_embd=384,
            block_size=256,
            max_position_embeddings=1024,
            vocab_size=256,  # Character-level
            dropout=0.1,
            batch_size=64,
            learning_rate=1e-3,
            num_epochs=8,
        ),
    }
    return presets


def load_alibi_config_preset(preset_name: str) -> GPTConfigALiBi:
    """
    Load a predefined configuration preset.
    
    Args:
        preset_name (str): Name of the preset ('tiny', 'small', 'medium', 'large', 'character')
        
    Returns:
        GPTConfigALiBi: The preset configuration
        
    Raises:
        ValueError: If preset_name is not recognized
    """
    presets = get_alibi_config_presets()
    if preset_name not in presets:
        available = ', '.join(presets.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")
    
    return presets[preset_name]


def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """
    Compute ALiBi slopes for the given number of attention heads.
    
    This function computes the slopes used in ALiBi attention bias.
    Each head gets a different slope, creating different rates of decay
    for different relative positions.
    
    Args:
        n_heads (int): Number of attention heads
        
    Returns:
        torch.Tensor: Slopes tensor of shape (n_heads,)
    """
    def get_slopes_power_of_2(n_heads):
        start = (2**(-2**-(math.log2(n_heads)-3)))
        ratio = start
        return [start*ratio**i for i in range(n_heads)]

    def get_slopes_non_power_of_2(n_heads):
        closest_power_of_2 = 2**math.floor(math.log2(n_heads))
        slopes_power_of_2 = get_slopes_power_of_2(closest_power_of_2)
        
        if closest_power_of_2 == n_heads:
            return slopes_power_of_2
        else:
            # Handle non-power-of-2 case
            extra_base = 2**(-(2**-(math.log2(2*closest_power_of_2)-3)))
            num_remaining_heads = min(closest_power_of_2, n_heads - closest_power_of_2)
            extra_slopes = [extra_base * (extra_base**i) for i in range(num_remaining_heads)]
            return slopes_power_of_2 + extra_slopes

    import math
    
    if n_heads & (n_heads - 1) == 0:  # Check if power of 2
        slopes = get_slopes_power_of_2(n_heads)
    else:
        slopes = get_slopes_non_power_of_2(n_heads)
    
    return torch.tensor(slopes, dtype=torch.float32)


def compute_alibi_bias(seq_len: int, n_heads: int, device: torch.device = None) -> torch.Tensor:
    """
    Compute ALiBi attention bias matrix.
    
    Args:
        seq_len (int): Sequence length
        n_heads (int): Number of attention heads
        device (torch.device, optional): Device to place tensor on
        
    Returns:
        torch.Tensor: ALiBi bias tensor of shape (n_heads, seq_len, seq_len)
    """
    # Get slopes for each head
    slopes = get_alibi_slopes(n_heads)
    if device is not None:
        slopes = slopes.to(device)
    
    # Create relative position matrix
    # For causal attention: positions to the right should have large negative bias
    positions = torch.arange(seq_len, device=device).unsqueeze(0) - torch.arange(seq_len, device=device).unsqueeze(1)
    positions = positions.abs()  # Take absolute value for distance
    
    # Apply slopes to create bias matrix for each head
    # Shape: (n_heads, seq_len, seq_len)
    alibi_bias = slopes.unsqueeze(-1).unsqueeze(-1) * positions.unsqueeze(0)
    
    # Make it negative (ALiBi uses negative bias)
    alibi_bias = -alibi_bias
    
    # For causal masking, set future positions to very negative values
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    alibi_bias = alibi_bias.masked_fill(causal_mask.unsqueeze(0), float('-inf'))
    
    return alibi_bias


# Example usage and testing
if __name__ == "__main__":
    # Test configuration creation and printing
    print("Testing GPTConfigALiBi...")
    
    # Default configuration
    config = GPTConfigALiBi()
    print_config_alibi(config, dataset_name="wikitext", max_samples=10000)
    
    # Test presets
    print("\n" + "="*80)
    print("TESTING PRESETS")
    print("="*80)
    
    for preset_name in ['tiny', 'small', 'medium', 'large', 'character']:
        print(f"\n--- {preset_name.upper()} PRESET ---")
        preset_config = load_alibi_config_preset(preset_name)
        params = estimate_model_parameters(preset_config)
        print(f"Parameters: {params/1e6:.2f}M")
        print(f"Context: {preset_config.block_size} → {preset_config.max_position_embeddings}")
        print(f"Architecture: {preset_config.n_layer}L-{preset_config.n_head}H-{preset_config.n_embd}D")
        print(f"Extrapolation: {preset_config.max_position_embeddings/preset_config.block_size:.1f}x")
    
    # Test ALiBi slopes computation
    print("\n" + "="*80)
    print("TESTING ALiBi SLOPES")
    print("="*80)
    
    for n_heads in [4, 8, 12, 16]:
        slopes = get_alibi_slopes(n_heads)
        print(f"Heads: {n_heads:2d} | Slopes: {slopes.tolist()}")
    
    # Test ALiBi bias computation
    print("\n--- ALiBi Bias Matrix (4 heads, 8 seq_len) ---")
    bias_matrix = compute_alibi_bias(8, 4)
    print(f"Shape: {bias_matrix.shape}")
    print("Head 0 bias matrix:")
    print(bias_matrix[0].numpy())
    
    print("\nALiBi configuration testing completed!")
