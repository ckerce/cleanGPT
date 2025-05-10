# ./inference/sampling_strategies.py
"""
Sampling Strategies for Text Generation
Provides functions for various token sampling approaches
"""

import torch
import torch.nn.functional as F
import logging
from typing import List, Dict, Optional, Union, Callable

logger = logging.getLogger(__name__)

def greedy_sampling(logits: torch.Tensor) -> torch.Tensor:
    """
    Greedy sampling - always select the most likely token.
    
    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        
    Returns:
        Tensor of token indices with shape (batch_size, 1)
    """
    # Get the highest probability token
    _, next_tokens = torch.max(logits, dim=-1)
    return next_tokens.unsqueeze(-1)

def temperature_sampling(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Temperature sampling - adjust probability distribution then sample.
    
    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        temperature: Sampling temperature (1.0 = no change,
                    <1.0 = less random, >1.0 = more random)
        
    Returns:
        Tensor of token indices with shape (batch_size, 1)
    """
    # Apply temperature scaling
    if temperature != 1.0:
        logits = logits / temperature
    
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Sample from the distribution
    next_tokens = torch.multinomial(probs, num_samples=1)
    return next_tokens

def top_k_sampling(logits: torch.Tensor, k: int = 50, temperature: float = 1.0) -> torch.Tensor:
    """
    Top-k sampling - restrict to the k most likely tokens, then sample.
    
    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        k: Number of top tokens to consider
        temperature: Sampling temperature
        
    Returns:
        Tensor of token indices with shape (batch_size, 1)
    """
    # Apply temperature scaling
    if temperature != 1.0:
        logits = logits / temperature
    
    # Get top-k logits and their indices
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
    
    # Convert top-k logits to probabilities
    top_k_probs = F.softmax(top_k_logits, dim=-1)
    
    # Sample from the top-k distribution
    next_token_positions = torch.multinomial(top_k_probs, num_samples=1)
    
    # Map positions back to token indices
    batch_size = logits.shape[0]
    batch_indices = torch.arange(batch_size, device=logits.device).unsqueeze(1)
    next_tokens = top_k_indices[batch_indices, next_token_positions]
    
    return next_tokens

def top_p_sampling(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
    """
    Top-p (nucleus) sampling - restrict to tokens with cumulative probability >= p.
    
    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        p: Probability threshold (0.0 to 1.0)
        temperature: Sampling temperature
        
    Returns:
        Tensor of token indices with shape (batch_size, 1)
    """
    # Apply temperature scaling
    if temperature != 1.0:
        logits = logits / temperature
    
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Remove tokens below the threshold (creates a mask where True = keep)
    nucleus_mask = cumulative_probs < p
    
    # Add the first token (highest prob) to ensure at least one token is selected
    nucleus_mask = torch.cat(
        [torch.ones_like(nucleus_mask[:, :1]), nucleus_mask[:, :-1]], dim=-1
    )
    
    # Filter tokens based on the mask
    filtered_indices = sorted_indices[nucleus_mask]
    filtered_probs = sorted_probs[nucleus_mask]
    
    # Reshape for batch processing
    batch_size = logits.shape[0]
    filtered_indices = filtered_indices.view(batch_size, -1)
    filtered_probs = filtered_probs.view(batch_size, -1)
    
    # Renormalize probabilities
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    
    # Sample from the filtered distribution
    token_positions = torch.multinomial(filtered_probs, num_samples=1)
    
    # Map positions back to token indices
    batch_indices = torch.arange(batch_size, device=logits.device).unsqueeze(1)
    next_tokens = filtered_indices[batch_indices, token_positions]
    
    return next_tokens

def combined_sampling(logits: torch.Tensor, 
                     temperature: float = 0.8,
                     top_k: Optional[int] = 50,
                     top_p: Optional[float] = 0.9) -> torch.Tensor:
    """
    Combined sampling using multiple techniques.
    
    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        temperature: Sampling temperature
        top_k: Number of top tokens to consider (None = disabled)
        top_p: Probability threshold (None = disabled)
        
    Returns:
        Tensor of token indices with shape (batch_size, 1)
    """
    # Apply temperature scaling
    if temperature != 1.0:
        logits = logits / temperature
    
    # Apply top-k filtering
    if top_k is not None:
        # Zero out all logits below top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
    
    # Apply top-p filtering
    if top_p is not None:
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Create a mask for tokens to remove
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift mask to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Apply mask to sorted indices
        batch_indices = torch.arange(logits.shape[0]).unsqueeze(-1).expand_as(sorted_indices)
        indices_to_remove = sorted_indices[batch_indices, sorted_indices_to_remove]
        
        # Create a mask for the original logits
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask.scatter_(-1, indices_to_remove, False)
        
        # Apply mask to logits
        logits = logits.masked_fill(~mask, float('-inf'))
    
    # Convert to probabilities and sample
    probs = F.softmax(logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1)
    
    return next_tokens

# Dictionary of available sampling functions
SAMPLING_STRATEGIES = {
    'greedy': greedy_sampling,
    'temperature': temperature_sampling,
    'top_k': top_k_sampling,
    'top_p': top_p_sampling,
    'combined': combined_sampling,
}

def get_sampling_fn(strategy: str) -> Callable:
    """
    Get a sampling function by name.
    
    Args:
        strategy: Name of the sampling strategy
        
    Returns:
        Sampling function
        
    Raises:
        ValueError: If strategy is not recognized
    """
    if strategy not in SAMPLING_STRATEGIES:
        available_strategies = list(SAMPLING_STRATEGIES.keys())
        raise ValueError(
            f"Unknown sampling strategy: {strategy}. "
            f"Available strategies: {available_strategies}"
        )
    
    return SAMPLING_STRATEGIES[strategy]
