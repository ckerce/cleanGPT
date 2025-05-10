"""
Tokenizers module for cleanGPT
Provides tokenization for various transformer models
"""

from .base_tokenizer import BaseTokenizer
from .character_tokenizer import CharacterTokenizer
from .gpt2_tokenizer import GPT2Tokenizer
from .factory import TokenizerFactory

# Convenience function for creating tokenizers
def create_tokenizer(tokenizer_type, **kwargs):
    """
    Create a tokenizer of the specified type.
    
    Args:
        tokenizer_type: Type of tokenizer ('character', 'gpt2', etc.)
        **kwargs: Additional arguments for tokenizer initialization
        
    Returns:
        Initialized tokenizer instance
    """
    return TokenizerFactory.create_tokenizer(tokenizer_type, **kwargs)

# Convenience function for loading pre-trained tokenizers
def from_pretrained(tokenizer_type, directory_or_name, **kwargs):
    """
    Load a pre-trained tokenizer.
    
    Args:
        tokenizer_type: Type of tokenizer ('character', 'gpt2', etc.)
        directory_or_name: Directory or name of pre-trained tokenizer
        **kwargs: Additional arguments for tokenizer initialization
        
    Returns:
        Loaded tokenizer instance
    """
    return TokenizerFactory.from_pretrained(tokenizer_type, directory_or_name, **kwargs)

# Export the main classes and functions
__all__ = [
    'BaseTokenizer',
    'CharacterTokenizer', 
    'GPT2Tokenizer',
    'TokenizerFactory',
    'create_tokenizer',
    'from_pretrained'
]
