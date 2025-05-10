# ./tokenizers/factory.py
"""
Tokenizer Factory
Creates and configures different tokenizer types in a unified way
"""

import logging
from typing import Dict, Any, Optional, Union

from .base_tokenizer import BaseTokenizer
from .character_tokenizer import CharacterTokenizer
from .gpt2_tokenizer import GPT2Tokenizer

# Import other tokenizer implementations as they're added

logger = logging.getLogger(__name__)

class TokenizerFactory:
    """
    Factory class for creating tokenizers with different strategies.
    
    This provides a central point for tokenizer creation and configuration,
    making it easy to switch between different tokenization approaches.
    """
    
    # Registry of available tokenizer types
    TOKENIZER_TYPES = {
        'character': CharacterTokenizer,
        'gpt2': GPT2Tokenizer,
        # Add more tokenizer types as they're implemented
    }
    
    @classmethod
    def create_tokenizer(cls, tokenizer_type: str, **kwargs) -> BaseTokenizer:
        """
        Create a tokenizer of the specified type.
        
        Args:
            tokenizer_type: Type of tokenizer to create ('character', 'gpt2', etc.)
            **kwargs: Additional arguments for tokenizer initialization
            
        Returns:
            Initialized tokenizer instance
            
        Raises:
            ValueError: If tokenizer_type is not recognized
        """
        if tokenizer_type not in cls.TOKENIZER_TYPES:
            raise ValueError(
                f"Unknown tokenizer type: {tokenizer_type}. "
                f"Available types: {list(cls.TOKENIZER_TYPES.keys())}"
            )
        
        logger.info(f"Creating tokenizer of type: {tokenizer_type}")
        tokenizer_class = cls.TOKENIZER_TYPES[tokenizer_type]
        return tokenizer_class(**kwargs)
    
    @classmethod
    def from_pretrained(cls, tokenizer_type: str, directory_or_name: str, **kwargs) -> BaseTokenizer:
        """
        Load a pre-trained tokenizer of the specified type.
        
        Args:
            tokenizer_type: Type of tokenizer to load ('character', 'gpt2', etc.)
            directory_or_name: Directory path or name of a predefined tokenizer
            **kwargs: Additional arguments for tokenizer initialization
            
        Returns:
            Loaded tokenizer instance
            
        Raises:
            ValueError: If tokenizer_type is not recognized
        """
        if tokenizer_type not in cls.TOKENIZER_TYPES:
            raise ValueError(
                f"Unknown tokenizer type: {tokenizer_type}. "
                f"Available types: {list(cls.TOKENIZER_TYPES.keys())}"
            )
        
        logger.info(f"Loading {tokenizer_type} tokenizer from: {directory_or_name}")
        tokenizer_class = cls.TOKENIZER_TYPES[tokenizer_type]
        return tokenizer_class.from_pretrained(directory_or_name, **kwargs)
    
    @classmethod
    def register_tokenizer(cls, name: str, tokenizer_class: type):
        """
        Register a new tokenizer type.
        
        Args:
            name: Name to register the tokenizer under
            tokenizer_class: The tokenizer class to register
            
        Raises:
            ValueError: If name is already registered or class doesn't inherit from BaseTokenizer
        """
        if name in cls.TOKENIZER_TYPES:
            raise ValueError(f"Tokenizer type '{name}' is already registered")
        
        if not issubclass(tokenizer_class, BaseTokenizer):
            raise ValueError(f"Tokenizer class must inherit from BaseTokenizer")
        
        cls.TOKENIZER_TYPES[name] = tokenizer_class
        logger.info(f"Registered new tokenizer type: {name}")
