# ./tokenizers/gpt2_tokenizer.py
"""
GPT-2 Tokenizer Wrapper
Provides a consistent interface to the Hugging Face GPT-2 tokenizer
"""

import os
from typing import List, Dict, Union, Optional, Any
import torch
import logging
from transformers import GPT2Tokenizer as HfGPT2Tokenizer, GPT2TokenizerFast

from .base_tokenizer import BaseTokenizer

logger = logging.getLogger(__name__)

class GPT2Tokenizer(BaseTokenizer):
    """
    Wrapper for the Hugging Face GPT-2 tokenizer to match our interface.
    
    This provides a consistent API for the GPT-2 tokenizer while leveraging
    the efficient implementation from Hugging Face.
    """
    
    def __init__(self, 
                 use_fast: bool = True,
                 padding_token: str = None,
                 max_length: int = 1024, 
                 **kwargs):
        """
        Initialize the GPT-2 tokenizer wrapper.
        
        Args:
            use_fast: Whether to use the fast tokenizer implementation
            padding_token: Custom padding token (default uses EOS token for padding)
            max_length: Maximum sequence length
            **kwargs: Additional arguments passed to the Hugging Face tokenizer
        """
        # Initialize with the underlying HF tokenizer
        tokenizer_class = GPT2TokenizerFast if use_fast else HfGPT2Tokenizer
        self._tokenizer = tokenizer_class.from_pretrained('gpt2', **kwargs)
        
        # Initialize base class after getting vocab size from HF tokenizer
        super().__init__(
            vocab_size=len(self._tokenizer),
            padding_token=padding_token or self._tokenizer.eos_token,
            eos_token=self._tokenizer.eos_token,
            bos_token=self._tokenizer.bos_token,
        )
        
        # Set the padding token if specified (GPT-2 uses EOS as padding by default)
        if padding_token:
            self._tokenizer.pad_token = padding_token
        else:
            # If no padding token specified, use EOS token as padding (common practice)
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Set model max length
        self.model_max_length = self._tokenizer.model_max_length
        if max_length:
            self._tokenizer.model_max_length = max_length
            self.model_max_length = max_length
        
        # Set token IDs from the underlying tokenizer
        self.pad_token_id = self._tokenizer.pad_token_id
        self.eos_token_id = self._tokenizer.eos_token_id
        self.bos_token_id = self._tokenizer.bos_token_id
        
        logger.info(f"Initialized GPT-2 tokenizer with vocab size: {len(self._tokenizer)}")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Split text into tokens without converting to IDs.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token strings
        """
        return self._tokenizer.tokenize(text)
    
    def encode(self, text: Union[str, List[str]], add_special_tokens: bool = True, **kwargs) -> Union[List[int], List[List[int]]]:
        """
        Encode text into token IDs.
        
        Args:
            text: Text to encode, can be a string or list of strings
            add_special_tokens: Whether to add special tokens like BOS/EOS
            **kwargs: Additional arguments passed to the Hugging Face tokenizer
            
        Returns:
            List of token IDs or list of lists for batch encoding
        """
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens, **kwargs)
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = False, **kwargs) -> str:
        """
        Decode token IDs back into text.
        
        Args:
            token_ids: List of token IDs or PyTorch tensor
            skip_special_tokens: Whether to skip special tokens in output
            **kwargs: Additional arguments passed to the Hugging Face tokenizer
            
        Returns:
            Decoded text string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
    
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Convert token(s) to their corresponding IDs.
        
        Args:
            tokens: Token string or list of token strings
            
        Returns:
            Token ID or list of token IDs
        """
        return self._tokenizer.convert_tokens_to_ids(tokens)
    
    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """
        Convert token ID(s) to their corresponding token strings.
        
        Args:
            ids: Token ID or list of token IDs
            
        Returns:
            Token string or list of token strings
        """
        return self._tokenizer.convert_ids_to_tokens(ids)
    
    def __call__(self, text: Union[str, List[str]], padding: bool = False, 
                truncation: bool = False, max_length: int = None, 
                return_tensors: str = None, **kwargs) -> Dict[str, Any]:
        """
        Process text input and prepare for model.
        
        Args:
            text: Text to tokenize
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            return_tensors: Output tensor format ('pt' for PyTorch)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Delegate to the Hugging Face implementation which is well-optimized
        return self._tokenizer(
            text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            **kwargs
        )
    
    def get_vocab(self) -> Dict[str, int]:
        """
        Get the vocabulary mapping.
        
        Returns:
            Dictionary mapping token strings to token IDs
        """
        return self._tokenizer.get_vocab()
    
    def save_pretrained(self, directory: str):
        """
        Save tokenizer configuration to a directory.
        
        Args:
            directory: Directory path to save tokenizer files
        """
        self._tokenizer.save_pretrained(directory)
        logger.info(f"GPT-2 tokenizer saved to {directory}")
    
    @classmethod
    def from_pretrained(cls, directory_or_name: str, **kwargs):
        """
        Load a tokenizer from a directory or a predefined name.
        
        Args:
            directory_or_name: Directory path or name of a predefined tokenizer
            **kwargs: Additional arguments for tokenizer initialization
            
        Returns:
            Initialized tokenizer instance
        """
        use_fast = kwargs.pop('use_fast', True)
        
        if os.path.isdir(directory_or_name):
            # Load the underlying HF tokenizer from the directory
            tokenizer_class = GPT2TokenizerFast if use_fast else HfGPT2Tokenizer
            hf_tokenizer = tokenizer_class.from_pretrained(directory_or_name, **kwargs)
            
            # Initialize our wrapper with the loaded tokenizer
            instance = cls(use_fast=use_fast, **kwargs)
            instance._tokenizer = hf_tokenizer
            
            # Update attributes
            instance.pad_token_id = hf_tokenizer.pad_token_id
            instance.eos_token_id = hf_tokenizer.eos_token_id
            instance.bos_token_id = hf_tokenizer.bos_token_id
            instance.model_max_length = hf_tokenizer.model_max_length
            
            return instance
        else:
            # Load from predefined name (e.g., 'gpt2', 'gpt2-medium')
            return cls(use_fast=use_fast, **kwargs)
