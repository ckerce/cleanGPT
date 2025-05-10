# ./tokenizers/base_tokenizer.py
"""
Base Tokenizer Abstract Class
Defines the interface that all tokenizers must implement
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional, Any
import torch


class BaseTokenizer(ABC):
    """
    Abstract base class for all tokenizers in the cleanGPT project.
    
    This class defines the interface that all tokenizer implementations
    must adhere to, ensuring consistency across different tokenization strategies.
    """
    
    def __init__(self, vocab_size: int = None, padding_token: str = None, 
                 eos_token: str = None, bos_token: str = None):
        """
        Initialize the base tokenizer.
        
        Args:
            vocab_size (int, optional): Size of the vocabulary
            padding_token (str, optional): Token used for padding
            eos_token (str, optional): End of sequence token
            bos_token (str, optional): Beginning of sequence token
        """
        self.vocab_size = vocab_size
        self._padding_token = padding_token
        self._eos_token = eos_token
        self._bos_token = bos_token
        
        # These will be set by implementations
        self.pad_token_id = None
        self.eos_token_id = None
        self.bos_token_id = None
        self.model_max_length = 1024  # Default, can be overridden
        
    @property
    def pad_token(self) -> Optional[str]:
        """Get the padding token."""
        return self._padding_token
    
    @pad_token.setter
    def pad_token(self, token: str):
        """Set the padding token and its ID."""
        self._padding_token = token
        if hasattr(self, 'convert_tokens_to_ids'):
            self.pad_token_id = self.convert_token_to_id(token)
    
    @property
    def eos_token(self) -> Optional[str]:
        """Get the end of sequence token."""
        return self._eos_token
    
    @eos_token.setter
    def eos_token(self, token: str):
        """Set the EOS token and its ID."""
        self._eos_token = token
        if hasattr(self, 'convert_tokens_to_ids'):
            self.eos_token_id = self.convert_token_to_id(token)
    
    @property
    def bos_token(self) -> Optional[str]:
        """Get the beginning of sequence token."""
        return self._bos_token
    
    @bos_token.setter
    def bos_token(self, token: str):
        """Set the BOS token and its ID."""
        self._bos_token = token
        if hasattr(self, 'convert_tokens_to_ids'):
            self.bos_token_id = self.convert_token_to_id(token)
    
    @abstractmethod
    def encode(self, text: Union[str, List[str]], **kwargs) -> Union[List[int], List[List[int]]]:
        """
        Encode text into token IDs.
        
        Args:
            text: Text to encode, can be a string or list of strings
            **kwargs: Additional arguments for specific tokenizer implementations
            
        Returns:
            List of token IDs or list of lists for batch encoding
        """
        pass
    
    @abstractmethod
    def decode(self, token_ids: Union[List[int], torch.Tensor], **kwargs) -> str:
        """
        Decode token IDs back into text.
        
        Args:
            token_ids: List of token IDs or PyTorch tensor
            **kwargs: Additional arguments like skip_special_tokens
            
        Returns:
            Decoded text string
        """
        pass
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Split text into tokens without converting to IDs.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token strings
        """
        pass
    
    @abstractmethod
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Convert token(s) to their corresponding IDs.
        
        Args:
            tokens: Token string or list of token strings
            
        Returns:
            Token ID or list of token IDs
        """
        pass
    
    def convert_token_to_id(self, token: str) -> int:
        """
        Convert a single token to its ID.
        
        Args:
            token: Single token string
            
        Returns:
            Token ID
        """
        return self.convert_tokens_to_ids([token])[0]
    
    @abstractmethod
    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """
        Convert token ID(s) to their corresponding token strings.
        
        Args:
            ids: Token ID or list of token IDs
            
        Returns:
            Token string or list of token strings
        """
        pass
    
    def __call__(self, text: Union[str, List[str]], padding: bool = False, 
                truncation: bool = False, max_length: int = None, 
                return_tensors: str = None, **kwargs) -> Dict[str, Any]:
        """
        Process text input and prepare for model.
        
        Args:
            text: Text to tokenize, can be a string or list of strings
            padding: Whether to pad sequences to max_length
            truncation: Whether to truncate sequences to max_length
            max_length: Maximum sequence length
            return_tensors: Format of returned tensors ('pt' for PyTorch)
            **kwargs: Additional arguments for specific implementations
            
        Returns:
            Dictionary with input_ids and potentially attention_mask
        """
        # This provides a default implementation that subclasses can override
        
        # Encode the text to token IDs
        if isinstance(text, str):
            input_ids = self.encode(text)
            batch_size = 1
            input_ids = [input_ids]
        else:
            input_ids = [self.encode(t) for t in text]
            batch_size = len(input_ids)
        
        # Handle truncation
        if truncation and max_length:
            input_ids = [ids[:max_length] for ids in input_ids]
        
        # Find max length for padding
        if padding:
            max_len = max([len(ids) for ids in input_ids]) if max_length is None else max_length
            attention_mask = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in input_ids]
            
            # Pad sequences
            padded_input_ids = []
            for ids in input_ids:
                if len(ids) < max_len:
                    padded_input_ids.append(ids + [self.pad_token_id] * (max_len - len(ids)))
                else:
                    padded_input_ids.append(ids)
            input_ids = padded_input_ids
        else:
            attention_mask = [[1] * len(ids) for ids in input_ids]
        
        # Convert to tensors if requested
        if return_tensors == 'pt':
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def get_vocab(self) -> Dict[str, int]:
        """
        Get the vocabulary mapping.
        
        Returns:
            Dictionary mapping token strings to token IDs
        """
        # This should be implemented by subclasses
        raise NotImplementedError("Subclass must implement get_vocab()")
    
    def save_pretrained(self, directory: str):
        """
        Save tokenizer configuration to a directory.
        
        Args:
            directory: Directory path to save tokenizer files
        """
        # This should be implemented by subclasses
        raise NotImplementedError("Subclass must implement save_pretrained()")
    
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
        # This should be implemented by subclasses
        raise NotImplementedError("Subclass must implement from_pretrained()")
