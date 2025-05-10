# ./tokenizers/character_tokenizer.py
"""
Character-Level Tokenizer Implementation
"""

import os
import json
from typing import List, Dict, Union, Optional, Any
import torch
import logging

from .base_tokenizer import BaseTokenizer

logger = logging.getLogger(__name__)

class CharacterTokenizer(BaseTokenizer):
    """
    Character-level tokenizer that splits text into individual characters.
    
    This tokenizer is simple but effective for certain tasks and languages.
    It creates a vocabulary of unique characters in the training data.
    """
    
    def __init__(self, 
                 vocab: Optional[Dict[str, int]] = None, 
                 unk_token: str = "<unk>",
                 pad_token: str = "<pad>",
                 bos_token: str = "<bos>",
                 eos_token: str = "<eos>",
                 additional_special_tokens: List[str] = None,
                 max_length: int = 1024):
        """
        Initialize the character tokenizer.
        
        Args:
            vocab: Optional predefined vocabulary mapping characters to IDs
            unk_token: Token for unknown characters
            pad_token: Token used for padding
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
            additional_special_tokens: List of additional special tokens
            max_length: Maximum sequence length
        """
        # Initialize base class
        super().__init__(
            padding_token=pad_token,
            eos_token=eos_token,
            bos_token=bos_token
        )
        
        # Set up special tokens
        self.unk_token = unk_token
        self.additional_special_tokens = additional_special_tokens or []
        self.all_special_tokens = [unk_token, pad_token, bos_token, eos_token] + self.additional_special_tokens
        
        # Set up vocabulary
        if vocab:
            self.vocab = vocab
        else:
            # If no vocab is provided, start with special tokens
            self.vocab = {token: i for i, token in enumerate(self.all_special_tokens)}
            
        # Set up reverse vocabulary (id to token)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Set token IDs for convenience
        self.unk_token_id = self.vocab.get(unk_token, 0)
        self.pad_token_id = self.vocab.get(pad_token, 1)
        self.bos_token_id = self.vocab.get(bos_token, 2)
        self.eos_token_id = self.vocab.get(eos_token, 3)
        
        # Set model max length
        self.model_max_length = max_length
        self.vocab_size = len(self.vocab)
        
    def build_vocab_from_texts(self, texts: List[str]):
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts: List of text strings to extract characters from
        """
        # Start with special tokens
        vocab = {token: i for i, token in enumerate(self.all_special_tokens)}
        next_id = len(vocab)
        
        # Add all unique characters from texts
        for text in texts:
            for char in text:
                if char not in vocab:
                    vocab[char] = next_id
                    next_id += 1
        
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        logger.info(f"Built vocabulary with {self.vocab_size} tokens")
        return self.vocab
    
    def tokenize(self, text: str) -> List[str]:
        """
        Split text into individual characters.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of characters (tokens)
        """
        return list(text)
    
    def encode(self, text: Union[str, List[str]], add_special_tokens: bool = False, **kwargs) -> Union[List[int], List[List[int]]]:
        """
        Encode text into token IDs.
        
        Args:
            text: Text to encode, can be a string or list of strings
            add_special_tokens: Whether to add BOS/EOS tokens
            **kwargs: Additional arguments
            
        Returns:
            List of token IDs or list of lists for batch encoding
        """
        if isinstance(text, list):
            return [self.encode(t, add_special_tokens, **kwargs) for t in text]
        
        tokens = self.tokenize(text)
        ids = self.convert_tokens_to_ids(tokens)
        
        # Add special tokens if requested
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
            
        return ids
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = False, **kwargs) -> str:
        """
        Decode token IDs back into text.
        
        Args:
            token_ids: List of token IDs or PyTorch tensor
            skip_special_tokens: Whether to skip special tokens in output
            **kwargs: Additional arguments
            
        Returns:
            Decoded text string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        tokens = self.convert_ids_to_tokens(token_ids)
        
        # Filter out special tokens if requested
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.all_special_tokens]
            
        # Join characters back into a string
        return ''.join(tokens)
    
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Convert token(s) to their corresponding IDs.
        
        Args:
            tokens: Token string or list of token strings
            
        Returns:
            Token ID or list of token IDs
        """
        if isinstance(tokens, str):
            return self.vocab.get(tokens, self.unk_token_id)
        else:
            return [self.vocab.get(token, self.unk_token_id) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """
        Convert token ID(s) to their corresponding token strings.
        
        Args:
            ids: Token ID or list of token IDs
            
        Returns:
            Token string or list of token strings
        """
        if isinstance(ids, int):
            return self.id_to_token.get(ids, self.unk_token)
        else:
            return [self.id_to_token.get(id_, self.unk_token) for id_ in ids]
    
    def get_vocab(self) -> Dict[str, int]:
        """
        Get the vocabulary mapping.
        
        Returns:
            Dictionary mapping token strings to token IDs
        """
        return self.vocab
    
    def save_pretrained(self, directory: str):
        """
        Save tokenizer configuration to a directory.
        
        Args:
            directory: Directory path to save tokenizer files
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save the vocabulary
        vocab_file = os.path.join(directory, "vocab.json")
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        # Save the tokenizer configuration
        config = {
            'tokenizer_type': 'character',
            'vocab_size': len(self.vocab),
            'unk_token': self.unk_token,
            'pad_token': self.pad_token,
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
            'additional_special_tokens': self.additional_special_tokens,
            'model_max_length': self.model_max_length,
        }
        
        config_file = os.path.join(directory, "tokenizer_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Tokenizer saved to {directory}")
    
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
        # Currently only supports loading from directory
        if os.path.isdir(directory_or_name):
            # Load vocab
            vocab_file = os.path.join(directory_or_name, "vocab.json")
            if not os.path.exists(vocab_file):
                raise FileNotFoundError(f"Vocabulary file not found at {vocab_file}")
                
            with open(vocab_file, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            
            # Load config
            config_file = os.path.join(directory_or_name, "tokenizer_config.json")
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # Update kwargs with config
                for k, v in config.items():
                    if k not in kwargs:
                        kwargs[k] = v
            
            # Initialize with loaded vocab and config
            return cls(vocab=vocab, **kwargs)
        else:
            raise ValueError(f"Directory not found: {directory_or_name}")
