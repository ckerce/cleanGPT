# ./utils/data_utils.py
"""
Data Loading and Preparation Utilities
Enhanced version with support for customizable tokenizers
"""

import logging
from typing import Dict, List, Union, Optional, Tuple, Any

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling

from mytokenizers import BaseTokenizer






logger = logging.getLogger(__name__)

class TokenizedDataset(Dataset):
    """
    Dataset wrapper for tokenized text data.
    
    This provides a PyTorch Dataset interface for tokenized text,
    supporting different tokenization strategies through our BaseTokenizer.
    """
    
    def __init__(self, 
                 tokenized_data: List[Dict[str, List[int]]],
                 block_size: int = 128):
        """
        Initialize the dataset with tokenized examples.
        
        Args:
            tokenized_data: List of dictionaries with 'input_ids' keys
            block_size: Maximum sequence length for input context
        """
        self.examples = tokenized_data
        self.block_size = block_size
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def load_and_prepare_data(dataset_name: str, 
                         dataset_config: Optional[str], 
                         tokenizer: BaseTokenizer,
                         max_samples: Optional[int] = None,
                         max_seq_length: int = 128,
                         batch_size: int = 32,
                         mlm: bool = False,
                         split: str = 'train',
                         shuffle: bool = True) -> Tuple[DataLoader, BaseTokenizer]:
    """
    Load a dataset, tokenize it, and prepare a DataLoader.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'wikipedia', 'wikitext')
        dataset_config: Dataset configuration name
        tokenizer: Tokenizer to use for text processing
        max_samples: Maximum number of samples to load
        max_seq_length: Maximum sequence length for tokenization
        batch_size: Batch size for the DataLoader
        mlm: Whether to use masked language modeling (True) or causal LM (False)
        split: Dataset split to use ('train', 'validation', 'test')
        shuffle: Whether to shuffle the data
        
    Returns:
        Tuple of (DataLoader, tokenizer)
    """
    # --- Load Dataset ---
    logger.info(f"Loading dataset: {dataset_name}" + 
               (f" ({dataset_config})" if dataset_config else ""))
    
    # Construct dataset split string with optional sample limit
    split_str = f"{split}[:{max_samples}]" if max_samples else split
    
    try:
        # Handle both with and without config
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, split=split_str, trust_remote_code=True)
        else:
            dataset = load_dataset(dataset_name, split=split_str, trust_remote_code=True)
            
        logger.info(f"Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
    
    # --- Preprocess Dataset ---
    logger.info("Preprocessing dataset...")
    
    # Define tokenization function
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None,  # Return Python lists for datasets processing
        )
        return outputs
    
    # Apply tokenization to the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    logger.info("Dataset tokenized.")
    
    # --- Create DataLoader ---
    # Create data collator for batching
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm,
        mlm_probability=0.15 if mlm else None,
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=shuffle,
    )
    
    logger.info(f"DataLoader created (batch_size={batch_size}, shuffle={shuffle})")
    
    return dataloader, tokenizer


def prepare_causal_lm_dataset(texts: List[str], 
                             tokenizer: BaseTokenizer,
                             block_size: int = 128,
                             batch_size: int = 32,
                             shuffle: bool = True) -> DataLoader:
    """
    Prepare a DataLoader from a list of text strings for causal language modeling.
    
    Args:
        texts: List of text samples
        tokenizer: Tokenizer to use for text processing
        block_size: Context window size for language modeling
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader for training
    """
    logger.info(f"Preparing dataset from {len(texts)} text samples...")
    
    # Tokenize all texts
    tokenized_data = []
    for text in texts:
        # Add special tokens for proper training
        encoded = tokenizer.encode(text, add_special_tokens=True)
        
        # Skip texts that are too short after tokenization
        if len(encoded) < 2:  # Need at least 2 tokens for input/target
            continue
            
        # Handle texts longer than block_size
        for i in range(0, len(encoded) - block_size + 1, block_size // 2):  # 50% overlap
            input_ids = encoded[i:i + block_size]
            if len(input_ids) == block_size:
                tokenized_data.append({'input_ids': input_ids})
    
    # Create a custom dataset
    dataset = TokenizedDataset(tokenized_data, block_size=block_size)
    
    # Create the DataCollator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal language modeling
    )
    
    # Create and return the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=shuffle,
    )
    
    logger.info(f"DataLoader created with {len(dataset)} examples")
    return dataloader
