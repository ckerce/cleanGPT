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

from mytokenizers import BaseTokenizer # Assuming this can import your GPT2Tokenizer

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
        # The DataCollatorForLanguageModeling expects a list of dicts, 
        # where each dict has 'input_ids'.
        # Our TokenizedDataset already stores data in this format if 
        # tokenized_data is [{ 'input_ids': [...] }, ...]
        return self.examples[idx]


def load_and_prepare_data(dataset_name: str, 
                         dataset_config: Optional[str], 
                         tokenizer: BaseTokenizer, # This is our wrapper
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
        tokenizer: Your BaseTokenizer wrapper instance.
        max_samples: Maximum number of samples to load
        max_seq_length: Maximum sequence length for tokenization
        batch_size: Batch size for the DataLoader
        mlm: Whether to use masked language modeling (True) or causal LM (False)
        split: Dataset split to use ('train', 'validation', 'test')
        shuffle: Whether to shuffle the data
        
    Returns:
        Tuple of (DataLoader, your BaseTokenizer wrapper instance)
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
        logger.error(f"Error loading dataset: {e}", exc_info=True)
        raise
    
    # --- Preprocess Dataset ---
    logger.info("Preprocessing dataset...")
    
    # Define tokenization function
    def tokenize_function(examples):
        text_field = 'text' 
        if 'text' not in examples and 'story' in examples:
            text_field = 'story'
        elif 'text' not in examples: 
            found_field = None
            for col_name, feat in examples.items(): 
                if isinstance(feat, list) and feat and isinstance(feat[0], str):
                    found_field = col_name
                    break
            if found_field:
                text_field = found_field
            else:
                logger.warning(f"Could not auto-detect text field in examples. Using 'text'. Available keys: {list(examples.keys())}")

        # Use the __call__ method of our tokenizer wrapper
        outputs = tokenizer(
            examples[text_field], 
            truncation=True,
            max_length=max_seq_length,
            padding=False, 
            return_tensors=None, 
        )
        return outputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names, 
        desc="Tokenizing dataset"
    )
    
    logger.info("Dataset tokenized.")
    
    # --- Create DataLoader ---
    # Determine the actual tokenizer to pass to the DataCollator
    # If our tokenizer wrapper has an underlying '_tokenizer' (like GPT2Tokenizer), use that.
    # Otherwise, use the wrapper itself (e.g., for CharacterTokenizer).
    tokenizer_for_collator = getattr(tokenizer, '_tokenizer', tokenizer)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer_for_collator, # Pass the effective tokenizer
        mlm=mlm,
        mlm_probability=0.15 if mlm else 0.0,
    )
    
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=shuffle,
    )
    
    logger.info(f"DataLoader created (batch_size={batch_size}, shuffle={shuffle})")
    
    return dataloader, tokenizer # Return the original wrapper


def prepare_causal_lm_dataset(texts: List[str], 
                             tokenizer: BaseTokenizer, # This is our wrapper
                             block_size: int = 128,
                             batch_size: int = 32,
                             shuffle: bool = True) -> DataLoader:
    """
    Prepare a DataLoader from a list of text strings for causal language modeling.
    
    Args:
        texts: List of text samples
        tokenizer: Your BaseTokenizer wrapper instance.
        block_size: Context window size for language modeling
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader for training
    """
    logger.info(f"Preparing dataset from {len(texts)} text samples...")
    
    tokenized_data = []
    for text in texts:
        # Encode using our tokenizer wrapper's encode method
        encoded = tokenizer.encode(text, add_special_tokens=True) 
        
        if len(encoded) < 2: 
            continue
            
        for i in range(0, len(encoded) - 1, block_size): 
            input_ids_chunk = encoded[i : i + block_size]
            if len(input_ids_chunk) > 1 : 
                 tokenized_data.append({'input_ids': input_ids_chunk})
    
    if not tokenized_data:
        logger.warning("No suitable examples found after tokenization and chunking. DataLoader will be empty.")

    dataset = TokenizedDataset(tokenized_data, block_size=block_size)
    
    # Determine the actual tokenizer to pass to the DataCollator
    tokenizer_for_collator = getattr(tokenizer, '_tokenizer', tokenizer)
        
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer_for_collator, # Pass the effective tokenizer
        mlm=False, 
        mlm_probability=0.0 
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=shuffle,
    )
    
    logger.info(f"DataLoader created with {len(dataset)} examples")
    return dataloader

