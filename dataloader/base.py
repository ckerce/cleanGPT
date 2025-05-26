# dataloaders/base.py
"""
Base classes for curriculum learning strategies.
"""

import torch
from torch.utils.data import IterableDataset
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator
import random
import logging

logger = logging.getLogger(__name__)


class CurriculumStrategy(ABC):
    """Base class for curriculum learning strategies."""
    
    def __init__(self, num_datasets: int, **kwargs):
        self.num_datasets = num_datasets
        self.current_epoch = 0
        
    @abstractmethod
    def get_weights(self, epoch: int) -> List[float]:
        """Return the weights for each dataset at the given epoch."""
        pass
    
    def update_epoch(self, epoch: int):
        """Update the current epoch."""
        self.current_epoch = epoch
        weights = self.get_weights(epoch)
        logger.info(f"Epoch {epoch}: Dataset weights = {[f'{w:.3f}' for w in weights]}")
        return weights


class CurriculumDataset(IterableDataset):
    """
    A flexible curriculum learning dataset that blends multiple datasets
    according to a specified strategy.
    """
    
    def __init__(
        self,
        datasets: List[Any],
        dataset_configs: List[Dict[str, Any]],
        strategy: CurriculumStrategy,
        tokenizer: Any,
        max_seq_length: int = 512,
        shuffle_buffer_size: int = 10000,
        seed: Optional[int] = None
    ):
        """
        Args:
            datasets: List of dataset objects
            dataset_configs: List of config dicts with keys like 'name', 'text_field', etc.
            strategy: CurriculumStrategy instance
            tokenizer: Tokenizer instance
            max_seq_length: Maximum sequence length for tokenization
            shuffle_buffer_size: Buffer size for shuffling (if applicable)
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        if len(datasets) != len(dataset_configs):
            raise ValueError("Number of datasets must match number of configs")
        
        self.datasets = datasets
        self.dataset_configs = dataset_configs
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.shuffle_buffer_size = shuffle_buffer_size
        
        if seed is not None:
            random.seed(seed)
            
        # Initialize dataset iterators
        self.dataset_iterators = []
        self._setup_iterators()
        
        # Current weights
        self.current_weights = strategy.get_weights(0)
        
    def _setup_iterators(self):
        """Initialize iterators for each dataset."""
        self.dataset_iterators = []
        
        for i, (dataset, config) in enumerate(zip(self.datasets, self.dataset_configs)):
            iterator = self._create_dataset_iterator(dataset, config, i)
            self.dataset_iterators.append(iterator)
    
    def _create_dataset_iterator(self, dataset, config: Dict[str, Any], dataset_idx: int) -> Iterator:
        """Create an iterator for a single dataset."""
        text_field = config.get('text_field', 'text')
        dataset_name = config.get('name', f'dataset_{dataset_idx}')
        
        def dataset_iter():
            # Handle different dataset types
            if hasattr(dataset, '__iter__'):
                # Regular dataset or list
                dataset_items = list(dataset) if not isinstance(dataset, list) else dataset
                
                # Shuffle if it's not a streaming dataset
                if config.get('shuffle', True):
                    random.shuffle(dataset_items)
                
                # Cycle through the dataset
                while True:
                    for item in dataset_items:
                        tokenized = self._tokenize_item(item, text_field, dataset_name)
                        if tokenized is not None:
                            yield tokenized
            else:
                # Streaming dataset
                for item in dataset:
                    tokenized = self._tokenize_item(item, text_field, dataset_name)
                    if tokenized is not None:
                        yield tokenized
        
        return dataset_iter()
    
    def _tokenize_item(self, item: Dict[str, Any], text_field: str, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Tokenize a single item from a dataset."""
        try:
            if text_field not in item or not item[text_field]:
                return None
                
            # Tokenize the text
            tokenized = self.tokenizer(
                item[text_field],
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
                return_tensors=None  # Return lists, not tensors
            )
            
            # Add metadata
            tokenized['dataset_source'] = dataset_name
            tokenized['original_length'] = len(item[text_field])
            
            return tokenized
            
        except Exception as e:
            logger.warning(f"Failed to tokenize item from {dataset_name}: {e}")
            return None
    
    def update_for_epoch(self, epoch: int):
        """Update the dataset for a new epoch."""
        self.current_weights = self.strategy.update_epoch(epoch)
        
        # Optionally reset iterators (depends on your needs)
        if epoch > 0 and hasattr(self, '_reset_on_epoch') and self._reset_on_epoch:
            self._setup_iterators()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """Sample from datasets according to current weights."""
        if not self.dataset_iterators:
            raise StopIteration
        
        # Choose dataset based on weights
        dataset_idx = random.choices(
            range(len(self.dataset_iterators)), 
            weights=self.current_weights
        )[0]
        
        try:
            return next(self.dataset_iterators[dataset_idx])
        except StopIteration:
            # If one dataset is exhausted, we might want to handle this differently
            # For now, we'll raise StopIteration to end the epoch
            logger.info(f"Dataset {dataset_idx} ({self.dataset_configs[dataset_idx]['name']}) exhausted")
            raise StopIteration


