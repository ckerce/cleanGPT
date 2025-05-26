# dataloaders/factory.py
"""
Factory functions for creating curriculum dataloaders.
"""

import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Optional, Union
from .base import CurriculumDataset
from .strategies import LinearTransitionStrategy, StepScheduleStrategy, ExponentialDecayStrategy


def get_strategy(strategy_name: str, strategy_params: Dict[str, Any]):
    """Factory function to create curriculum strategies."""
    strategies = {
        'linear_transition': LinearTransitionStrategy,
        'step_schedule': StepScheduleStrategy,
        'exponential_decay': ExponentialDecayStrategy,
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
    
    return strategies[strategy_name](**strategy_params)


def collate_curriculum_batch(batch):
    """
    Collate function for curriculum learning batches.
    Handles variable length sequences and includes metadata.
    """
    # Extract input_ids and other data
    input_ids = [item['input_ids'] for item in batch]
    
    # Pad sequences
    max_len = max(len(ids) for ids in input_ids)
    padded_ids = []
    attention_masks = []
    
    for ids in input_ids:
        padding_length = max_len - len(ids)
        padded_ids.append(ids + [0] * padding_length)  # Assuming 0 is pad token
        attention_masks.append([1] * len(ids) + [0] * padding_length)
    
    # Create batch dict
    batch_dict = {
        'input_ids': torch.tensor(padded_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
        'dataset_sources': [item.get('dataset_source', 'unknown') for item in batch],
        'original_lengths': [item.get('original_length', 0) for item in batch]
    }
    
    return batch_dict


def get_curriculum_dataloader(
    datasets: List[Any],
    dataset_configs: List[Dict[str, Any]],
    strategy: Union[str, Dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int = 512,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_buffer_size: int = 10000,
    seed: Optional[int] = None,
    collate_fn: Optional[callable] = None
) -> DataLoader:
    """
    Factory function to create a curriculum learning DataLoader.
    
    Args:
        datasets: List of datasets to blend
        dataset_configs: Configuration for each dataset
        strategy: Either strategy name (str) or dict with 'name' and 'params'
        tokenizer: Tokenizer instance
        max_seq_length: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle_buffer_size: Buffer size for shuffling
        seed: Random seed
        collate_fn: Custom collate function (defaults to collate_curriculum_batch)
    
    Returns:
        DataLoader configured for curriculum learning
    """
    
    # Handle strategy configuration
    if isinstance(strategy, str):
        # Default parameters for common strategies
        default_params = {
            'linear_transition': {
                'start_weights': [1.0 / len(datasets)] * len(datasets),
                'end_weights': [1.0 / len(datasets)] * len(datasets),
                'transition_epochs': 5
            }
        }
        strategy_name = strategy
        strategy_params = default_params.get(strategy_name, {})
    else:
        strategy_name = strategy['name']
        strategy_params = strategy.get('params', {})
    
    # Create strategy instance
    strategy_instance = get_strategy(strategy_name, strategy_params)
    
    # Create curriculum dataset
    curriculum_dataset = CurriculumDataset(
        datasets=datasets,
        dataset_configs=dataset_configs,
        strategy=strategy_instance,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed
    )
    
    # Use default collate function if none provided
    if collate_fn is None:
        collate_fn = collate_curriculum_batch
    
    # Create DataLoader
    dataloader = DataLoader(
        curriculum_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    # Add convenience method to update epoch
    def update_epoch(epoch):
        curriculum_dataset.update_for_epoch(epoch)
    
    dataloader.update_epoch = update_epoch
    dataloader.curriculum_dataset = curriculum_dataset
    
    return dataloader

