# dataloaders/__init__.py
"""
Curriculum DataLoader Module

This module provides flexible curriculum learning strategies for blending multiple datasets
during training. It supports various scheduling strategies and is designed to be easily
extensible.

Usage:
    from dataloaders import get_curriculum_dataloader
    
    dataloader = get_curriculum_dataloader(
        datasets=[wiki_dataset, code_dataset],
        dataset_configs=[
            {'name': 'wikipedia', 'text_field': 'text', 'weight': 0.7},
            {'name': 'code', 'text_field': 'func_code_string', 'weight': 0.3}
        ],
        strategy='linear_transition',
        strategy_params={'start_weights': [0.7, 0.3], 'end_weights': [0.9, 0.1], 'transition_epochs': 5},
        tokenizer=tokenizer,
        max_seq_length=512,
        batch_size=32
    )
"""

from .base import CurriculumDataset, CurriculumStrategy
from .strategies import LinearTransitionStrategy, StepScheduleStrategy, ExponentialDecayStrategy
from .factory import get_curriculum_dataloader, get_strategy

__all__ = [
    'CurriculumDataset',
    'CurriculumStrategy', 
    'LinearTransitionStrategy',
    'StepScheduleStrategy',
    'ExponentialDecayStrategy',
    'get_curriculum_dataloader',
    'get_strategy'
]

