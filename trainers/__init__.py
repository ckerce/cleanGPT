# ./trainers/__init__.py
"""
Trainers module for cleanGPT
Provides training loop implementations
"""

import logging
from typing import Dict, Type, Any

from .base_trainer import BaseTrainer
from .simple_trainer import SimpleTrainer

logger = logging.getLogger(__name__)

# Registry of available trainer types
TRAINER_REGISTRY: Dict[str, Type[BaseTrainer]] = {
    'simple': SimpleTrainer,
    # Add more trainer types as they're implemented
}

def get_trainer(trainer_type: str, **kwargs) -> BaseTrainer:
    """
    Factory function to get a trainer instance.
    
    Args:
        trainer_type: Type of trainer to use ('simple', etc.)
        **kwargs: Arguments for trainer initialization
        
    Returns:
        Initialized trainer instance
        
    Raises:
        ValueError: If trainer_type is not recognized
    """
    if trainer_type not in TRAINER_REGISTRY:
        available_trainers = list(TRAINER_REGISTRY.keys())
        raise ValueError(
            f"Unknown trainer type: {trainer_type}. "
            f"Available types: {available_trainers}"
        )
    
    trainer_class = TRAINER_REGISTRY[trainer_type]
    return trainer_class(**kwargs)

def register_trainer(name: str, trainer_class: Type[BaseTrainer]):
    """
    Register a new trainer type.
    
    Args:
        name: Name to register the trainer under
        trainer_class: The trainer class to register
        
    Raises:
        ValueError: If name is already registered or class doesn't
                   inherit from BaseTrainer
    """
    if name in TRAINER_REGISTRY:
        raise ValueError(f"Trainer type '{name}' is already registered")
    
    if not issubclass(trainer_class, BaseTrainer):
        raise ValueError("Trainer class must inherit from BaseTrainer")
    
    TRAINER_REGISTRY[name] = trainer_class
    logger.info(f"Registered new trainer type: {name}")

# Export main classes and functions
__all__ = [
    'BaseTrainer',
    'SimpleTrainer',
    'get_trainer',
    'register_trainer'
]
