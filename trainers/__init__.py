# ./trainers/__init__.py
"""
Trainers module for cleanGPT.

This module provides the necessary classes and factory functions for
creating and managing different training loop implementations (trainers)
and their associated callbacks.
"""

import torch
import logging
from typing import Dict, Type, Any, List, Optional # Added List and Optional

# Import base classes first
from .base_trainer import BaseTrainer, Callback

# Import concrete trainer implementations
from .simple_trainer import SimpleTrainer
# Example: from .advanced_trainer import AdvancedTrainer # If you add more

logger = logging.getLogger(__name__)

# Registry of available trainer types
# This dictionary maps a string identifier (e.g., 'simple') to the
# corresponding trainer class.
TRAINER_REGISTRY: Dict[str, Type[BaseTrainer]] = {
    'simple': SimpleTrainer,
    # 'advanced': AdvancedTrainer, # Add more trainers here as they are implemented
}

def get_trainer(trainer_type: str,
                model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                callbacks: Optional[List[Callback]] = None,
                **kwargs) -> BaseTrainer:
    """
    Factory function to get an initialized trainer instance.

    This function looks up the trainer_type in the TRAINER_REGISTRY
    and instantiates it with the provided arguments.

    Args:
        trainer_type (str): The type of trainer to use (e.g., 'simple').
                            Must be a key in TRAINER_REGISTRY.
        model (torch.nn.Module): The model to be trained.
        dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        device (torch.device): The device (CPU or GPU) to train on.
        callbacks (Optional[List[Callback]]): A list of callback instances to
                                              be used during training. Defaults to None.
        **kwargs: Additional keyword arguments specific to the chosen trainer type.
                  These will be passed directly to the trainer's constructor.
                  Common arguments might include 'num_epochs', 'output_dir', etc.

    Returns:
        An initialized instance of the requested trainer class.

    Raises:
        ValueError: If the trainer_type is not recognized (i.e., not in TRAINER_REGISTRY).
    """
    if trainer_type not in TRAINER_REGISTRY:
        available_trainers = list(TRAINER_REGISTRY.keys())
        raise ValueError(
            f"Unknown trainer type: '{trainer_type}'. "
            f"Available types: {available_trainers}"
        )

    trainer_class = TRAINER_REGISTRY[trainer_type]
    logger.info(f"Initializing trainer of type: {trainer_type}")

    # Pass common arguments and any trainer-specific kwargs
    return trainer_class(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        callbacks=callbacks, # Pass the callbacks list
        **kwargs # Pass along other trainer-specific arguments like num_epochs, output_dir
    )

def register_trainer(name: str, trainer_class: Type[BaseTrainer]):
    """
    Register a new trainer type in the TRAINER_REGISTRY.

    This allows for extending the framework with custom trainer implementations
    that can then be accessed via get_trainer using the provided name.

    Args:
        name (str): The name (identifier) to register the trainer under.
        trainer_class (Type[BaseTrainer]): The trainer class to register.
                                           Must be a subclass of BaseTrainer.

    Raises:
        ValueError: If the name is already registered or if the
                    trainer_class is not a subclass of BaseTrainer.
    """
    if name in TRAINER_REGISTRY:
        raise ValueError(f"Trainer type '{name}' is already registered.")

    if not issubclass(trainer_class, BaseTrainer):
        raise ValueError(
            f"Trainer class '{trainer_class.__name__}' must inherit from BaseTrainer."
        )

    TRAINER_REGISTRY[name] = trainer_class
    logger.info(f"Registered new trainer type: '{name}' -> {trainer_class.__name__}")

# Define what is exported when 'from trainers import *' is used.
# It's generally good practice to be explicit.
__all__ = [
    'BaseTrainer',
    'Callback',
    'SimpleTrainer',
    # 'AdvancedTrainer', # Add other trainers as they are created
    'get_trainer',
    'register_trainer',
    'TRAINER_REGISTRY' # Exposing the registry can be useful for introspection
]

