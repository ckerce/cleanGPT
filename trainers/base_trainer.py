# ./trainers/base_trainer.py
"""
Base Trainer Abstract Class and Callback System
Defines the interface that all trainers must implement and a base for callbacks.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List # Added List

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# --- Callback Base Class ---
class Callback:
    """
    Base class for creating callbacks.
    Callbacks can be used to customize the training loop by adding actions
    at various stages (e.g., at the end of an epoch, beginning of training).
    """
    def __init__(self):
        # These attributes will be set by the trainer
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.dataloader: Optional[DataLoader] = None
        self.device: Optional[torch.device] = None
        self.output_dir: Optional[str] = None
        self.current_epoch: Optional[int] = None
        self.current_batch: Optional[int] = None # Batch index within an epoch
        self.num_epochs: Optional[int] = None
        self.trainer_state: Dict[str, Any] = {} # For trainer to share general state

    def set_trainer_references(self, trainer: 'BaseTrainer'):
        """
        Allows the trainer to pass references to itself and its components.
        This method is called by the trainer during its initialization.

        Args:
            trainer: The trainer instance.
        """
        self.model = trainer.model
        self.optimizer = trainer.optimizer
        self.dataloader = trainer.dataloader # Main training dataloader
        self.device = trainer.device
        self.output_dir = trainer.output_dir
        # Specific trainers might have more attributes to set
        if hasattr(trainer, 'num_epochs'):
            self.num_epochs = trainer.num_epochs
        # You can add more attributes from trainer if needed by callbacks

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """
        Called at the beginning of an epoch.
        Args:
            epoch (int): The current epoch number (1-indexed).
            logs (dict, optional): Currently no logs are passed.
        """
        self.current_epoch = epoch

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """
        Called at the end of an epoch.
        Args:
            epoch (int): The current epoch number (1-indexed).
            logs (dict, optional): Contains metrics like 'loss', 'epoch_duration'.
        """
        pass

    def on_batch_begin(self, batch_idx: int, logs: Optional[Dict[str, Any]] = None):
        """
        Called at the beginning of a training batch.
        Args:
            batch_idx (int): The current batch index within the epoch.
            logs (dict, optional): Contains 'batch_data'.
        """
        self.current_batch = batch_idx

    def on_batch_end(self, batch_idx: int, logs: Optional[Dict[str, Any]] = None):
        """
        Called at the end of a training batch.
        Args:
            batch_idx (int): The current batch index within the epoch.
            logs (dict, optional): Contains 'loss' for the batch.
        """
        pass

    def on_evaluate_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of an evaluation phase."""
        pass

    def on_evaluate_end(self, logs: Optional[Dict[str, Any]] = None):
        """
        Called at the end of an evaluation phase.
        Args:
            logs (dict, optional): Contains evaluation metrics like 'loss', 'perplexity'.
        """
        pass


# --- Base Trainer Abstract Class ---
class BaseTrainer(ABC):
    """
    Abstract base class for all trainers in the cleanGPT project.

    This class defines the interface that all trainer implementations
    must adhere to, ensuring consistency across different training strategies.
    It also includes a callback system for extending training loop behavior.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 output_dir: Optional[str] = None,
                 callbacks: Optional[List[Callback]] = None):
        """
        Initialize the base trainer.

        Args:
            model: The model to train.
            dataloader: DataLoader providing training batches.
            optimizer: Optimizer for parameter updates.
            device: Device to run training on.
            output_dir: Directory to save outputs (e.g., checkpoints). Optional.
            callbacks: A list of Callback instances. Optional.
        """
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.output_dir = output_dir
        self.callbacks = callbacks if callbacks is not None else []
        self.trainer_state: Dict[str, Any] = {} # General state for callbacks

        # Set trainer references for each callback
        for cb in self.callbacks:
            cb.set_trainer_references(self)

        # Create output directory if specified and it doesn't exist
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")

    def _trigger_callbacks(self, event_name: str, *args, **kwargs):
        """
        Helper method to call a specific event method on all registered callbacks.

        Args:
            event_name (str): The name of the callback event (e.g., 'on_epoch_end').
            *args: Positional arguments to pass to the callback method.
            **kwargs: Keyword arguments to pass to the callback method.
        """
        for cb in self.callbacks:
            method = getattr(cb, event_name, None)
            if method and callable(method):
                try:
                    method(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in callback {cb.__class__.__name__}.{event_name}: {e}", exc_info=True)


    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """
        Execute the training loop.
        This method must be implemented by concrete trainer classes.

        Returns:
            A dictionary containing training metrics (e.g., final loss, training time).
        """
        pass

    @abstractmethod
    def evaluate(self, eval_dataloader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Evaluate the model.
        This method must be implemented by concrete trainer classes.

        Args:
            eval_dataloader: DataLoader for evaluation. If None, the trainer might
                             use its default training dataloader or raise an error.

        Returns:
            A dictionary containing evaluation metrics (e.g., loss, perplexity).
        """
        pass

    def save_checkpoint(self, path: str, epoch: Optional[int] = None, **kwargs):
        """
        Save a training checkpoint.

        Args:
            path (str): Path to save the checkpoint file.
            epoch (int, optional): The epoch number, if applicable.
            **kwargs: Additional information to save in the checkpoint.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            **kwargs # Include any additional user-provided data
        }

        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to: {path}")

    def load_checkpoint(self, path: str, map_location: Optional[Union[str, torch.device]] = None):
        """
        Load a training checkpoint.

        Args:
            path (str): Path to the checkpoint file.
            map_location (str or torch.device, optional): Specifies how to remap
                storage locations. Default is self.device.

        Returns:
            The loaded checkpoint dictionary.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        load_location = map_location if map_location is not None else self.device
        checkpoint = torch.load(path, map_location=load_location)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            logger.warning("Optimizer state not found in checkpoint or optimizer not initialized.")

        logger.info(f"Checkpoint loaded from: {path}")
        # Optionally, return epoch or other info if needed by the caller
        return checkpoint

    def log_batch(self,
                  batch_idx: int,
                  loss: float,
                  epoch: Optional[int] = None,
                  metrics: Optional[Dict[str, Any]] = None):
        """
        Log information about a training batch.

        Args:
            batch_idx (int): Index of the current batch.
            loss (float): Training loss for the batch.
            epoch (int, optional): Current epoch number.
            metrics (dict, optional): Additional metrics to log.
        """
        metrics_str = ""
        if metrics:
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())

        epoch_str = f"Epoch {epoch}, " if epoch is not None else ""
        logger.info(f"{epoch_str}Batch {batch_idx}, Loss: {loss:.4f}" +
                   (f", {metrics_str}" if metrics_str else ""))

    def log_epoch(self,
                  epoch: int,
                  avg_loss: float,
                  metrics: Optional[Dict[str, Any]] = None):
        """
        Log information about a training epoch.

        Args:
            epoch (int): Current epoch number.
            avg_loss (float): Average loss for the epoch.
            metrics (dict, optional): Additional metrics to log (e.g., validation loss).
        """
        metrics_str = ""
        if metrics:
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())

        logger.info(f"Epoch {epoch} completed, Avg Training Loss: {avg_loss:.4f}" +
                   (f", {metrics_str}" if metrics_str else ""))


