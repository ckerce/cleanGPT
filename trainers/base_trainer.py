# ./trainers/base_trainer.py
"""
Base Trainer Abstract Class
Defines the interface that all trainers must implement
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class BaseTrainer(ABC):
    """
    Abstract base class for all trainers in the cleanGPT project.
    
    This class defines the interface that all trainer implementations
    must adhere to, ensuring consistency across different training strategies.
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 output_dir: Optional[str] = None):
        """
        Initialize the base trainer.
        
        Args:
            model: The model to train
            dataloader: DataLoader providing training batches
            optimizer: Optimizer for parameter updates
            device: Device to run training on
            output_dir: Directory to save outputs (optional)
        """
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.output_dir = output_dir
        
        # Create output directory if specified and not exists
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
    
    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """
        Execute the training loop.
        
        Returns:
            Dictionary containing training metrics
        """
        pass
    
    @abstractmethod
    def evaluate(self, eval_dataloader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Evaluate the model.
        
        Args:
            eval_dataloader: DataLoader for evaluation (uses training dataloader if None)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        pass
    
    def save_checkpoint(self, path: str):
        """
        Save a training checkpoint.
        
        Args:
            path: Path to save the checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to: {path}")
    
    def load_checkpoint(self, path: str):
        """
        Load a training checkpoint.
        
        Args:
            path: Path to the checkpoint file
            
        Returns:
            The loaded checkpoint data
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded from: {path}")
        return checkpoint
    
    def log_batch(self, batch_idx: int, loss: float, 
                  metrics: Optional[Dict[str, Any]] = None):
        """
        Log information about a training batch.
        
        Args:
            batch_idx: Index of the current batch
            loss: Training loss for the batch
            metrics: Additional metrics to log
        """
        metrics_str = ""
        if metrics:
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        
        logger.info(f"Batch {batch_idx}, Loss: {loss:.4f}" + 
                   (f", {metrics_str}" if metrics_str else ""))
    
    def log_epoch(self, epoch: int, avg_loss: float, 
                  metrics: Optional[Dict[str, Any]] = None):
        """
        Log information about a training epoch.
        
        Args:
            epoch: Current epoch number
            avg_loss: Average loss for the epoch
            metrics: Additional metrics to log
        """
        metrics_str = ""
        if metrics:
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        
        logger.info(f"Epoch {epoch} completed, Avg Loss: {avg_loss:.4f}" + 
                   (f", {metrics_str}" if metrics_str else ""))
