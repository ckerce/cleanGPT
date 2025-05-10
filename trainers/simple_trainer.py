# ./trainers/simple_trainer.py
"""
Simple Trainer Implementation
Basic training loop with progress tracking
"""

import time
import logging
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

class SimpleTrainer(BaseTrainer):
    """
    Simple trainer implementation with a standard training loop.
    
    This trainer provides a straightforward training process with
    progress tracking and basic logging.
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 num_epochs: int = 5,
                 output_dir: Optional[str] = None,
                 clip_grad_norm: Optional[float] = None,
                 log_interval: int = 10):
        """
        Initialize the simple trainer.
        
        Args:
            model: Model to train
            dataloader: DataLoader for training data
            optimizer: Optimizer for parameter updates
            device: Device to train on
            num_epochs: Number of training epochs
            output_dir: Directory to save outputs
            clip_grad_norm: Maximum norm for gradient clipping (None = no clipping)
            log_interval: Number of batches between logging
        """
        super().__init__(model, dataloader, optimizer, device, output_dir)
        self.num_epochs = num_epochs
        self.clip_grad_norm = clip_grad_norm
        self.log_interval = log_interval
        
        logger.info(f"SimpleTrainer initialized with {num_epochs} epochs")
        if clip_grad_norm:
            logger.info(f"Gradient clipping enabled with max norm: {clip_grad_norm}")
    
    def train(self) -> Dict[str, Any]:
        """
        Execute the training loop.
        
        Returns:
            Dictionary with training metrics
        """
        logger.info("Starting training...")
        self.model.to(self.device)
        self.model.train()
        
        total_start_time = time.time()
        training_metrics = {
            'epoch_losses': [],
            'final_loss': 0.0,
            'training_time': 0.0
        }
        
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            
            # Use tqdm for a progress bar
            progress_bar = tqdm(
                self.dataloader, 
                desc=f"Epoch {epoch+1}/{self.num_epochs}", 
                leave=False
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs['loss']
                
                # Check if loss is valid
                if loss is None:
                    logger.warning("Loss is None for a batch, skipping optimization step.")
                    continue
                if torch.isnan(loss):
                    logger.error("Loss is NaN, stopping training.")
                    return training_metrics
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                
                # Optional gradient clipping
                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.clip_grad_norm
                    )
                
                self.optimizer.step()
                
                # Update metrics
                batch_loss = loss.item()
                epoch_loss += batch_loss
                
                # Update progress bar
                progress_bar.set_postfix({"loss": f"{batch_loss:.4f}"})
                
                # Log batch information
                if batch_idx % self.log_interval == 0:
                    self.log_batch(batch_idx, batch_loss)
            
            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / len(self.dataloader)
            training_metrics['epoch_losses'].append(avg_epoch_loss)
            
            # Log epoch information
            epoch_duration = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} completed in {epoch_duration:.2f}s")
            self.log_epoch(epoch+1, avg_epoch_loss)
            
            # Save checkpoint
            if self.output_dir:
                checkpoint_path = f"{self.output_dir}/checkpoint_epoch_{epoch+1}.pt"
                self.save_checkpoint(checkpoint_path)
        
        # Calculate final metrics
        training_metrics['final_loss'] = training_metrics['epoch_losses'][-1]
        training_metrics['training_time'] = time.time() - total_start_time
        
        logger.info(f"Training completed in {training_metrics['training_time']:.2f}s")
        logger.info(f"Final loss: {training_metrics['final_loss']:.6f}")
        
        return training_metrics
    
    def evaluate(self, eval_dataloader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Evaluate the model on a dataset.
        
        Args:
            eval_dataloader: DataLoader for evaluation data
                            (uses training dataloader if None)
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Starting evaluation...")
        
        # Use provided dataloader or training dataloader
        dataloader = eval_dataloader if eval_dataloader else self.dataloader
        
        # Set model to evaluation mode
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        # Evaluate without computing gradients
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs['loss']
                
                # Skip invalid loss values
                if loss is None or torch.isnan(loss):
                    continue
                
                # Update metrics
                batch_size = batch['input_ids'].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        # Calculate average loss
        avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
        
        # Set model back to training mode
        self.model.train()
        
        # Prepare results
        eval_metrics = {
            'loss': avg_loss,
            'perplexity': torch.exp(torch.tensor(avg_loss)).item()
        }
        
        logger.info(f"Evaluation results: Loss: {avg_loss:.6f}, Perplexity: {eval_metrics['perplexity']:.6f}")
        
        return eval_metrics
