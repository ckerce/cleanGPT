# ./trainers/simple_trainer.py
"""
Simple Trainer Implementation
Basic training loop with progress tracking and callback integration.
"""

import time
import logging
from typing import Dict, Any, Optional, List # Added List
import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Assuming Callback is in base_trainer or accessible via from .base_trainer import Callback
from .base_trainer import BaseTrainer, Callback


logger = logging.getLogger(__name__)

class SimpleTrainer(BaseTrainer):
    """
    Simple trainer implementation with a standard training loop.
    This trainer provides a straightforward training process with
    progress tracking, basic logging, and callback support.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 num_epochs: int = 5,
                 output_dir: Optional[str] = None,
                 clip_grad_norm: Optional[float] = None,
                 log_interval: int = 10,
                 callbacks: Optional[List[Callback]] = None):
        """
        Initialize the simple trainer.

        Args:
            model: Model to train.
            dataloader: DataLoader for training data.
            optimizer: Optimizer for parameter updates.
            device: Device to train on.
            num_epochs: Number of training epochs.
            output_dir: Directory to save outputs (e.g., checkpoints).
            clip_grad_norm: Maximum norm for gradient clipping (None = no clipping).
            log_interval: Number of batches between logging.
            callbacks: A list of Callback instances. Optional.
        """
        super().__init__(model, dataloader, optimizer, device, output_dir, callbacks)
        self.num_epochs = num_epochs
        self.clip_grad_norm = clip_grad_norm
        self.log_interval = log_interval
        # self.trainer_state is inherited from BaseTrainer

        logger.info(f"SimpleTrainer initialized with {self.num_epochs} epochs.")
        if self.clip_grad_norm:
            logger.info(f"Gradient clipping enabled with max norm: {self.clip_grad_norm}")
        if self.callbacks:
            logger.info(f"Attached callbacks: {[cb.__class__.__name__ for cb in self.callbacks]}")


    def train(self) -> Dict[str, Any]:
        """
        Execute the training loop.

        Returns:
            A dictionary containing training metrics.
        """
        logger.info("Starting training...")
        self.model.to(self.device)
        self.model.train() # Set model to training mode

        self.trainer_state['num_epochs'] = self.num_epochs
        self._trigger_callbacks('on_train_begin', logs=self.trainer_state)

        total_start_time = time.time()
        training_metrics = {
            'epoch_losses': [],
            'final_loss': float('nan'), # Initialize with NaN
            'training_time': 0.0
        }

        for epoch in range(1, self.num_epochs + 1): # Epochs 1-indexed for callbacks
            self.trainer_state['current_epoch'] = epoch
            self._trigger_callbacks('on_epoch_begin', epoch, logs=self.trainer_state)
            epoch_start_time = time.time()
            epoch_loss = 0.0
            num_batches_processed = 0

            progress_bar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch}/{self.num_epochs}",
                leave=False
            )

            for batch_idx, batch_data in enumerate(progress_bar):
                self.trainer_state['current_batch_idx'] = batch_idx
                batch_logs = {'batch_data_keys': list(batch_data.keys())} # Example log
                self._trigger_callbacks('on_batch_begin', batch_idx, logs=batch_logs)

                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch_data.items() if isinstance(v, torch.Tensor)}

                # Forward pass
                outputs = self.model(**batch) # Assumes model returns dict with 'loss'
                loss = outputs.get('loss')

                if loss is None:
                    logger.warning(f"Epoch {epoch}, Batch {batch_idx}: Loss is None. Skipping optimization.")
                    self._trigger_callbacks('on_batch_end', batch_idx, logs={'loss': None})
                    continue
                if torch.isnan(loss):
                    logger.error(f"Epoch {epoch}, Batch {batch_idx}: Loss is NaN. Stopping training.")
                    self.trainer_state['status'] = 'NaN Loss'
                    self._trigger_callbacks('on_train_end', logs=self.trainer_state)
                    training_metrics['training_time'] = time.time() - total_start_time
                    return training_metrics

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()

                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.clip_grad_norm
                    )

                self.optimizer.step()

                batch_loss_item = loss.item()
                epoch_loss += batch_loss_item
                num_batches_processed += 1

                self._trigger_callbacks('on_batch_end', batch_idx, logs={'loss': batch_loss_item})
                progress_bar.set_postfix({"loss": f"{batch_loss_item:.4f}"})

                if batch_idx % self.log_interval == 0:
                    self.log_batch(batch_idx, batch_loss_item, epoch=epoch)

            avg_epoch_loss = epoch_loss / num_batches_processed if num_batches_processed > 0 else float('nan')
            training_metrics['epoch_losses'].append(avg_epoch_loss)

            epoch_duration = time.time() - epoch_start_time
            self.log_epoch(epoch, avg_epoch_loss) # Uses BaseTrainer's log_epoch

            epoch_end_logs = {'loss': avg_epoch_loss, 'epoch_duration': epoch_duration}
            self.trainer_state.update(epoch_end_logs)
            self._trigger_callbacks('on_epoch_end', epoch, logs=self.trainer_state)

            if self.output_dir:
                checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")
                self.save_checkpoint(checkpoint_path, epoch=epoch, loss=avg_epoch_loss)

        if training_metrics['epoch_losses']:
             training_metrics['final_loss'] = training_metrics['epoch_losses'][-1]
        training_metrics['training_time'] = time.time() - total_start_time

        logger.info(f"Training completed in {training_metrics['training_time']:.2f}s")
        logger.info(f"Final average training loss: {training_metrics['final_loss']:.6f}")

        self.trainer_state['status'] = 'Completed'
        self.trainer_state.update(training_metrics)
        self._trigger_callbacks('on_train_end', logs=self.trainer_state)

        return training_metrics

    def evaluate(self, eval_dataloader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Evaluate the model on a dataset.

        Args:
            eval_dataloader: DataLoader for evaluation data.
                             If None, uses the training dataloader (not recommended for true eval).

        Returns:
            A dictionary containing evaluation metrics.
        """
        logger.info("Starting evaluation...")
        if eval_dataloader is None:
            logger.warning("eval_dataloader not provided to evaluate(). Using training dataloader. This may not be a proper evaluation.")
            eval_dataloader = self.dataloader

        self.trainer_state['eval_dataloader_len'] = len(eval_dataloader)
        self._trigger_callbacks('on_evaluate_begin', logs=self.trainer_state)

        self.model.eval() # Set model to evaluation mode

        total_loss = 0.0
        total_samples = 0 # Or total batches if loss is already averaged
        num_batches_processed = 0


        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
                batch_logs = {'batch_data_keys': list(batch_data.keys())}
                self._trigger_callbacks('on_batch_begin', batch_idx, logs=batch_logs) # Can reuse batch hooks if desired

                batch = {k: v.to(self.device) for k, v in batch_data.items() if isinstance(v, torch.Tensor)}
                outputs = self.model(**batch)
                loss = outputs.get('loss')

                if loss is None or torch.isnan(loss):
                    logger.warning(f"Evaluation Batch {batch_idx}: Loss is None or NaN. Skipping.")
                    self._trigger_callbacks('on_batch_end', batch_idx, logs={'loss': None})
                    continue

                # Assuming loss is per-sample or needs to be scaled by batch size
                # If loss is already mean loss for the batch:
                # total_loss += loss.item()
                # num_batches_processed += 1
                # else (if loss is sum):
                batch_size = batch.get('input_ids', next(iter(batch.values()))).size(0)
                total_loss += loss.item() * batch_size # Accumulate sum of losses
                total_samples += batch_size
                num_batches_processed +=1


                self._trigger_callbacks('on_batch_end', batch_idx, logs={'loss': loss.item()})

        avg_loss = total_loss / total_samples if total_samples > 0 else float('nan')
        # If using already averaged batch losses:
        # avg_loss = total_loss / num_batches_processed if num_batches_processed > 0 else float('nan')


        eval_metrics = {'loss': avg_loss}
        if avg_loss is not None and not torch.isnan(torch.tensor(avg_loss)):
            eval_metrics['perplexity'] = torch.exp(torch.tensor(avg_loss)).item()
        else:
            eval_metrics['perplexity'] = float('nan')

        self.model.train() # Set model back to training mode

        logger.info(f"Evaluation results: Loss: {eval_metrics['loss']:.6f}, Perplexity: {eval_metrics['perplexity']:.6f}")
        self.trainer_state.update(eval_metrics)
        self._trigger_callbacks('on_evaluate_end', logs=self.trainer_state)

        return eval_metrics

