# ./distillation/trainers/base_trainer.py
"""
Base trainer class for distillation. Provides common functionality.
"""
import os
import logging
import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

class BaseDistillationTrainer:
    def __init__(self,
                 teacher_model,
                 student_model,
                 tokenizer,
                 train_dataloader,
                 device,
                 output_dir,
                 log_interval,
                 optimizer_cls=torch.optim.AdamW):
        """
        Base trainer for distillation tasks.
        
        Args:
            teacher_model: The teacher model
            student_model: The student model
            tokenizer: Tokenizer for text processing
            train_dataloader: DataLoader for training data
            device: Device to use (cuda, cpu, mps)
            output_dir: Directory to save outputs
            log_interval: How often to log during training
            optimizer_cls: Optimizer class to use for training
        """
        self.teacher_model = teacher_model.to(device).eval()
        self.student_model = student_model.to(device)
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.device = device
        self.output_dir = output_dir
        self.log_interval = log_interval
        self.optimizer_cls = optimizer_cls
        
        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")
            
    def save_checkpoint(self, path, **kwargs):
        """Save a checkpoint with model state and additional information.
        
        Args:
            path: Path to save the checkpoint
            **kwargs: Additional information to save in the checkpoint
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            # Attempt to handle GPTConfig safely if used by student_model.config
            if hasattr(self.student_model.config, '__class__'):
                config_class = self.student_model.config.__class__
                # Check if it's likely the GPTConfig from config_distillation
                if "GPTConfig" in str(config_class) and "config_distillation" in str(config_class.__module__):
                     import torch.serialization
                     if hasattr(torch.serialization, 'add_safe_globals'):
                         torch.serialization.add_safe_globals([config_class])
                         logger.debug(f"Added {config_class} to safe globals for pickling.")
        except ImportError:
            logger.debug("Could not import GPTConfig for pickling check.")
        except Exception as e:
            logger.warning(f"Could not add student_model.config's class to safe globals: {e}")
        
        checkpoint = {
            'model_state_dict': self.student_model.state_dict(),
            'student_config': self.student_model.config, 
            **kwargs
        }
        
        torch.save(checkpoint, path, pickle_protocol=4) 
        logger.info(f"Checkpoint saved to {path}")
        
    def load_checkpoint(self, path):
        """Load a checkpoint.
        
        Args:
            path: Path to the checkpoint to load
            
        Returns:
            The loaded checkpoint
        """
        if not os.path.exists(path):
            logger.error(f"Checkpoint path does not exist: {path}")
            return None
            
        try:
            checkpoint = torch.load(path, map_location=self.device)
            logger.info(f"Loaded checkpoint from {path}")
            return checkpoint
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None
            
    def _run_training_epoch(self, optimizer, loss_fn, params_to_optimize, max_grad_norm, 
                            epoch, num_epochs, desc_prefix):
        """Common training loop logic for one epoch.
        
        Args:
            optimizer: Optimizer to use
            loss_fn: Loss function
            params_to_optimize: Parameters being optimized
            max_grad_norm: Maximum gradient norm for clipping
            epoch: Current epoch number
            num_epochs: Total number of epochs
            desc_prefix: Prefix for progress bar description
            
        Returns:
            Tuple of (total_loss, num_batches_processed)
        """
        epoch_loss_sum = 0.0
        num_batches_processed = 0
        progress_bar = tqdm(
            self.train_dataloader, 
            desc=f"Epoch {epoch+1}/{num_epochs} {desc_prefix}", 
            leave=False
        )

        for batch_idx, batch_data in enumerate(progress_bar):
            input_ids = batch_data['input_ids'].to(self.device)
            attention_mask = batch_data.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            optimizer.zero_grad()
            
            # The actual forward pass and loss computation will be implemented
            # in the derived classes that use this method
            
            # Placeholder to be overridden
            loss = None
            
            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"{desc_prefix}, Epoch {epoch+1}, Batch {batch_idx}: "
                             f"Loss is {loss}. Skipping batch.")
                continue
                
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(params_to_optimize, max_grad_norm)
            optimizer.step()

            epoch_loss_sum += loss.item()
            num_batches_processed += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if batch_idx > 0 and batch_idx % self.log_interval == 0:
                current_avg_loss = epoch_loss_sum / num_batches_processed
                logger.info(f"{desc_prefix}, Epoch {epoch+1}, Batch {batch_idx}/"
                           f"{len(self.train_dataloader)}: Avg Loss: {current_avg_loss:.4f}, "
                           f"Current Loss: {loss.item():.4f}")
                
        avg_epoch_loss = epoch_loss_sum / num_batches_processed if num_batches_processed > 0 else float('nan')
        logger.info(f"{desc_prefix}, Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
        
        return epoch_loss_sum, num_batches_processed
