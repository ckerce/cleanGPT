# ./distillation/trainers/head_trainer.py
"""
Trainer for language model head distillation.
"""
import os
import logging
import torch
from tqdm.auto import tqdm
from typing import List, Optional, Union

from distillation.trainers.base_trainer import BaseDistillationTrainer
from distillation.losses.logit_loss import LogitDistillationLoss

logger = logging.getLogger(__name__)

class HeadDistillationTrainer(BaseDistillationTrainer):
    """Trainer for language model head distillation."""
    
    def __init__(self,
                 teacher_model,
                 student_model,
                 tokenizer,
                 train_dataloader,
                 logit_loss_type="kl_div",
                 logit_loss_temperature=2.0,
                 logit_loss_weight=1.0,
                 device=torch.device("cpu"),
                 output_dir="./distilled_model",
                 log_interval=50,
                 optimizer_cls=torch.optim.AdamW):
        """
        Initialize the LM head distillation trainer.
        
        Args:
            teacher_model: Teacher model to distill from
            student_model: Student model to train
            tokenizer: Tokenizer for processing text
            train_dataloader: DataLoader for training examples
            logit_loss_type: Type of logit loss ("mse", "kl_div", or "ce")
            logit_loss_temperature: Temperature for logit loss
            logit_loss_weight: Weight for logit loss
            device: Device to train on
            output_dir: Directory to save model checkpoints
            log_interval: How often to log during training
            optimizer_cls: Optimizer class to use
        """
        super().__init__(
            teacher_model=teacher_model,
            student_model=student_model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            device=device,
            output_dir=output_dir,
            log_interval=log_interval,
            optimizer_cls=optimizer_cls
        )
        
        # Configure logit loss
        self.logit_loss_type = logit_loss_type
        self.logit_loss_temperature = logit_loss_temperature
        self.logit_loss_weight = logit_loss_weight
        
        self.logit_loss_fn = LogitDistillationLoss(
            loss_type=self.logit_loss_type,
            temperature=self.logit_loss_temperature
        ).to(device)
    
    def initialize_from_teacher(self):
        """
        Initialize student LM head from teacher.
        
        This is a new feature that copies the teacher LM head weights to the student
        as a better starting point for distillation. It handles dimensional mismatches
        with a simple projection when needed.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        logger.info("Initializing student LM head from teacher...")
        try:
            # Check if teacher and student have lm_head
            if not (hasattr(self.teacher_model, 'lm_head') and hasattr(self.student_model, 'lm_head')):
                logger.warning("Could not initialize LM head: missing lm_head attribute in teacher or student")
                return False
                
            # Get dimensions
            teacher_dim = self.teacher_model.lm_head.weight.shape[1]
            student_dim = self.student_model.lm_head.weight.shape[1]
            
            # Direct copy if dimensions match
            if teacher_dim == student_dim:
                logger.info(f"Teacher and student LM head dimensions match ({teacher_dim}). Direct weight copy.")
                self.student_model.lm_head.weight.data.copy_(
                    self.teacher_model.lm_head.weight.data)
                if hasattr(self.student_model.lm_head, 'bias') and self.student_model.lm_head.bias is not None:
                    if hasattr(self.teacher_model.lm_head, 'bias') and self.teacher_model.lm_head.bias is not None:
                        self.student_model.lm_head.bias.data.copy_(
                            self.teacher_model.lm_head.bias.data)
            else:
                # Initialize with a projection of teacher weights
                logger.info(f"Teacher ({teacher_dim}) and student ({student_dim}) LM head dimensions differ. "
                           f"Using projection for initialization.")
                with torch.no_grad():
                    # Create a simple linear projection
                    projection = torch.nn.Linear(teacher_dim, student_dim, bias=False).to(self.device)
                    # Project each vocabulary embedding
                    student_lm_head_weights = []
                    
                    # Process in batches to avoid potential memory issues with large vocab
                    batch_size = 512
                    teacher_weights = self.teacher_model.lm_head.weight
                    for i in range(0, teacher_weights.size(0), batch_size):
                        end_idx = min(i + batch_size, teacher_weights.size(0))
                        batch = teacher_weights[i:end_idx]
                        # Project and store
                        projected_batch = projection(batch)
                        student_lm_head_weights.append(projected_batch)
                    
                    # Concatenate all batches
                    projected_weights = torch.cat(student_lm_head_weights, dim=0)
                    # Copy to student
                    self.student_model.lm_head.weight.data.copy_(projected_weights)
                    
            logger.info(f"Successfully initialized student LM head from teacher")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing LM head from teacher: {e}")
            return False
            
    def _get_lm_head_parameters(self) -> List[torch.nn.Parameter]:
        """
        Get parameters of the LM head and final layer norm.
        
        Returns:
            List of parameters to optimize
        """
        params_to_train = []
        
        # Add LM head parameters
        if hasattr(self.student_model, 'lm_head'):
            params_to_train.extend(list(self.student_model.lm_head.parameters()))
            logger.debug(f"Added {len(list(self.student_model.lm_head.parameters()))} params from lm_head.")
            
        # Add final layer norm parameters
        if hasattr(self.student_model.transformer, 'ln_f'): 
            params_to_train.extend(list(self.student_model.transformer.ln_f.parameters()))
            logger.debug(f"Added {len(list(self.student_model.transformer.ln_f.parameters()))} params from ln_f.")
        
        # Deduplicate parameters
        unique_params_to_train = []
        seen_params_ids = set()
        for p in params_to_train:
            if p.requires_grad and id(p) not in seen_params_ids:
                unique_params_to_train.append(p)
                seen_params_ids.add(id(p))
        
        logger.info(f"Optimizing {len(unique_params_to_train)} parameters for LM head.")
        return unique_params_to_train
        
    def distill_lm_head(self,
                        num_epochs: int,
                        learning_rate: float,
                        weight_decay: float = 0.01,
                        max_grad_norm: Optional[float] = 1.0):
        """
        Distill the language model head.
        
        Args:
            num_epochs: Number of epochs to train
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            max_grad_norm: Maximum gradient norm for clipping
            
        Returns:
            Average loss for the LM head
        """
        logger.info(f"--- Starting distillation for Language Model Head ---")
        self.student_model.train()
        
        # Freeze all student transformer blocks and embeddings
        for i in range(len(self.student_model.transformer.h)):
            for param in self.student_model.transformer.h[i].parameters():
                param.requires_grad = False
        
        if hasattr(self.student_model.transformer, 'wte'):
            for param in self.student_model.transformer.wte.parameters(): param.requires_grad = False
        if hasattr(self.student_model.transformer, 'wpe'):
            for param in self.student_model.transformer.wpe.parameters(): param.requires_grad = False
        
        # Ensure lm_head and final layer norm (ln_f) are trainable
        if hasattr(self.student_model, 'lm_head'):
            for param in self.student_model.lm_head.parameters(): param.requires_grad = True
        if hasattr(self.student_model.transformer, 'ln_f'):
            for param in self.student_model.transformer.ln_f.parameters(): param.requires_grad = True
        
        logger.info("Froze student transformer blocks and embeddings for LM head distillation.")
        
        # Get parameters to optimize
        params_to_optimize = self._get_lm_head_parameters()
        if not params_to_optimize:
            logger.warning("No parameters found for language model head. Skipping LM head distillation.")
            return float('nan')

        # Create optimizer
        optimizer = self.optimizer_cls(params_to_optimize, lr=learning_rate, weight_decay=weight_decay)

        # Train for specified number of epochs
        final_avg_loss = float('nan')
        for epoch in range(num_epochs):
            logger.info(f"LM Head Distillation, Epoch {epoch + 1}/{num_epochs}")
            
            epoch_loss_sum = 0.0
            num_batches_processed = 0
            progress_bar = tqdm(
                self.train_dataloader, 
                desc=f"Epoch {epoch+1} LM Head", 
                leave=False
            )

            for batch_idx_iter, batch_data in enumerate(progress_bar):
                input_ids = batch_data['input_ids'].to(self.device)
                attention_mask = batch_data.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                optimizer.zero_grad()

                # Get teacher logits
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                teacher_logits = teacher_outputs.logits

                # Get student logits
                student_outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                student_logits = student_outputs.get('logits')
                if student_logits is None:
                    raise ValueError("Student model did not return 'logits' during LM head distillation.")

                # Compute loss
                loss = self.logit_loss_fn(student_logits, teacher_logits)
                
                # Apply weight if needed
                if self.logit_loss_weight != 1.0:
                    loss = loss * self.logit_loss_weight

                # Skip problematic losses
                if loss is None or torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"LM Head, Epoch {epoch+1}, Batch {batch_idx_iter}: "
                                f"Loss is {loss}. Skipping batch.")
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.save_checkpoint(
                            os.path.join(self.output_dir, "student_model_error_loss_lm_head.pt"), 
                            error_loss=float(loss.item() if loss is not None else float('nan'))
                        )
                        raise RuntimeError(f"Fatal {loss} loss in LM head distillation.")
                    continue
                
                # Backward pass and optimization
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, max_grad_norm)
                optimizer.step()

                # Logging
                epoch_loss_sum += loss.item()
                num_batches_processed += 1
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                if batch_idx_iter > 0 and batch_idx_iter % self.log_interval == 0:
                    current_avg_loss = epoch_loss_sum / num_batches_processed if num_batches_processed > 0 else float('nan')
                    logger.info(f"LM Head, Epoch {epoch+1}, Batch {batch_idx_iter}/"
                               f"{len(self.train_dataloader)}: Avg Loss: {current_avg_loss:.4f}, "
                               f"Current Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss_sum / num_batches_processed if num_batches_processed > 0 else float('nan')
            logger.info(f"LM Head, Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
            final_avg_loss = avg_epoch_loss
        
        logger.info(f"--- Finished distillation for Language Model Head ---")
        lm_head_checkpoint_path = os.path.join(self.output_dir, "student_model_lm_head_distilled.pt")
        self.save_checkpoint(
            lm_head_checkpoint_path, 
            lm_head_distilled=True, 
            epoch=num_epochs, 
            avg_loss=final_avg_loss
        )
        logger.info(f"Saved student model checkpoint after LM head distillation to {lm_head_checkpoint_path}")
        
        return final_avg_loss
        
    def train_head(self, 
                  lm_head_epochs: int, 
                  lm_head_lr: float, 
                  lm_head_wd: float = 0.01, 
                  lm_head_max_grad_norm: Optional[float] = 1.0,
                  initialize_from_teacher: bool = True):
        """
        Train the language model head.
        
        Args:
            lm_head_epochs: Number of epochs to train
            lm_head_lr: Learning rate
            lm_head_wd: Weight decay
            lm_head_max_grad_norm: Maximum gradient norm
            initialize_from_teacher: Whether to initialize the LM head from the teacher
            
        Returns:
            Average loss for the LM head
        """
        # Initialize from teacher if requested
        if initialize_from_teacher:
            self.initialize_from_teacher()
            
        # Distill LM head
        return self.distill_lm_head(
            num_epochs=lm_head_epochs,
            learning_rate=lm_head_lr,
            weight_decay=lm_head_wd,
            max_grad_norm=lm_head_max_grad_norm
        )
