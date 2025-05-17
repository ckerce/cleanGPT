# ./distillation_module.py
"""
Module for block-by-block distillation of transformer models.
Contains the BlockDistillationTrainer and loss functions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm
import logging
import os
import math
from typing import Optional, Dict, Any, List 

logger = logging.getLogger(__name__)

class DistillationLoss(nn.Module):
    """
    Wrapper for various distillation loss functions.
    Currently supports MSE. Can be extended for KL divergence, etc.
    """
    def __init__(self, loss_type="mse", temperature=1.0):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.temperature = temperature
        if self.loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif self.loss_type == "kl_div":
            # For KL divergence, inputs are expected to be log-probabilities (student)
            # and probabilities (teacher), or both probabilities.
            # Assumes student_outputs are logits and teacher_outputs are logits.
            self.loss_fn = nn.KLDivLoss(reduction='batchmean') # 'batchmean' averages over batch and elements
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}. Supported: 'mse', 'kl_div'.")

    def forward(self, student_outputs, teacher_outputs):
        """
        Args:
            student_outputs: Tensor from student model (e.g., hidden state or logits).
            teacher_outputs: Tensor from teacher model (e.g., hidden state or logits).
        """
        if teacher_outputs.shape != student_outputs.shape:
            logger.warning(f"Shape mismatch for loss calculation: student {student_outputs.shape}, teacher {teacher_outputs.shape}. MSE might fail or be ill-defined.")

        if self.loss_type == "mse":
            return self.loss_fn(student_outputs, teacher_outputs.detach())
        elif self.loss_type == "kl_div":
            # Assumes inputs are logits.
            # Normalize by temperature.
            student_log_probs = F.log_softmax(student_outputs / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_outputs.detach() / self.temperature, dim=-1)
            # KLDivLoss expects input (log_probs) and target (probs)
            # The scaling factor (temperature**2) is common in distillation literature.
            return self.loss_fn(student_log_probs, teacher_probs) * (self.temperature ** 2)

        return None # Should not be reached if loss_type is validated in __init__


class BlockDistillationTrainer:
    def __init__(self,
                 teacher_model: nn.Module,
                 student_model: nn.Module,
                 tokenizer, # For potential debugging or data checks
                 train_dataloader,
                 distill_loss_type: str = "mse",
                 distill_loss_temperature: float = 1.0,
                 optimizer_cls = AdamW,
                 device: torch.device = torch.device("cpu"),
                 output_dir: str = "./distilled_model",
                 log_interval: int = 50,
                 freeze_previous_blocks: bool = True):
        """
        Initializes the BlockDistillationTrainer.

        Args:
            teacher_model: The pre-trained teacher model.
            student_model: The student model to be trained.
            tokenizer: Tokenizer used for the models.
            train_dataloader: DataLoader for the training dataset.
            distill_loss_type: Type of loss for distillation ('mse', 'kl_div').
            distill_loss_temperature: Temperature for KL divergence loss.
            optimizer_cls: The optimizer class (e.g., torch.optim.AdamW).
            device: The device to run training on ('cuda', 'cpu', 'mps').
            output_dir: Directory to save model checkpoints.
            log_interval: How often to log training progress.
            freeze_previous_blocks: If True, freezes parameters of previously distilled blocks.
        """
        self.teacher_model = teacher_model.to(device).eval() # Teacher is always in eval mode
        self.student_model = student_model.to(device)
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.loss_fn = DistillationLoss(loss_type=distill_loss_type, temperature=distill_loss_temperature)
        self.optimizer_cls = optimizer_cls
        self.device = device
        self.output_dir = output_dir
        self.log_interval = log_interval
        self.freeze_previous_blocks = freeze_previous_blocks

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")

        # Ensure student model config has output_hidden_states = True
        if not hasattr(self.student_model.config, 'output_hidden_states') or \
           not self.student_model.config.output_hidden_states:
            raise ValueError("Student model's config must have 'output_hidden_states' set to True.")
        
        # Teacher model (Hugging Face) uses 'output_hidden_states' in forward call, not config typically.

    def _get_block_parameters(self, block_idx: int) -> List[torch.nn.Parameter]:
        """
        Returns an iterator of parameters for the specified student block
        and optionally all preceding blocks if not freezing.
        Also includes embedding parameters when training the first block (block_idx == 0).
        """
        params_to_train: List[torch.nn.Parameter] = []
        
        # Parameters of the current block being distilled
        # Assuming student model has 'transformer.h' ModuleList
        if block_idx < len(self.student_model.transformer.h):
            current_block_params = list(self.student_model.transformer.h[block_idx].parameters())
            params_to_train.extend(current_block_params)
        else:
            logger.warning(f"Block index {block_idx} is out of range for student model with {len(self.student_model.transformer.h)} layers.")
            return []


        if not self.freeze_previous_blocks:
            # Add parameters from all preceding blocks
            for i in range(block_idx):
                params_to_train.extend(list(self.student_model.transformer.h[i].parameters()))
            # Also include embeddings and initial layers if not freezing anything,
            # particularly important when starting with the first block or fine-tuning everything.
            if hasattr(self.student_model.transformer, 'wte'):
                params_to_train.extend(list(self.student_model.transformer.wte.parameters()))
            if hasattr(self.student_model.transformer, 'wpe'):
                params_to_train.extend(list(self.student_model.transformer.wpe.parameters()))
        elif block_idx == 0: # If freezing, but it's the first block, still train embeddings
            if hasattr(self.student_model.transformer, 'wte'):
                params_to_train.extend(list(self.student_model.transformer.wte.parameters()))
            if hasattr(self.student_model.transformer, 'wpe'):
                params_to_train.extend(list(self.student_model.transformer.wpe.parameters()))
        
        # Filter out parameters that do not require gradients (e.g., if some were manually frozen)
        return [p for p in params_to_train if p.requires_grad]

    def distill_block(self,
                      block_idx: int,
                      num_epochs: int,
                      learning_rate: float,
                      weight_decay: float = 0.01,
                      max_grad_norm: Optional[float] = 1.0):
        """
        Distills a single block of the student model.
        """
        logger.info(f"--- Starting distillation for Block {block_idx + 1}/{self.student_model.config.n_layer} ---")
        self.student_model.train() # Set student to train mode

        # Get parameters for the current block (and preceding if not frozen)
        params_to_optimize = self._get_block_parameters(block_idx)
        if not params_to_optimize:
            logger.warning(f"No parameters to optimize for block {block_idx}. Skipping.")
            return

        optimizer = self.optimizer_cls(params_to_optimize, lr=learning_rate, weight_decay=weight_decay)

        for epoch in range(num_epochs):
            logger.info(f"Block {block_idx + 1}, Epoch {epoch + 1}/{num_epochs}")
            epoch_loss = 0.0
            num_batches_processed = 0
            
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1} Block {block_idx+1}", leave=False)

            for batch_idx_iter, batch_data in enumerate(progress_bar):
                input_ids = batch_data['input_ids'].to(self.device)
                attention_mask = batch_data.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                optimizer.zero_grad()

                # --- Teacher Forward Pass ---
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True # Request hidden states
                    )
                # teacher_outputs.hidden_states is a tuple.
                # hidden_states[0] is the embedding output.
                # hidden_states[i+1] is the output of the i-th transformer layer.
                if teacher_outputs.hidden_states is None or len(teacher_outputs.hidden_states) <= block_idx + 1:
                    raise ValueError(f"Teacher model did not return enough hidden states. "
                                     f"Requested output for block {block_idx} (index {block_idx + 1} in hidden_states list), "
                                     f"but got {len(teacher_outputs.hidden_states) if teacher_outputs.hidden_states else 0} states.")
                teacher_hidden_state = teacher_outputs.hidden_states[block_idx + 1]

                # --- Student Forward Pass ---
                # Student model (custom) returns a dict with 'hidden_states' key.
                # student_outputs['hidden_states'][0] is embedding output.
                # student_outputs['hidden_states'][i+1] is output of i-th layer.
                student_full_outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                    # output_hidden_states is controlled by student_model.config.output_hidden_states
                )
                student_hidden_states_list = student_full_outputs.get('hidden_states')
                if student_hidden_states_list is None or len(student_hidden_states_list) <= block_idx + 1:
                    raise ValueError(f"Student model did not return enough hidden states under 'hidden_states' key. "
                                     f"Requested output for block {block_idx} (index {block_idx + 1} in list), "
                                     f"got {len(student_hidden_states_list) if student_hidden_states_list else 0} states.")
                student_hidden_state_current_block = student_hidden_states_list[block_idx + 1]


                # --- Calculate Loss ---
                if student_hidden_state_current_block.size(-1) != teacher_hidden_state.size(-1):
                    raise ValueError(
                        f"Dimension mismatch for distillation at block {block_idx}: "
                        f"Student hidden_dim={student_hidden_state_current_block.size(-1)}, "
                        f"Teacher hidden_dim={teacher_hidden_state.size(-1)}. "
                        "A projection layer is likely needed for the student's hidden states."
                    )

                loss = self.loss_fn(student_hidden_state_current_block, teacher_hidden_state)

                if loss is None or torch.isnan(loss):
                    logger.error(f"Block {block_idx+1}, Epoch {epoch+1}, Batch {batch_idx_iter}: Loss is NaN or None. Skipping batch.")
                    if torch.isnan(loss):
                         logger.error("Stopping distillation due to NaN loss.")
                         # Consider saving a checkpoint here before exiting if it's a long run
                         # self.save_checkpoint(f"student_model_nan_loss_block_{block_idx}.pt")
                         raise RuntimeError("NaN loss encountered during distillation.") # Or return gracefully
                    continue

                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, max_grad_norm)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches_processed +=1
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                if batch_idx_iter % self.log_interval == 0 and batch_idx_iter > 0:
                    avg_batch_loss = epoch_loss / num_batches_processed
                    logger.info(f"Block {block_idx+1}, Epoch {epoch+1}, Batch {batch_idx_iter}: Avg Loss: {avg_batch_loss:.4f}")
            
            avg_epoch_loss = epoch_loss / num_batches_processed if num_batches_processed > 0 else float('nan')
            logger.info(f"Block {block_idx+1}, Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")

        logger.info(f"--- Finished distillation for Block {block_idx + 1} ---")

        # Save checkpoint after each block is distilled
        block_checkpoint_path = os.path.join(self.output_dir, f"student_model_block_{block_idx+1}_distilled.pt")
        self.save_checkpoint(block_checkpoint_path, block_idx=block_idx, epoch=num_epochs) # Pass more info
        logger.info(f"Saved student model checkpoint after block {block_idx+1} to {block_checkpoint_path}")

    def save_checkpoint(self, path: str, **kwargs):
        """Saves a student model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Add GPTConfig to safe globals for PyTorch 2.6+ compatibility
        try:
            from config_distillation import GPTConfig
            import torch.serialization
            if hasattr(torch.serialization, 'add_safe_globals'):
                torch.serialization.add_safe_globals([GPTConfig])
        except Exception as e:
            logger.warning(f"Could not add GPTConfig to safe globals: {e}")
            logger.warning("The saved checkpoint might have compatibility issues with PyTorch 2.6+")
        
        checkpoint = {
            'model_state_dict': self.student_model.state_dict(),
            'student_config': self.student_model.config, # Save config for easy reloading
            **kwargs # e.g., block_idx, epoch
        }
        
        # Use a lower pickle protocol for better compatibility
        torch.save(checkpoint, path, pickle_protocol=4)
        logger.info(f"Student model checkpoint saved to {path}")


    def train(self,
              epochs_per_block: int,
              lr_per_block: float, # Can be a list or a single float
              wd_per_block: float = 0.01, # Can be a list or a single float
              max_grad_norm_per_block: Optional[float] = 1.0): # Can be a list or single
        """
        Runs the full block-by-block distillation process.
        """
        num_student_layers = self.student_model.config.n_layer
        # For Hugging Face GPT2Model, config.n_layer or config.num_hidden_layers
        num_teacher_layers = self.teacher_model.config.num_hidden_layers 

        if num_student_layers != num_teacher_layers:
            logger.warning(
                f"Teacher has {num_teacher_layers} layers, Student has {num_student_layers} layers. "
                f"Distillation will proceed for {min(num_student_layers, num_teacher_layers)} layers, matching teacher layer outputs to student layer outputs sequentially."
            )
        
        n_layers_to_distill = min(num_student_layers, num_teacher_layers)
        if n_layers_to_distill == 0:
            logger.error("No layers to distill (student or teacher has 0 layers). Exiting.")
            return

        logger.info(f"Starting block-by-block distillation for {n_layers_to_distill} layers.")

        for block_idx in range(n_layers_to_distill):
            current_lr = lr_per_block[block_idx] if isinstance(lr_per_block, list) else lr_per_block
            current_wd = wd_per_block[block_idx] if isinstance(wd_per_block, list) else wd_per_block
            current_max_grad_norm = max_grad_norm_per_block[block_idx] if isinstance(max_grad_norm_per_block, list) else max_grad_norm_per_block
            
            if self.freeze_previous_blocks:
                # Freeze parameters of all blocks up to block_idx-1
                for i in range(block_idx):
                    if i < len(self.student_model.transformer.h):
                        for param in self.student_model.transformer.h[i].parameters():
                            param.requires_grad = False
                
                # Ensure current block and subsequent blocks are trainable (if they were frozen before)
                for i in range(block_idx, num_student_layers):
                    if i < len(self.student_model.transformer.h):
                         for param in self.student_model.transformer.h[i].parameters():
                            param.requires_grad = True
                
                # Manage requires_grad for embeddings based on current block_idx
                if block_idx > 0 : # If not the first block, embeddings might have been trained and should be frozen
                    if hasattr(self.student_model.transformer, 'wte'):
                        for param in self.student_model.transformer.wte.parameters(): param.requires_grad = False
                    if hasattr(self.student_model.transformer, 'wpe'):
                        for param in self.student_model.transformer.wpe.parameters(): param.requires_grad = False
                else: # First block, ensure embeddings are trainable
                    if hasattr(self.student_model.transformer, 'wte'):
                        for param in self.student_model.transformer.wte.parameters(): param.requires_grad = True
                    if hasattr(self.student_model.transformer, 'wpe'):
                        for param in self.student_model.transformer.wpe.parameters(): param.requires_grad = True
            else: # Not freezing, ensure all relevant parts are trainable
                if hasattr(self.student_model.transformer, 'wte'):
                    for param in self.student_model.transformer.wte.parameters(): param.requires_grad = True
                if hasattr(self.student_model.transformer, 'wpe'):
                    for param in self.student_model.transformer.wpe.parameters(): param.requires_grad = True
                for i in range(num_student_layers):
                    if i < len(self.student_model.transformer.h):
                        for param in self.student_model.transformer.h[i].parameters():
                            param.requires_grad = True


            self.distill_block(
                block_idx,
                num_epochs=epochs_per_block,
                learning_rate=current_lr,
                weight_decay=current_wd,
                max_grad_norm=current_max_grad_norm
            )

        final_model_path = os.path.join(self.output_dir, "student_model_final_distilled.pt")
        self.save_checkpoint(final_model_path, status="final_distillation_complete")
        logger.info(f"Full distillation complete. Final student model saved to {final_model_path}")
        logger.info("To use the distilled model, load its state_dict into an instance of the student model class, potentially using the saved config.")
