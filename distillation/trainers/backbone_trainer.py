# ./distillation/trainers/backbone_trainer.py
"""
Trainer for the transformer backbone (block-by-block distillation).
"""
import os
import logging
import torch
from tqdm.auto import tqdm
from typing import List, Optional, Union

# Import from the distillation module
from distillation.trainers.base_trainer import BaseDistillationTrainer
from distillation.losses.hidden_state_loss import DistillationLoss

logger = logging.getLogger(__name__)

# Try to import stitching layer if available
try:
    from stitching_layers import StitchingLayer, StitchingDistillationLoss
    STITCHING_AVAILABLE = True
except ImportError:
    logger.warning("stitching_layers module not found. Stitching-related features will not work.")
    STITCHING_AVAILABLE = False

class BackboneDistillationTrainer(BaseDistillationTrainer):
    """Trainer for block-by-block distillation of transformer backbones."""
    
    def __init__(self,
                 teacher_model,
                 student_model,
                 tokenizer,
                 train_dataloader,
                 distill_loss_type="mse",
                 distill_loss_temperature=1.0,
                 use_stitching_layers=True,
                 stitching_layer_bias=True,
                 freeze_previous_blocks=True,
                 device=torch.device("cpu"),
                 output_dir="./distilled_model",
                 log_interval=50,
                 optimizer_cls=torch.optim.AdamW):
        """
        Initialize the backbone distillation trainer.
        
        Args:
            teacher_model: Teacher model to distill from
            student_model: Student model to train
            tokenizer: Tokenizer for processing text
            train_dataloader: DataLoader for training examples
            distill_loss_type: Type of distillation loss ("mse" or "kl_div")
            distill_loss_temperature: Temperature for distillation loss
            use_stitching_layers: Whether to use stitching layers
            stitching_layer_bias: Whether to use bias in stitching layers
            freeze_previous_blocks: Whether to freeze previously trained blocks
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
        
        self.use_stitching_layers = use_stitching_layers
        self.freeze_previous_blocks = freeze_previous_blocks
        
        # Configure hidden state loss
        if self.use_stitching_layers:
            if not STITCHING_AVAILABLE:
                raise ImportError("StitchingDistillationLoss is not available. Cannot use stitching layers.")
            
            logger.info("Using Stitching Layers for hidden state distillation.")
            try:
                teacher_actual_dim = teacher_model.config.hidden_size
                # Prefer teacher_n_embd from student config if available, otherwise fallback
                student_hidden_dim_input_to_stitching = getattr(student_model.config, 'teacher_n_embd', None)

                if student_hidden_dim_input_to_stitching is None:
                    logger.warning(
                        "student_model.config.teacher_n_embd is None. Falling back to "
                        "teacher_model.config.hidden_size for student input to stitching."
                    )
                    student_hidden_dim_input_to_stitching = teacher_actual_dim
                
                if student_hidden_dim_input_to_stitching is None:
                    raise ValueError("Cannot determine student output hidden dimension for stitching layer input. "
                                    "Check teacher_model.config.hidden_size and student_model.config.teacher_n_embd.")

                logger.info(f"StitchingDistillationLoss configured for student dim (input to stitch): "
                           f"{student_hidden_dim_input_to_stitching}")
                logger.info(f"StitchingDistillationLoss will project to teacher's dim (output): {teacher_actual_dim}")

                self.hidden_state_loss_fn = StitchingDistillationLoss(
                    loss_type=distill_loss_type,
                    temperature=distill_loss_temperature,
                    student_dims=student_hidden_dim_input_to_stitching,
                    teacher_dims=teacher_actual_dim,
                    use_bias=stitching_layer_bias
                ).to(device)

            except AttributeError as e:
                logger.error(f"Could not access required model config attributes: {e}")
                logger.info("Falling back to dynamic stitching layer creation.")
                self.hidden_state_loss_fn = StitchingDistillationLoss(
                    loss_type=distill_loss_type,
                    temperature=distill_loss_temperature,
                    use_bias=stitching_layer_bias
                ).to(device)
        else:
            logger.info("Using standard DistillationLoss (no stitching layers) for hidden states.")
            self.hidden_state_loss_fn = DistillationLoss(
                loss_type=distill_loss_type,
                temperature=distill_loss_temperature
            ).to(device)

        # Ensure hidden states output is enabled
        if not hasattr(self.student_model.config, 'output_hidden_states') or \
            not self.student_model.config.output_hidden_states:
            logger.warning("Setting student model config.output_hidden_states to True for distillation.")
            self.student_model.config.output_hidden_states = True
        
        if hasattr(self.teacher_model.config, 'output_hidden_states') and \
            not self.teacher_model.config.output_hidden_states:
            logger.warning("Setting teacher model config.output_hidden_states to True for distillation.")
            self.teacher_model.config.output_hidden_states = True
            
    def _get_block_parameters(self, block_idx: int) -> List[torch.nn.Parameter]:
        """
        Get parameters of the specified transformer block and optionally stitching layer.
        
        Args:
            block_idx: Index of the transformer block to get parameters for
        
        Returns:
            List of parameters to optimize
        """
        params_to_train = []
        
        if block_idx < len(self.student_model.transformer.h):
            current_block_params = list(self.student_model.transformer.h[block_idx].parameters())
            params_to_train.extend(current_block_params)
            logger.debug(f"Added {len(current_block_params)} params from student block {block_idx}.")
        else:
            logger.warning(f"Block index {block_idx} is out of range for student model with "
                          f"{len(self.student_model.transformer.h)} layers.")

        if not self.freeze_previous_blocks:
            for i in range(block_idx):
                if i < len(self.student_model.transformer.h):
                    params_to_train.extend(list(self.student_model.transformer.h[i].parameters()))
            if hasattr(self.student_model.transformer, 'wte'):
                params_to_train.extend(list(self.student_model.transformer.wte.parameters()))
            if hasattr(self.student_model.transformer, 'wpe'):
                params_to_train.extend(list(self.student_model.transformer.wpe.parameters()))
        elif block_idx == 0: 
            if hasattr(self.student_model.transformer, 'wte'):
                params_to_train.extend(list(self.student_model.transformer.wte.parameters()))
            if hasattr(self.student_model.transformer, 'wpe'):
                params_to_train.extend(list(self.student_model.transformer.wpe.parameters()))
        
        if self.use_stitching_layers and hasattr(self.hidden_state_loss_fn, 'stitching_layers'):
            layer_key = str(block_idx)
            if hasattr(self.hidden_state_loss_fn, '_get_stitching_layer'): 
                try:
                    teacher_actual_dim = self.teacher_model.config.hidden_size
                    student_hidden_dim_input_to_stitching = getattr(
                        self.student_model.config, 
                        'teacher_n_embd', 
                        teacher_actual_dim
                    )
                    self.hidden_state_loss_fn._get_stitching_layer(
                        block_idx, 
                        student_hidden_dim_input_to_stitching, 
                        teacher_actual_dim
                    )
                except Exception as e:
                    logger.warning(f"Error ensuring stitching layer {block_idx} exists: {e}")
            
            if layer_key in self.hidden_state_loss_fn.stitching_layers:
                stitching_params = list(self.hidden_state_loss_fn.stitching_layers[layer_key].parameters())
                params_to_train.extend(stitching_params)
                logger.debug(f"Added {len(stitching_params)} params from stitching layer {block_idx}.")
            else:
                available_keys = list(self.hidden_state_loss_fn.stitching_layers.keys()) \
                                if hasattr(self.hidden_state_loss_fn, 'stitching_layers') else 'N/A'
                logger.warning(f"Stitching layer {block_idx} not found. Available keys: {available_keys}")

        # Deduplicate parameters
        unique_params_to_train = []
        seen_params_ids = set()
        for p in params_to_train:
            if p.requires_grad and id(p) not in seen_params_ids:
                unique_params_to_train.append(p)
                seen_params_ids.add(id(p))
        
        logger.info(f"Optimizing {len(unique_params_to_train)} parameters for block {block_idx}.")
        return unique_params_to_train
        
    def distill_block(self,
                      block_idx: int,
                      num_epochs: int,
                      learning_rate: float,
                      weight_decay: float = 0.01,
                      max_grad_norm: Optional[float] = 1.0):
        """
        Distill a single transformer block.
        
        Args:
            block_idx: Index of the block to distill
            num_epochs: Number of epochs to train
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            max_grad_norm: Maximum gradient norm for clipping
        
        Returns:
            Average loss for the block
        """
        logger.info(f"--- Starting distillation for Block {block_idx + 1}/{self.student_model.config.n_layer} ---")
        self.student_model.train()

        # Configure stitching layers if present
        if self.use_stitching_layers and hasattr(self.hidden_state_loss_fn, 'stitching_layers'):
            for s_layer_idx_str, s_layer_module in self.hidden_state_loss_fn.stitching_layers.items():
                s_layer_idx = int(s_layer_idx_str)
                should_train = (s_layer_idx == block_idx) or not self.freeze_previous_blocks
                for param in s_layer_module.parameters():
                    param.requires_grad = should_train
                if should_train:
                    logger.debug(f"Stitching layer {s_layer_idx} parameters set to trainable for block {block_idx}.")

        # Get parameters to optimize
        params_to_optimize = self._get_block_parameters(block_idx)
        if not params_to_optimize:
            logger.warning(f"No parameters to optimize for block {block_idx}. Skipping.")
            self.save_checkpoint(
                os.path.join(self.output_dir, f"student_model_block_{block_idx+1}_skipped_no_params.pt"),
                block_idx=block_idx, epoch=0, status="skipped_no_trainable_params"
            )
            return float('nan')

        # Create optimizer
        optimizer = self.optimizer_cls(params_to_optimize, lr=learning_rate, weight_decay=weight_decay)

        # Train for specified number of epochs
        final_avg_loss = float('nan')
        for epoch in range(num_epochs):
            logger.info(f"Block {block_idx + 1}, Epoch {epoch + 1}/{num_epochs}")
            
            epoch_loss_sum = 0.0
            num_batches_processed = 0
            progress_bar = tqdm(
                self.train_dataloader, 
                desc=f"Epoch {epoch+1} Block {block_idx+1}", 
                leave=False
            )

            for batch_idx_iter, batch_data in enumerate(progress_bar):
                input_ids = batch_data['input_ids'].to(self.device)
                attention_mask = batch_data.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                optimizer.zero_grad()

                # Get teacher hidden states
                with torch.no_grad():
                    teacher_outputs_obj = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                if not hasattr(teacher_outputs_obj, 'hidden_states') or teacher_outputs_obj.hidden_states is None or len(teacher_outputs_obj.hidden_states) <= block_idx + 1:
                    raise ValueError(f"Teacher model did not return enough hidden states for block {block_idx}.")
                teacher_hidden_state = teacher_outputs_obj.hidden_states[block_idx + 1]

                # Get student hidden states
                student_full_outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                student_hidden_states_list = student_full_outputs.get('hidden_states')
                if student_hidden_states_list is None or len(student_hidden_states_list) <= block_idx + 1:
                    raise ValueError(f"Student model did not return enough hidden states for block {block_idx}.")
                student_hidden_state_current_block = student_hidden_states_list[block_idx + 1]
                
                # Check dimensions if not using stitching
                if not self.use_stitching_layers and student_hidden_state_current_block.size(-1) != teacher_hidden_state.size(-1):
                    raise ValueError(
                        f"Dimension mismatch at block {block_idx} without stitching: "
                        f"Student_dim={student_hidden_state_current_block.size(-1)}, "
                        f"Teacher_dim={teacher_hidden_state.size(-1)}."
                    )

                # Compute loss
                if self.use_stitching_layers:
                    loss = self.hidden_state_loss_fn(
                        student_hidden_state_current_block, 
                        teacher_hidden_state, 
                        layer_idx=block_idx
                    )
                else:
                    loss = self.hidden_state_loss_fn(
                        student_hidden_state_current_block, 
                        teacher_hidden_state
                    )

                # Skip problematic losses
                if loss is None or torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"Block {block_idx+1}, Epoch {epoch+1}, Batch {batch_idx_iter}: "
                                 f"Loss is {loss}. Skipping batch.")
                    if torch.isnan(loss) or torch.isinf(loss):
                         self.save_checkpoint(
                             os.path.join(self.output_dir, f"student_model_error_loss_block_{block_idx}.pt"), 
                             error_loss=float(loss.item() if loss is not None else float('nan'))
                         )
                         raise RuntimeError(f"Fatal {loss} loss encountered during block distillation.")
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
                   logger.info(f"Block {block_idx+1}, Epoch {epoch+1}, Batch {batch_idx_iter}/"
                              f"{len(self.train_dataloader)}: Avg Loss: {current_avg_loss:.4f}, "
                              f"Current Loss: {loss.item():.4f}")

            avg_epoch_loss = epoch_loss_sum / num_batches_processed if num_batches_processed > 0 else float('nan')
            logger.info(f"Block {block_idx+1}, Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
            final_avg_loss = avg_epoch_loss

        logger.info(f"--- Finished distillation for Block {block_idx + 1} ---")
        block_checkpoint_path = os.path.join(self.output_dir, f"student_model_block_{block_idx+1}_distilled.pt")
        self.save_checkpoint(block_checkpoint_path, block_idx=block_idx, epoch=num_epochs, avg_loss=final_avg_loss)
        logger.info(f"Saved student model checkpoint after block {block_idx+1} to {block_checkpoint_path}")

        return final_avg_loss

    def train_backbone(self,
                       epochs_per_block,
                       lr_per_block,
                       wd_per_block=0.01,
                       max_grad_norm_per_block=1.0):
        """
        Train the student backbone block by block.
 
        Args:
            epochs_per_block: Number of epochs to train each block
            lr_per_block: Learning rate for each block (float or list)
            wd_per_block: Weight decay for each block (float or list)
            max_grad_norm_per_block: Max gradient norm for each block (float or list)
 
        Returns:
            List of average losses for each block
        """
        num_student_layers = self.student_model.config.n_layer
        num_teacher_layers = getattr(
            self.teacher_model.config,
            'n_layer',
            getattr(self.teacher_model.config, 'num_hidden_layers', 0)
        )
 
        if num_student_layers != num_teacher_layers:
            logger.warning(
                f"Teacher has {num_teacher_layers} layers, Student has {num_student_layers} layers. "
                f"Distillation will proceed for {min(num_student_layers, num_teacher_layers)} layers."
            )
 
        n_layers_to_distill = min(num_student_layers, num_teacher_layers)
        if n_layers_to_distill == 0:
            logger.error("No layers to distill (student or teacher has 0 layers). Exiting.")
            return []
 
        logger.info(f"Starting block-by-block distillation for {n_layers_to_distill} layers.")
 
        block_losses = []
        for block_idx in range(n_layers_to_distill):
            # Get parameters for this block
            current_lr = lr_per_block[block_idx] if isinstance(lr_per_block, list) else lr_per_block
            current_wd = wd_per_block[block_idx] if isinstance(wd_per_block, list) else wd_per_block
            current_max_grad_norm = max_grad_norm_per_block[block_idx] if isinstance(max_grad_norm_per_block, list) else max_grad_norm_per_block
 
            if self.freeze_previous_blocks:
                # Freeze all transformer layers first
                for i in range(len(self.student_model.transformer.h)):
                    for param in self.student_model.transformer.h[i].parameters():
                        param.requires_grad = False
                # Unfreeze current block
                if block_idx < len(self.student_model.transformer.h):
                    for param in self.student_model.transformer.h[block_idx].parameters():
                        param.requires_grad = True
 
                train_embeddings = (block_idx == 0)
                if hasattr(self.student_model.transformer, 'wte'):
                    for param in self.student_model.transformer.wte.parameters(): param.requires_grad = train_embeddings
                if hasattr(self.student_model.transformer, 'wpe'):
                    for param in self.student_model.transformer.wpe.parameters(): param.requires_grad = train_embeddings
                logger.debug(f"Block {block_idx} (freezing): Student block {block_idx} trainable. "
                            f"Embeddings trainable: {train_embeddings}")
            else:
                for param_group in self.student_model.parameters():
                    param_group.requires_grad = True # Ensure all student params are trainable
                logger.debug(f"Block {block_idx} (not freezing): All student model parameters set to trainable.")
 
            # Distill current block
            avg_loss = self.distill_block(
                block_idx,
                num_epochs=epochs_per_block,
                learning_rate=current_lr,
                weight_decay=current_wd,
                max_grad_norm=current_max_grad_norm
            )
            block_losses.append(avg_loss)
 
        return block_losses
