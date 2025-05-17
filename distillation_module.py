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
from typing import Optional, Dict, Any, List, Union

# Added imports for stitching layers
from stitching_layers import StitchingLayer, StitchingDistillationLoss 

logger = logging.getLogger(__name__)

class DistillationLoss(nn.Module):
    """
    Wrapper for various distillation loss functions.
    This is used when stitching layers are NOT active.
    """
    def __init__(self, loss_type="mse", temperature=1.0):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.temperature = temperature
        if self.loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif self.loss_type == "kl_div":
            self.loss_fn = nn.KLDivLoss(reduction='batchmean') 
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}. Supported: 'mse', 'kl_div'.")

    def forward(self, student_outputs, teacher_outputs):
        # This loss is used when no stitching layer is present.
        # A direct comparison implies dimensions should match or an error will occur.
        if student_outputs.shape != teacher_outputs.shape:
             logger.error(
                 f"Standard DistillationLoss: Shape mismatch between student ({student_outputs.shape}) "
                 f"and teacher ({teacher_outputs.shape}). This will likely cause an error. "
                 f"Ensure dimensions match or use stitching layers for projection."
            )
             # Depending on the loss_fn, this might still raise an error.
             # For MSE, it will definitely error if shapes are not broadcastable/identical.

        if self.loss_type == "mse":
            return self.loss_fn(student_outputs, teacher_outputs.detach())
        elif self.loss_type == "kl_div":
            student_log_probs = F.log_softmax(student_outputs / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_outputs.detach() / self.temperature, dim=-1)
            return self.loss_fn(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        raise ValueError(f"Loss calculation failed for loss type {self.loss_type}")


class BlockDistillationTrainer:
    def __init__(self,
                 teacher_model: nn.Module,
                 student_model: nn.Module,
                 tokenizer, 
                 train_dataloader,
                 distill_loss_type: str = "mse",
                 distill_loss_temperature: float = 1.0,
                 use_stitching_layers: bool = True,
                 stitching_layer_bias: bool = True,
                 optimizer_cls = AdamW,
                 device: torch.device = torch.device("cpu"),
                 output_dir: str = "./distilled_model",
                 log_interval: int = 50,
                 freeze_previous_blocks: bool = True):
        """
        Initializes the BlockDistillationTrainer.
        """
        self.teacher_model = teacher_model.to(device).eval() 
        self.student_model = student_model.to(device)
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.use_stitching_layers = use_stitching_layers
        self.device = device
        self.output_dir = output_dir
        self.log_interval = log_interval
        self.freeze_previous_blocks = freeze_previous_blocks
        self.optimizer_cls = optimizer_cls

        if self.use_stitching_layers:
            logger.info("Using Stitching Layers for distillation.")
            try:
                # Dimension of the teacher's hidden states (target for stitching layer output)
                teacher_actual_dim = teacher_model.config.hidden_size 

                # Dimension of the student's hidden states *as they are output by the student model's blocks*
                # This is the input dimension for the stitching layer.
                # student_model.config.teacher_n_embd should store the dimension of the student's
                # hidden states after any internal projection within the student model itself.
                # This was indicated by logs: "Added hidden state projection from 384 to 768",
                # where 768 would be student_model.config.teacher_n_embd.
                student_hidden_dim_input_to_stitching = student_model.config.teacher_n_embd

                if student_hidden_dim_input_to_stitching is None:
                    logger.warning(
                        "student_model.config.teacher_n_embd is None. This field is expected to store the "
                        "dimension of the student's hidden states that will be fed into the stitching layer. "
                        "This might be the student's native n_embd or a dimension it projects to. "
                        "Falling back to teacher_model.config.hidden_size, assuming student already projects to teacher's dim."
                    )
                    # This fallback assumes the student's output already matches the teacher's dimension.
                    student_hidden_dim_input_to_stitching = teacher_actual_dim
                
                if student_hidden_dim_input_to_stitching is None: # Should not happen if teacher_model.config.hidden_size exists
                     logger.error("Cannot determine student output hidden dimension for stitching layer input. "
                                  "Please check teacher_model.config.hidden_size and student_model.config.teacher_n_embd.")
                     raise ValueError("Cannot determine student output hidden dimension for stitching layer input.")

                logger.info(f"StitchingDistillationLoss will be configured for student hidden states of dim (input to stitch): {student_hidden_dim_input_to_stitching}")
                logger.info(f"StitchingDistillationLoss will project student states to teacher's dim (output of stitch): {teacher_actual_dim}")

                num_student_layers = student_model.config.n_layer
                
                self.loss_fn = StitchingDistillationLoss(
                    loss_type=distill_loss_type,
                    temperature=distill_loss_temperature,
                    student_dims=student_hidden_dim_input_to_stitching, # Input dim for EACH stitching layer
                    teacher_dims=teacher_actual_dim,                   # Output dim for EACH stitching layer
                    use_bias=stitching_layer_bias
                ).to(device)

            except AttributeError as e:
                logger.error(f"Could not access required model config attributes (e.g., hidden_size, teacher_n_embd, n_layer): {e}")
                logger.info("Falling back to dynamic stitching layer creation within StitchingDistillationLoss "
                            "(dimensions will be inferred at runtime, which is less safe for pre-optimization).")
                self.loss_fn = StitchingDistillationLoss(
                    loss_type=distill_loss_type,
                    temperature=distill_loss_temperature,
                    use_bias=stitching_layer_bias
                    # student_dims, teacher_dims, num_layers left None for fully dynamic creation
                ).to(device)
        else:
            logger.info("Using standard Distillation Loss (no stitching layers).")
            self.loss_fn = DistillationLoss(
                loss_type=distill_loss_type, 
                temperature=distill_loss_temperature
            ).to(device)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")

        # Ensure student model is configured to output hidden states
        if not hasattr(self.student_model.config, 'output_hidden_states') or \
           not self.student_model.config.output_hidden_states:
            logger.warning("Student model's config.output_hidden_states is False. Setting to True for distillation.")
            self.student_model.config.output_hidden_states = True


    def _get_block_parameters(self, block_idx: int) -> List[torch.nn.Parameter]:
        params_to_train: List[torch.nn.Parameter] = []
        
        # Add parameters of the current student transformer block
        if block_idx < len(self.student_model.transformer.h):
            current_block_params = list(self.student_model.transformer.h[block_idx].parameters())
            params_to_train.extend(current_block_params)
            logger.debug(f"Added {len(current_block_params)} params from student block {block_idx}.")
        else:
            logger.warning(f"Block index {block_idx} is out of range for student model with {len(self.student_model.transformer.h)} layers.")

        # Add parameters from preceding blocks if not freezing
        if not self.freeze_previous_blocks:
            for i in range(block_idx): 
                if i < len(self.student_model.transformer.h):
                    logger.debug(f"Not freezing: Adding params from student block {i}.")
                    params_to_train.extend(list(self.student_model.transformer.h[i].parameters()))
            if hasattr(self.student_model.transformer, 'wte'): # Token embeddings
                logger.debug("Not freezing: Adding wte params.")
                params_to_train.extend(list(self.student_model.transformer.wte.parameters()))
            if hasattr(self.student_model.transformer, 'wpe'): # Positional embeddings
                logger.debug("Not freezing: Adding wpe params.")
                params_to_train.extend(list(self.student_model.transformer.wpe.parameters()))
        elif block_idx == 0: # If freezing, but it's the first block, still train embeddings
            if hasattr(self.student_model.transformer, 'wte'):
                logger.debug("Block 0 (freezing active): Adding wte params.")
                params_to_train.extend(list(self.student_model.transformer.wte.parameters()))
            if hasattr(self.student_model.transformer, 'wpe'):
                logger.debug("Block 0 (freezing active): Adding wpe params.")
                params_to_train.extend(list(self.student_model.transformer.wpe.parameters()))
        
        # Add parameters of the stitching layer for the current block
        if self.use_stitching_layers and hasattr(self.loss_fn, 'stitching_layers'):
            layer_key = str(block_idx)
            # Attempt to get/create the layer to ensure its parameters are available for the optimizer.
            # This is crucial if layers are not pre-created or if dynamic creation is relied upon.
            if hasattr(self.loss_fn, '_get_stitching_layer'):
                try:
                    # Determine the dimensions the stitching layer for this block_idx will use.
                    # These must match the dimensions used in the forward pass and __init__.
                    teacher_actual_dim = self.teacher_model.config.hidden_size
                    student_hidden_dim_input_to_stitching = self.student_model.config.teacher_n_embd
                    if student_hidden_dim_input_to_stitching is None:
                        student_hidden_dim_input_to_stitching = teacher_actual_dim # Fallback
                    
                    # This call will create the layer if it doesn't exist or return the existing one.
                    # It's important that student_hidden_dim_input_to_stitching and teacher_actual_dim are correct.
                    self.loss_fn._get_stitching_layer(block_idx, student_hidden_dim_input_to_stitching, teacher_actual_dim)
                except Exception as e:
                    logger.warning(f"Error trying to ensure stitching layer {block_idx} exists for optimizer: {e}")

            if layer_key in self.loss_fn.stitching_layers:
                stitching_params = list(self.loss_fn.stitching_layers[layer_key].parameters())
                params_to_train.extend(stitching_params)
                logger.debug(f"Added {len(stitching_params)} parameters from stitching layer {block_idx} for training.")
            else:
                logger.warning(f"Stitching layer for block {block_idx} not found in loss_fn.stitching_layers "
                               f"when setting up optimizer. If created dynamically during forward pass, "
                               f"its parameters might not be optimized in the first step of this block.")


        # Filter for unique parameters that require gradients
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
        logger.info(f"--- Starting distillation for Block {block_idx + 1}/{self.student_model.config.n_layer} ---")
        self.student_model.train() # Ensure student model is in training mode

        # Set requires_grad for stitching layers based on freezing strategy for the current block
        # This ensures that only the relevant stitching layer parameters are updated.
        if self.use_stitching_layers and hasattr(self.loss_fn, 'stitching_layers'):
            for s_layer_idx_str, s_layer_module in self.loss_fn.stitching_layers.items():
                s_layer_idx = int(s_layer_idx_str)
                # A stitching layer's parameters should be trainable if it corresponds to the current block_idx
                # OR if we are not freezing previous blocks (in which case all are trainable).
                is_current_block_stitching_layer = (s_layer_idx == block_idx)
                should_train_stitching_layer_params = is_current_block_stitching_layer or not self.freeze_previous_blocks
                
                for param in s_layer_module.parameters():
                    param.requires_grad = should_train_stitching_layer_params
                
                if should_train_stitching_layer_params:
                     logger.debug(f"Stitching layer {s_layer_idx} parameters set to trainable for block {block_idx} distillation.")
                # else: # If not training, they should remain frozen (requires_grad=False)
                #      logger.debug(f"Stitching layer {s_layer_idx} parameters remain frozen for block {block_idx} distillation.")


        params_to_optimize = self._get_block_parameters(block_idx) # Get parameters after setting requires_grad
        if not params_to_optimize:
            logger.warning(f"No parameters to optimize for block {block_idx} after requires_grad checks. Skipping.")
            block_checkpoint_path = os.path.join(self.output_dir, f"student_model_block_{block_idx+1}_skipped_no_params.pt")
            self.save_checkpoint(block_checkpoint_path, block_idx=block_idx, epoch=0, status="skipped_no_trainable_params")
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
                    teacher_outputs_obj = self.teacher_model( 
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True # Request hidden states
                    )
                # teacher_outputs_obj.hidden_states is a tuple.
                # hidden_states[0] is the embedding output.
                # hidden_states[i+1] is the output of the i-th transformer layer.
                if teacher_outputs_obj.hidden_states is None or len(teacher_outputs_obj.hidden_states) <= block_idx + 1:
                    raise ValueError(f"Teacher model did not return enough hidden states. "
                                     f"Requested output for block {block_idx} (idx {block_idx + 1} in hidden_states list), "
                                     f"but got {len(teacher_outputs_obj.hidden_states) if teacher_outputs_obj.hidden_states else 0} states.")
                teacher_hidden_state = teacher_outputs_obj.hidden_states[block_idx + 1] 

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
                                     f"Requested output for block {block_idx} (idx {block_idx + 1} in list), "
                                     f"got {len(student_hidden_states_list) if student_hidden_states_list else 0} states.")
                student_hidden_state_current_block = student_hidden_states_list[block_idx + 1]
                
                # --- Calculate Loss ---
                if not self.use_stitching_layers and student_hidden_state_current_block.size(-1) != teacher_hidden_state.size(-1):
                    raise ValueError(
                        f"Dimension mismatch for distillation at block {block_idx} without stitching layers: "
                        f"Student hidden_dim={student_hidden_state_current_block.size(-1)}, "
                        f"Teacher hidden_dim={teacher_hidden_state.size(-1)}. "
                        "Enable stitching layers or ensure student model projects its hidden states to match teacher's."
                    )

                if self.use_stitching_layers:
                    # StitchingDistillationLoss's forward takes (student_outputs, teacher_outputs, layer_idx)
                    loss = self.loss_fn(student_hidden_state_current_block, teacher_hidden_state, layer_idx=block_idx)
                else:
                    # Standard DistillationLoss's forward takes (student_outputs, teacher_outputs)
                    loss = self.loss_fn(student_hidden_state_current_block, teacher_hidden_state)

                # --- Error Checking and Backward Pass ---
                if loss is None or torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"Block {block_idx+1}, Epoch {epoch+1}, Batch {batch_idx_iter}: Loss is {loss}. Skipping batch.")
                    if torch.isnan(loss) or torch.isinf(loss): # Critical error
                         logger.error(f"Stopping distillation due to {loss} loss.")
                         # Consider saving a checkpoint before raising error for long runs
                         # self.save_checkpoint(os.path.join(self.output_dir, f"student_model_error_loss_block_{block_idx}.pt"), error_loss=float(loss))
                         raise RuntimeError(f"{loss} loss encountered during distillation.")
                    continue # Skip batch if loss is None but not NaN/Inf (e.g. if loss can be legitimately None)

                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, max_grad_norm)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches_processed +=1
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                if batch_idx_iter > 0 and batch_idx_iter % self.log_interval == 0 :
                    avg_batch_loss = epoch_loss / num_batches_processed if num_batches_processed > 0 else float('nan')
                    logger.info(f"Block {block_idx+1}, Epoch {epoch+1}, Batch {batch_idx_iter}/{len(self.train_dataloader)}: Avg Loss: {avg_batch_loss:.4f}, Current Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / num_batches_processed if num_batches_processed > 0 else float('nan')
            logger.info(f"Block {block_idx+1}, Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")

        logger.info(f"--- Finished distillation for Block {block_idx + 1} ---")

        # Save checkpoint after each block is distilled
        block_checkpoint_path = os.path.join(self.output_dir, f"student_model_block_{block_idx+1}_distilled.pt")
        self.save_checkpoint(block_checkpoint_path, block_idx=block_idx, epoch=num_epochs, avg_loss=avg_epoch_loss)
        logger.info(f"Saved student model checkpoint after block {block_idx+1} to {block_checkpoint_path}")

    def save_checkpoint(self, path: str, **kwargs):
        """Saves a student model checkpoint, including stitching layers if used."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Attempt to make GPTConfig from config_distillation.py safe for PyTorch 2.6+ pickling
        try:
            from config_distillation import GPTConfig 
            import torch.serialization
            if hasattr(torch.serialization, 'add_safe_globals'):
                torch.serialization.add_safe_globals([GPTConfig])
        except ImportError:
            logger.warning("Could not import GPTConfig from config_distillation. GPTConfig will not be added to safe globals for checkpointing.")
        except Exception as e: 
            logger.warning(f"Could not add GPTConfig to safe globals: {e}")
        
        checkpoint = {
            'model_state_dict': self.student_model.state_dict(),
            'student_config': self.student_model.config, 
            **kwargs 
        }
        
        # Save stitching layers' state_dict if they exist and are used
        if self.use_stitching_layers and hasattr(self.loss_fn, 'stitching_layers') and len(self.loss_fn.stitching_layers) > 0:
            checkpoint['stitching_layers_state_dict'] = self.loss_fn.stitching_layers.state_dict()
            checkpoint['use_stitching_layers'] = True # Explicitly save this flag
            logger.info("Saved stitching layers' state_dict to checkpoint.")
        
        torch.save(checkpoint, path, pickle_protocol=4) 
        logger.info(f"Student model checkpoint saved to {path}")


    def train(self,
              epochs_per_block: int,
              lr_per_block: Union[float, List[float]], 
              wd_per_block: Union[float, List[float]] = 0.01, 
              max_grad_norm_per_block: Optional[Union[float, List[float]]] = 1.0):
        num_student_layers = self.student_model.config.n_layer
        num_teacher_layers = self.teacher_model.config.num_hidden_layers 

        if num_student_layers != num_teacher_layers:
            logger.warning(
                f"Teacher has {num_teacher_layers} layers, Student has {num_student_layers} layers. "
                f"Distillation will proceed for {min(num_student_layers, num_teacher_layers)} layers."
            )
        
        n_layers_to_distill = min(num_student_layers, num_teacher_layers)
        if n_layers_to_distill == 0:
            logger.error("No layers to distill (student or teacher has 0 layers). Exiting.")
            return

        logger.info(f"Starting block-by-block distillation for {n_layers_to_distill} layers.")

        for block_idx in range(n_layers_to_distill):
            # Determine current learning rate, weight decay, and max_grad_norm for the block
            current_lr = lr_per_block[block_idx] if isinstance(lr_per_block, list) else lr_per_block
            current_wd = wd_per_block[block_idx] if isinstance(wd_per_block, list) else wd_per_block
            current_max_grad_norm = max_grad_norm_per_block[block_idx] if isinstance(max_grad_norm_per_block, list) else max_grad_norm_per_block
            
            # --- Parameter Freezing Logic for Student Model (Transformer Blocks & Embeddings) ---
            if self.freeze_previous_blocks:
                # Freeze parameters of all student transformer blocks up to block_idx-1
                for i in range(block_idx):
                    if i < len(self.student_model.transformer.h): # Check if layer index is valid
                        for param in self.student_model.transformer.h[i].parameters():
                            param.requires_grad = False
                
                # Ensure the current block being distilled is trainable.
                if block_idx < len(self.student_model.transformer.h):
                     for param in self.student_model.transformer.h[block_idx].parameters():
                        param.requires_grad = True
                # Subsequent blocks (if any) will have their requires_grad status handled when their turn comes.
                
                # Manage requires_grad for embeddings: train with first block, then freeze.
                train_embeddings = (block_idx == 0)
                if hasattr(self.student_model.transformer, 'wte'):
                    for param in self.student_model.transformer.wte.parameters(): param.requires_grad = train_embeddings
                if hasattr(self.student_model.transformer, 'wpe'):
                    for param in self.student_model.transformer.wpe.parameters(): param.requires_grad = train_embeddings
                logger.debug(f"Block {block_idx} (freeze_previous_blocks=True): Student transformer block {block_idx} trainable. Embeddings trainable: {train_embeddings}")

            else: # Not freezing previous blocks: all student model parameters should be trainable.
                if hasattr(self.student_model.transformer, 'wte'):
                    for param in self.student_model.transformer.wte.parameters(): param.requires_grad = True
                if hasattr(self.student_model.transformer, 'wpe'):
                    for param in self.student_model.transformer.wpe.parameters(): param.requires_grad = True
                for i in range(num_student_layers): # All student transformer blocks
                    if i < len(self.student_model.transformer.h):
                        for param in self.student_model.transformer.h[i].parameters():
                            param.requires_grad = True
                logger.debug(f"Block {block_idx} (freeze_previous_blocks=False): All student model parameters (blocks & embeddings) set to trainable.")
            
            # Note: requires_grad for stitching layers is handled inside self.distill_block()
            # right before the optimizer for that block is created. This ensures that
            # _get_block_parameters() correctly identifies which stitching layer parameters
            # should be included in the optimizer for the current block.

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


