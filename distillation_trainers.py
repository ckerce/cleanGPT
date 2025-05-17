# ./distillation_trainers.py
"""
Module containing the BlockDistillationTrainer class for transformer model distillation.
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm.auto import tqdm
import logging
import os
import math
from typing import Optional, Dict, Any, List, Union

# Import loss functions from the distillation_loss module using a direct import
try:
    from distillation_loss import DistillationLoss, LogitDistillationLoss
except ImportError as e:
    logging.error(f"Failed to import from distillation_loss: {e}. Ensure distillation_loss.py is accessible.")
    # Define placeholders to prevent immediate crash, though functionality will be broken.
    DistillationLoss = None
    LogitDistillationLoss = None


# Assuming stitching_layers.py is in the same directory or accessible in PYTHONPATH
try:
    from stitching_layers import StitchingLayer, StitchingDistillationLoss
except ImportError:
    logging.warning("stitching_layers module not found. Stitching-related features will not work.")
    StitchingLayer = None 
    StitchingDistillationLoss = None 

logger = logging.getLogger(__name__)

class BlockDistillationTrainer:
    def __init__(self,
                 teacher_model: nn.Module,
                 student_model: nn.Module,
                 tokenizer,
                 train_dataloader,
                 distill_loss_type: str = "mse",
                 distill_loss_temperature: float = 1.0,
                 logit_loss_type: str = "kl_div",
                 logit_loss_temperature: float = 2.0,
                 logit_loss_weight: float = 1.0, 
                 use_stitching_layers: bool = True,
                 stitching_layer_bias: bool = True,
                 optimizer_cls=AdamW,
                 device: torch.device = torch.device("cpu"),
                 output_dir: str = "./distilled_model",
                 log_interval: int = 50,
                 freeze_previous_blocks: bool = True):
        """
        Initializes the BlockDistillationTrainer.
        """
        if DistillationLoss is None or LogitDistillationLoss is None:
            raise ImportError("Core loss functions (DistillationLoss, LogitDistillationLoss) are not loaded. Cannot initialize BlockDistillationTrainer.")

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

        self.logit_loss_type = logit_loss_type
        self.logit_loss_temperature = logit_loss_temperature
        self.logit_loss_weight = logit_loss_weight

        self.logit_loss_fn = LogitDistillationLoss(
            loss_type=self.logit_loss_type,
            temperature=self.logit_loss_temperature
        ).to(device)

        if self.use_stitching_layers:
            if StitchingDistillationLoss is None:
                raise ImportError("StitchingDistillationLoss is not available. Cannot use stitching layers.")
            logger.info("Using Stitching Layers for hidden state distillation.")
            try:
                teacher_actual_dim = teacher_model.config.hidden_size
                # Prefer teacher_n_embd from student config if available, otherwise fallback
                student_hidden_dim_input_to_stitching = getattr(student_model.config, 'teacher_n_embd', None)

                if student_hidden_dim_input_to_stitching is None:
                    logger.warning(
                        "student_model.config.teacher_n_embd is None. Falling back to teacher_model.config.hidden_size for student input to stitching."
                    )
                    student_hidden_dim_input_to_stitching = teacher_actual_dim
                
                if student_hidden_dim_input_to_stitching is None: # Should not happen if teacher_model.config.hidden_size exists
                     raise ValueError("Cannot determine student output hidden dimension for stitching layer input. "
                                      "Check teacher_model.config.hidden_size and student_model.config.teacher_n_embd.")


                logger.info(f"StitchingDistillationLoss will be configured for student hidden states of dim (input to stitch): {student_hidden_dim_input_to_stitching}")
                logger.info(f"StitchingDistillationLoss will project student states to teacher's dim (output of stitch): {teacher_actual_dim}")

                self.hidden_state_loss_fn = StitchingDistillationLoss(
                    loss_type=distill_loss_type,
                    temperature=distill_loss_temperature,
                    student_dims=student_hidden_dim_input_to_stitching,
                    teacher_dims=teacher_actual_dim,
                    use_bias=stitching_layer_bias
                ).to(device)

            except AttributeError as e:
                logger.error(f"Could not access required model config attributes (e.g., hidden_size, teacher_n_embd): {e}")
                logger.info("Falling back to dynamic stitching layer creation within StitchingDistillationLoss.")
                self.hidden_state_loss_fn = StitchingDistillationLoss(
                    loss_type=distill_loss_type,
                    temperature=distill_loss_temperature,
                    use_bias=stitching_layer_bias
                    # student_dims, teacher_dims, num_layers left None for fully dynamic creation
                ).to(device)
        else:
            logger.info("Using standard DistillationLoss (no stitching layers) for hidden states.")
            self.hidden_state_loss_fn = DistillationLoss(
                loss_type=distill_loss_type,
                temperature=distill_loss_temperature
            ).to(device)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")

        if not hasattr(self.student_model.config, 'output_hidden_states') or \
           not self.student_model.config.output_hidden_states:
            logger.warning("Student model's config.output_hidden_states is False. Setting to True for distillation.")
            self.student_model.config.output_hidden_states = True
        
        if hasattr(self.teacher_model.config, 'output_hidden_states') and not self.teacher_model.config.output_hidden_states:
            logger.warning("Teacher model's config.output_hidden_states is False. Setting to True for distillation.")
            self.teacher_model.config.output_hidden_states = True


    def _get_block_parameters(self, block_idx: int) -> List[torch.nn.Parameter]:
        params_to_train: List[torch.nn.Parameter] = []
        
        if block_idx < len(self.student_model.transformer.h):
            current_block_params = list(self.student_model.transformer.h[block_idx].parameters())
            params_to_train.extend(current_block_params)
            logger.debug(f"Added {len(current_block_params)} params from student block {block_idx}.")
        else:
            logger.warning(f"Block index {block_idx} is out of range for student model with {len(self.student_model.transformer.h)} layers.")

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
                    student_hidden_dim_input_to_stitching = getattr(self.student_model.config, 'teacher_n_embd', teacher_actual_dim)
                    self.hidden_state_loss_fn._get_stitching_layer(block_idx, student_hidden_dim_input_to_stitching, teacher_actual_dim)
                except Exception as e:
                    logger.warning(f"Error ensuring stitching layer {block_idx} exists for optimizer: {e}")
            
            if layer_key in self.hidden_state_loss_fn.stitching_layers:
                stitching_params = list(self.hidden_state_loss_fn.stitching_layers[layer_key].parameters())
                params_to_train.extend(stitching_params)
                logger.debug(f"Added {len(stitching_params)} params from stitching layer {block_idx}.")
            else:
                logger.warning(f"Stitching layer {block_idx} not found for optimizer setup (key: {layer_key}). Available keys: {list(self.hidden_state_loss_fn.stitching_layers.keys()) if hasattr(self.hidden_state_loss_fn, 'stitching_layers') else 'N/A'}")


        unique_params_to_train = []
        seen_params_ids = set()
        for p in params_to_train:
            if p.requires_grad and id(p) not in seen_params_ids:
                unique_params_to_train.append(p)
                seen_params_ids.add(id(p))
        
        logger.info(f"Optimizing {len(unique_params_to_train)} parameters for block {block_idx}.")
        return unique_params_to_train

    def _get_lm_head_parameters(self) -> List[torch.nn.Parameter]:
        params_to_train = []
        if hasattr(self.student_model, 'lm_head'):
            params_to_train.extend(list(self.student_model.lm_head.parameters()))
            logger.debug(f"Added {len(list(self.student_model.lm_head.parameters()))} params from lm_head.")
        if hasattr(self.student_model.transformer, 'ln_f'): 
            params_to_train.extend(list(self.student_model.transformer.ln_f.parameters()))
            logger.debug(f"Added {len(list(self.student_model.transformer.ln_f.parameters()))} params from final layer norm (ln_f).")
        
        unique_params_to_train = []
        seen_params_ids = set()
        for p in params_to_train:
            if p.requires_grad and id(p) not in seen_params_ids:
                unique_params_to_train.append(p)
                seen_params_ids.add(id(p))
        
        logger.info(f"Optimizing {len(unique_params_to_train)} parameters for LM head.")
        return unique_params_to_train

    def distill_block(self,
                      block_idx: int,
                      num_epochs: int,
                      learning_rate: float,
                      weight_decay: float = 0.01,
                      max_grad_norm: Optional[float] = 1.0):
        logger.info(f"--- Starting distillation for Block {block_idx + 1}/{self.student_model.config.n_layer} ---")
        self.student_model.train()

        if self.use_stitching_layers and hasattr(self.hidden_state_loss_fn, 'stitching_layers'):
            for s_layer_idx_str, s_layer_module in self.hidden_state_loss_fn.stitching_layers.items():
                s_layer_idx = int(s_layer_idx_str)
                should_train = (s_layer_idx == block_idx) or not self.freeze_previous_blocks
                for param in s_layer_module.parameters():
                    param.requires_grad = should_train
                if should_train:
                    logger.debug(f"Stitching layer {s_layer_idx} parameters set to trainable for block {block_idx}.")


        params_to_optimize = self._get_block_parameters(block_idx)
        if not params_to_optimize:
            logger.warning(f"No parameters to optimize for block {block_idx}. Skipping.")
            self.save_checkpoint(
                os.path.join(self.output_dir, f"student_model_block_{block_idx+1}_skipped_no_params.pt"),
                block_idx=block_idx, epoch=0, status="skipped_no_trainable_params"
            )
            return

        optimizer = self.optimizer_cls(params_to_optimize, lr=learning_rate, weight_decay=weight_decay)

        for epoch in range(num_epochs):
            logger.info(f"Block {block_idx + 1}, Epoch {epoch + 1}/{num_epochs}")
            epoch_loss_sum = 0.0
            num_batches_processed = 0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1} Block {block_idx+1}", leave=False)

            for batch_idx_iter, batch_data in enumerate(progress_bar):
                input_ids = batch_data['input_ids'].to(self.device)
                attention_mask = batch_data.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                optimizer.zero_grad()

                with torch.no_grad():
                    teacher_outputs_obj = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                if not hasattr(teacher_outputs_obj, 'hidden_states') or teacher_outputs_obj.hidden_states is None or len(teacher_outputs_obj.hidden_states) <= block_idx + 1:
                    raise ValueError(f"Teacher model did not return enough hidden states for block {block_idx}. Expected at least {block_idx + 2}, got {len(teacher_outputs_obj.hidden_states) if teacher_outputs_obj.hidden_states else 0}.")
                teacher_hidden_state = teacher_outputs_obj.hidden_states[block_idx + 1]

                student_full_outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                student_hidden_states_list = student_full_outputs.get('hidden_states')
                if student_hidden_states_list is None or len(student_hidden_states_list) <= block_idx + 1:
                    raise ValueError(f"Student model did not return enough hidden states for block {block_idx}. Expected at least {block_idx + 2}, got {len(student_hidden_states_list) if student_hidden_states_list else 0}.")
                student_hidden_state_current_block = student_hidden_states_list[block_idx + 1]
                
                if not self.use_stitching_layers and student_hidden_state_current_block.size(-1) != teacher_hidden_state.size(-1):
                    raise ValueError(
                        f"Dimension mismatch at block {block_idx} without stitching: "
                        f"Student_dim={student_hidden_state_current_block.size(-1)}, "
                        f"Teacher_dim={teacher_hidden_state.size(-1)}."
                    )

                if self.use_stitching_layers:
                    loss = self.hidden_state_loss_fn(student_hidden_state_current_block, teacher_hidden_state, layer_idx=block_idx)
                else:
                    loss = self.hidden_state_loss_fn(student_hidden_state_current_block, teacher_hidden_state)

                if loss is None or torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"Block {block_idx+1}, Epoch {epoch+1}, Batch {batch_idx_iter}: Loss is {loss}. Skipping batch.")
                    if torch.isnan(loss) or torch.isinf(loss):
                         self.save_checkpoint(os.path.join(self.output_dir, f"student_model_error_loss_block_{block_idx}.pt"), error_loss=float(loss.item() if loss is not None else float('nan')))
                         raise RuntimeError(f"Fatal {loss} loss encountered during block distillation.")
                    continue

                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, max_grad_norm)
                optimizer.step()

                epoch_loss_sum += loss.item()
                num_batches_processed += 1
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                if batch_idx_iter > 0 and batch_idx_iter % self.log_interval == 0:
                    current_avg_loss = epoch_loss_sum / num_batches_processed if num_batches_processed > 0 else float('nan')
                    logger.info(f"Block {block_idx+1}, Epoch {epoch+1}, Batch {batch_idx_iter}/{len(self.train_dataloader)}: Avg Loss: {current_avg_loss:.4f}, Current Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss_sum / num_batches_processed if num_batches_processed > 0 else float('nan')
            logger.info(f"Block {block_idx+1}, Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")

        logger.info(f"--- Finished distillation for Block {block_idx + 1} ---")
        block_checkpoint_path = os.path.join(self.output_dir, f"student_model_block_{block_idx+1}_distilled.pt")
        self.save_checkpoint(block_checkpoint_path, block_idx=block_idx, epoch=num_epochs, avg_loss=avg_epoch_loss)
        logger.info(f"Saved student model checkpoint after block {block_idx+1} to {block_checkpoint_path}")


    def distill_lm_head(self,
                        num_epochs: int,
                        learning_rate: float,
                        weight_decay: float = 0.01,
                        max_grad_norm: Optional[float] = 1.0):
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
        
        params_to_optimize = self._get_lm_head_parameters() # This will now correctly get lm_head and ln_f if they require grad
        if not params_to_optimize:
            logger.warning("No parameters found for language model head (lm_head, ln_f). Skipping LM head distillation.")
            return

        optimizer = self.optimizer_cls(params_to_optimize, lr=learning_rate, weight_decay=weight_decay)

        for epoch in range(num_epochs):
            logger.info(f"LM Head Distillation, Epoch {epoch + 1}/{num_epochs}")
            epoch_loss_sum = 0.0
            num_batches_processed = 0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1} LM Head", leave=False)

            for batch_idx_iter, batch_data in enumerate(progress_bar):
                input_ids = batch_data['input_ids'].to(self.device)
                attention_mask = batch_data.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                optimizer.zero_grad()

                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                teacher_logits = teacher_outputs.logits

                student_outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                student_logits = student_outputs.get('logits')
                if student_logits is None:
                    raise ValueError("Student model did not return 'logits' during LM head distillation.")

                loss = self.logit_loss_fn(student_logits, teacher_logits)
                
                if self.logit_loss_weight != 1.0:
                    loss = loss * self.logit_loss_weight

                if loss is None or torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"LM Head, Epoch {epoch+1}, Batch {batch_idx_iter}: Loss is {loss}. Skipping batch.")
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.save_checkpoint(os.path.join(self.output_dir, "student_model_error_loss_lm_head.pt"), error_loss=float(loss.item() if loss is not None else float('nan')))
                        raise RuntimeError(f"Fatal {loss} loss in LM head distillation.")
                    continue
                
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, max_grad_norm)
                optimizer.step()

                epoch_loss_sum += loss.item()
                num_batches_processed += 1
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                if batch_idx_iter > 0 and batch_idx_iter % self.log_interval == 0:
                    current_avg_loss = epoch_loss_sum / num_batches_processed if num_batches_processed > 0 else float('nan')
                    logger.info(f"LM Head, Epoch {epoch+1}, Batch {batch_idx_iter}/{len(self.train_dataloader)}: Avg Loss: {current_avg_loss:.4f}, Current Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss_sum / num_batches_processed if num_batches_processed > 0 else float('nan')
            logger.info(f"LM Head, Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
        
        logger.info(f"--- Finished distillation for Language Model Head ---")
        lm_head_checkpoint_path = os.path.join(self.output_dir, "student_model_lm_head_distilled.pt")
        self.save_checkpoint(lm_head_checkpoint_path, lm_head_distilled=True, epoch=num_epochs, avg_loss=avg_epoch_loss)
        logger.info(f"Saved student model checkpoint after LM head distillation to {lm_head_checkpoint_path}")


    def save_checkpoint(self, path: str, **kwargs):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            # Attempt to make config_distillation.GPTConfig safe if it's the one used by student_model.config
            # This is a best-effort attempt. The actual config class type matters.
            if hasattr(self.student_model.config, '__class__'):
                config_class = self.student_model.config.__class__
                # Check if it's likely the GPTConfig from config_distillation
                if "GPTConfig" in str(config_class) and "config_distillation" in str(config_class.__module__):
                     import torch.serialization
                     if hasattr(torch.serialization, 'add_safe_globals'):
                         torch.serialization.add_safe_globals([config_class])
                         logger.debug(f"Added {config_class} to safe globals for pickling.")
        except ImportError:
            logger.debug("Could not import config_distillation.GPTConfig for pickling check.")
        except Exception as e:
            logger.warning(f"Could not add student_model.config's class to safe globals: {e}")
        
        checkpoint = {
            'model_state_dict': self.student_model.state_dict(),
            'student_config': self.student_model.config, 
            **kwargs
        }
        
        if self.use_stitching_layers and hasattr(self.hidden_state_loss_fn, 'stitching_layers') and len(self.hidden_state_loss_fn.stitching_layers) > 0:
            checkpoint['stitching_layers_state_dict'] = self.hidden_state_loss_fn.stitching_layers.state_dict()
            checkpoint['use_stitching_layers'] = True

        torch.save(checkpoint, path, pickle_protocol=4) 
        logger.info(f"Student model checkpoint saved to {path}")

    def train(self,
              epochs_per_block: int,
              lr_per_block: Union[float, List[float]],
              wd_per_block: Union[float, List[float]] = 0.01,
              max_grad_norm_per_block: Optional[Union[float, List[float]]] = 1.0,
              train_lm_head: bool = True, 
              lm_head_epochs: int = 1,    
              lm_head_lr: float = 1e-4,   
              lm_head_wd: float = 0.01,   
              lm_head_max_grad_norm: Optional[float] = 1.0
              ):
        num_student_layers = self.student_model.config.n_layer
        num_teacher_layers = getattr(self.teacher_model.config, 'n_layer', getattr(self.teacher_model.config, 'num_hidden_layers', 0))

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
                logger.debug(f"Block {block_idx} (freezing): Student block {block_idx} trainable. Embeddings trainable: {train_embeddings}")
            else: 
                for param_group in self.student_model.parameters(): 
                    param_group.requires_grad = True # Ensure all student params are trainable
                logger.debug(f"Block {block_idx} (not freezing): All student model parameters set to trainable.")
            
            self.distill_block(
                block_idx,
                num_epochs=epochs_per_block,
                learning_rate=current_lr,
                weight_decay=current_wd,
                max_grad_norm=current_max_grad_norm
            )

        if train_lm_head:
            logger.info("Proceeding to LM head distillation phase.")
            self.distill_lm_head(
                num_epochs=lm_head_epochs,
                learning_rate=lm_head_lr,
                weight_decay=lm_head_wd,
                max_grad_norm=lm_head_max_grad_norm
            )
        else:
            logger.info("Skipping LM head distillation phase as per configuration.")

        final_model_path = os.path.join(self.output_dir, "student_model_final_distilled.pt")
        self.save_checkpoint(final_model_path, status="final_distillation_complete", trained_lm_head=train_lm_head)
        logger.info(f"Full distillation complete. Final student model saved to {final_model_path}")


