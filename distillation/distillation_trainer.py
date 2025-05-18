# ./distillation/distillation_trainer.py
"""
Main orchestrator for distillation training.
"""
import os
import logging
import torch
from typing import List, Optional, Union, Dict, Any

from distillation.trainers.backbone_trainer import BackboneDistillationTrainer
from distillation.trainers.head_trainer import HeadDistillationTrainer

logger = logging.getLogger(__name__)

class DistillationTrainer:
    """
    Main distillation trainer that coordinates backbone and head training.
    """
    
    def __init__(self,
                 teacher_model,
                 student_model,
                 tokenizer,
                 train_dataloader,
                 device=torch.device("cpu"),
                 output_dir="./distilled_model",
                 log_interval=50,
                 # Backbone trainer parameters
                 distill_loss_type="mse",
                 distill_loss_temperature=1.0,
                 use_stitching_layers=True,
                 stitching_layer_bias=True,
                 freeze_previous_blocks=True,
                 # Head trainer parameters
                 logit_loss_type="kl_div",
                 logit_loss_temperature=2.0,
                 logit_loss_weight=1.0,
                 # Common parameters
                 optimizer_cls=torch.optim.AdamW):
        """
        Initialize the distillation trainer.
        
        Args:
            teacher_model: Teacher model to distill from
            student_model: Student model to train
            tokenizer: Tokenizer for processing text
            train_dataloader: DataLoader for training examples
            device: Device to train on
            output_dir: Directory to save model checkpoints
            log_interval: How often to log during training
            distill_loss_type: Type of hidden state distillation loss
            distill_loss_temperature: Temperature for hidden state loss
            use_stitching_layers: Whether to use stitching layers
            stitching_layer_bias: Whether to use bias in stitching layers
            freeze_previous_blocks: Whether to freeze previously trained blocks
            logit_loss_type: Type of logit loss
            logit_loss_temperature: Temperature for logit loss
            logit_loss_weight: Weight for logit loss
            optimizer_cls: Optimizer class to use
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.device = device
        self.output_dir = output_dir
        
        # Create backbone trainer
        self.backbone_trainer = BackboneDistillationTrainer(
            teacher_model=teacher_model,
            student_model=student_model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            device=device,
            output_dir=output_dir,
            log_interval=log_interval,
            distill_loss_type=distill_loss_type,
            distill_loss_temperature=distill_loss_temperature,
            use_stitching_layers=use_stitching_layers,
            stitching_layer_bias=stitching_layer_bias,
            freeze_previous_blocks=freeze_previous_blocks,
            optimizer_cls=optimizer_cls
        )
        
        # Create head trainer
        self.head_trainer = HeadDistillationTrainer(
            teacher_model=teacher_model,
            student_model=student_model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            device=device,
            output_dir=output_dir,
            log_interval=log_interval,
            logit_loss_type=logit_loss_type,
            logit_loss_temperature=logit_loss_temperature,
            logit_loss_weight=logit_loss_weight,
            optimizer_cls=optimizer_cls
        )
        
    def train(self,
              epochs_per_block: int,
              lr_per_block: Union[float, List[float]],
              wd_per_block: Union[float, List[float]] = 0.01,
              max_grad_norm_per_block: Optional[Union[float, List[float]]] = 1.0,
              train_lm_head: bool = True,
              lm_head_epochs: int = 1,
              lm_head_lr: float = 1e-4,
              lm_head_wd: float = 0.01,
              lm_head_max_grad_norm: Optional[float] = 1.0,
              initialize_head_from_teacher: bool = True):
        """
        Run full distillation process (backbone and optionally head).
        
        Args:
            epochs_per_block: Number of epochs to train each block
            lr_per_block: Learning rate for each block (float or list)
            wd_per_block: Weight decay for each block (float or list)
            max_grad_norm_per_block: Max gradient norm for each block (float or list)
            train_lm_head: Whether to train the LM head
            lm_head_epochs: Number of epochs for LM head training
            lm_head_lr: Learning rate for LM head training
            lm_head_wd: Weight decay for LM head training
            lm_head_max_grad_norm: Max gradient norm for LM head training
            initialize_head_from_teacher: Whether to initialize the LM head from the teacher
            
        Returns:
            Dictionary with training results
        """
        results = {
            'backbone_losses': [],
            'head_loss': None
        }
        
        # Train backbone
        logger.info("Starting backbone distillation...")
        backbone_losses = self.backbone_trainer.train_backbone(
            epochs_per_block=epochs_per_block,
            lr_per_block=lr_per_block,
            wd_per_block=wd_per_block,
            max_grad_norm_per_block=max_grad_norm_per_block
        )
        results['backbone_losses'] = backbone_losses
        
        # Train LM head if requested
        if train_lm_head:
            logger.info("Starting LM head distillation...")
            head_loss = self.head_trainer.train_head(
                lm_head_epochs=lm_head_epochs,
                lm_head_lr=lm_head_lr,
                lm_head_wd=lm_head_wd,
                lm_head_max_grad_norm=lm_head_max_grad_norm,
                initialize_from_teacher=initialize_head_from_teacher
            )
            results['head_loss'] = head_loss
        else:
            logger.info("Skipping LM head distillation as per configuration.")
        
        # Save final model
        final_model_path = os.path.join(self.output_dir, "student_model_final_distilled.pt")
        self.backbone_trainer.save_checkpoint(
            final_model_path, 
            status="final_distillation_complete", 
            trained_lm_head=train_lm_head,
            results=results
        )
        logger.info(f"Full distillation complete. Final student model saved to {final_model_path}")
        
        return results
