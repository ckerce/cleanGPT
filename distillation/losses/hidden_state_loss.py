# ./distillation/losses/hidden_state_loss.py
"""
Module for distillation loss functions related to hidden states.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class DistillationLoss(nn.Module):
    """
    Wrapper for various distillation loss functions for hidden states.
    This is used when stitching layers are NOT active for hidden state distillation.
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

        if self.loss_type == "mse":
            return self.loss_fn(student_outputs, teacher_outputs.detach())
        elif self.loss_type == "kl_div":
            student_log_probs = F.log_softmax(student_outputs / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_outputs.detach() / self.temperature, dim=-1)
            return self.loss_fn(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        raise ValueError(f"Loss calculation failed for loss type {self.loss_type}")
