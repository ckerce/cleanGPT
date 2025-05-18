# ./distillation/losses/logit_loss.py
"""
Module for distillation loss functions related to logits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class LogitDistillationLoss(nn.Module):
    """
    Specialized loss for distillation at the logit level (language model head).
    Compares the output logits of the student model with the teacher model.
    """
    def __init__(self, loss_type="kl_div", temperature=2.0):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.temperature = temperature
        
        if self.loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif self.loss_type == "kl_div":
            self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        elif self.loss_type == "ce":
            # Cross-entropy loss - useful for when we want to match exact probabilities
            self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        else:
            raise ValueError(f"Unsupported loss type for logit distillation: {loss_type}. Supported: 'mse', 'kl_div', 'ce'.")
    
    def forward(self, student_logits, teacher_logits):
        """
        Compute loss between student and teacher logits.
        Args:
            student_logits: Logits from student model [batch_size, seq_len, vocab_size]
            teacher_logits: Logits from teacher model [batch_size, seq_len, vocab_size]
        """
        # Check shapes match
        if student_logits.shape != teacher_logits.shape:
            logger.error(
                f"LogitDistillationLoss: Shape mismatch between student logits ({student_logits.shape}) "
                f"and teacher logits ({teacher_logits.shape}). Ensure vocabularies match."
            )
            # This might cause an error depending on the loss function
        
        # Compute loss based on selected loss type
        if self.loss_type == "mse":
            return self.loss_fn(student_logits, teacher_logits.detach())
        
        elif self.loss_type == "kl_div":
            # KL divergence between student and teacher distributions
            student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits.detach() / self.temperature, dim=-1)
            # Scale by temperature^2 as per the original distillation paper
            return self.loss_fn(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        elif self.loss_type == "ce":
            # Cross-entropy loss - treating teacher distribution as ground truth
            # Reshape for cross entropy: [batch_size * seq_len, vocab_size]
            student_logits_reshaped = student_logits.view(-1, student_logits.size(-1))
            # Create pseudo-labels from teacher by taking argmax
            # Ensure teacher_logits is detached before argmax if it might have grads
            teacher_pseudo_labels = teacher_logits.detach().argmax(dim=-1).view(-1)
            return self.loss_fn(student_logits_reshaped, teacher_pseudo_labels)
        
        raise ValueError(f"Logit loss calculation failed for loss type {self.loss_type}")
