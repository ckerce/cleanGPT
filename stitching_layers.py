# stitching_layers.py
# This file contains all the components for implementing a trainable linear stitching layer
# between teacher and student models in distillation.

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class StitchingLayer(nn.Module):
    """
    A trainable linear transformation layer to map between student and teacher hidden state spaces.
    
    This layer helps bridge the representational gap between different model architectures,
    allowing more effective distillation even when the models have different internal structures.
    """
    def __init__(self, student_dim: int, teacher_dim: int, bias: bool = True):
        super().__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        
        # Create a linear projection from student space to teacher space
        self.projection = nn.Linear(student_dim, teacher_dim, bias=bias)
        
        # Initialize with identity-like transformation when dimensions match
        # or a scaled random initialization when they don't
        if student_dim == teacher_dim:
            # Initialize close to identity
            nn.init.eye_(self.projection.weight)
            if bias:
                nn.init.zeros_(self.projection.bias)
        else:
            # Use standard initialization with scaling
            nn.init.xavier_uniform_(self.projection.weight)
            if bias:
                nn.init.zeros_(self.projection.bias)
    
    def forward(self, student_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Map student hidden states to teacher hidden space.
        
        Args:
            student_hidden_states: Hidden states from student model
                                   Shape: [batch_size, sequence_length, student_dim]
        
        Returns:
            Transformed hidden states in teacher space
            Shape: [batch_size, sequence_length, teacher_dim]
        """
        return self.projection(student_hidden_states)


class StitchingDistillationLoss(nn.Module):
    """
    Distillation loss that uses trainable stitching layers to map between
    student and teacher hidden state spaces before comparison.
    """
    def __init__(self, 
                 loss_type: str = "mse", 
                 temperature: float = 1.0, 
                 student_dims: Optional[List[int]] = None, 
                 teacher_dims: Optional[List[int]] = None, 
                 use_bias: bool = True):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.temperature = temperature
        
        if self.loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif self.loss_type == "kl_div":
            self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}. Supported: 'mse', 'kl_div'.")
        
        # Set up stitching layers for each transformer block
        self.stitching_layers = nn.ModuleDict()
        self.use_bias = use_bias # Store use_bias for dynamic layer creation
        
        # If dimensions are provided, pre-create the layers
        if student_dims is not None and teacher_dims is not None:
            # Ensure student_dims and teacher_dims are lists
            s_dims_list = [student_dims] * len(teacher_dims) if isinstance(student_dims, int) and isinstance(teacher_dims, list) else student_dims
            t_dims_list = [teacher_dims] * len(s_dims_list) if isinstance(teacher_dims, int) and isinstance(s_dims_list, list) else teacher_dims
            
            if not isinstance(s_dims_list, list) or not isinstance(t_dims_list, list) or len(s_dims_list) != len(t_dims_list):
                 # Fallback if single integers were passed for both, assuming one layer
                if isinstance(s_dims_list, int) and isinstance(t_dims_list, int):
                    s_dims_list = [s_dims_list]
                    t_dims_list = [t_dims_list]
                else:
                    raise ValueError("student_dims and teacher_dims must be lists of the same length, or student_dims can be an int if teacher_dims is a list (or vice-versa).")

            for i, (s_dim, t_dim) in enumerate(zip(s_dims_list, t_dims_list)):
                self.stitching_layers[str(i)] = StitchingLayer(s_dim, t_dim, bias=self.use_bias)
                
    def _get_stitching_layer(self, layer_idx: int, student_dim: int, teacher_dim: int) -> StitchingLayer:
        """Get or create a stitching layer for a specific block"""
        layer_key = str(layer_idx)
        if layer_key not in self.stitching_layers:
            logger.info(f"Dynamically creating stitching layer for block {layer_idx} ({student_dim} -> {teacher_dim})")
            # Ensure the new layer is on the same device as existing parameters
            device = next(self.parameters(), torch.zeros(1)).device 
            self.stitching_layers[layer_key] = StitchingLayer(
                student_dim, teacher_dim, bias=self.use_bias
            ).to(device)
        return self.stitching_layers[layer_key]
        
    def forward(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        """
        Compute loss between student and teacher outputs using the appropriate stitching layer.
        
        Args:
            student_outputs: Tensor from student model (e.g., hidden state).
            teacher_outputs: Tensor from teacher model (e.g., hidden state).
            layer_idx: Index of the transformer layer for selecting the correct stitching layer.
        """
        # Get student and teacher hidden dimensions
        student_dim = student_outputs.size(-1)
        teacher_dim = teacher_outputs.size(-1)
        
        # Get the appropriate stitching layer
        stitching_layer = self._get_stitching_layer(layer_idx, student_dim, teacher_dim)
        
        # Project student outputs to teacher space
        projected_student = stitching_layer(student_outputs)
        
        # Now compute the loss between the projected student outputs and teacher outputs
        if self.loss_type == "mse":
            return self.loss_fn(projected_student, teacher_outputs.detach())
        elif self.loss_type == "kl_div":
            # Assumes inputs are logits.
            student_log_probs = F.log_softmax(projected_student / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_outputs.detach() / self.temperature, dim=-1)
            return self.loss_fn(student_log_probs, teacher_probs) * (self.temperature ** 2)

        # Should not be reached if loss_type is validated in __init__
        raise ValueError(f"Loss calculation failed for loss type {self.loss_type}")


