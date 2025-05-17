# ./model/model_SASPV_distillation.py
"""
SASP Transformer model adapted for distillation.
This version extends SASPTransformerModel with specific distillation capabilities.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# Import the base model from which we'll inherit
from model.model_SASPV import SASPTransformerModel

class SASPTransformerModelDistillation(SASPTransformerModel):
    """
    Extension of SASPTransformerModel specifically designed for distillation.
    Changes:
    1. Always outputs hidden states from all layers
    2. Ensures compatibility with the teacher model's hidden state dimensions
    3. Adds support for hidden state projection if dimensions don't match
    """
    def __init__(self, config):
        # Ensure output_hidden_states is enabled for distillation
        config.output_hidden_states = True
        super().__init__(config)
        
        # If specified in config, add projection layers for hidden states
        # This is needed if teacher and student embedding dimensions don't match
        self.hidden_state_projection = None
        if hasattr(config, 'teacher_n_embd') and config.teacher_n_embd != config.n_embd:
            self.hidden_state_projection = nn.Linear(config.n_embd, config.teacher_n_embd, bias=False)
            print(f"Added hidden state projection from {config.n_embd} to {config.teacher_n_embd}")

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass for the SASP Transformer model with distillation support.
        Collects hidden states from each layer for distillation purposes.
        """
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Standard forward pass implementation collecting hidden states
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        
        # Get embeddings
        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Collection of hidden states for distillation
        all_hidden_states = []
        # Store the initial state (after embeddings)
        all_hidden_states.append(x)
        
        # Pass through layers, collecting hidden states
        for block in self.transformer.h:
            x = block(x)
            all_hidden_states.append(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Project hidden states if dimensions need adjustment
        if self.hidden_state_projection is not None:
            projected_hidden_states = []
            for hidden_state in all_hidden_states:
                projected_hidden_states.append(self.hidden_state_projection(hidden_state))
            all_hidden_states = projected_hidden_states
            
        # Return results with hidden states
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': all_hidden_states
        }
