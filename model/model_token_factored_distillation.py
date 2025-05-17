# ./model/model_token_factored_distillation.py
"""
Factored Transformer model with Pre-Layer Normalization, adapted for distillation.
This version extends FactoredTransformerModel with specific distillation capabilities.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# Import the base model from which we'll inherit
from model.model_token_factored import FactoredTransformerModel, FactoredPreLNBlock, LayerNorm

class FactoredTransformerModelDistillation(FactoredTransformerModel):
    """
    Extension of FactoredTransformerModel specifically designed for distillation.
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
        Forward pass for the Factored Transformer model with distillation support.
        Extends the base implementation to collect and return hidden states from all layers.
        Args:
            input_ids: Input token IDs (batch_size, sequence_length)
            attention_mask: Optional mask for padded tokens
            labels: Optional target token IDs for loss calculation
        Returns:
            dict: Contains 'loss', 'logits', and 'hidden_states'
        """
        device = input_ids.device
        b, t = input_ids.size() # batch_size, sequence_length
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Positional indices (0, 1, ..., t-1)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # Shape (1, t)

        # Initial Embeddings
        tok_emb = self.transformer.wte(input_ids) # Token embeddings: (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)       # Positional embeddings: (1, t, n_embd)
        
        # Initialize xt and xe streams
        xt = tok_emb + pos_emb # xt starts as the sum of token and positional embeddings
        xt = self.transformer.drop(xt) # Apply dropout to the initial xt
        xe = torch.zeros_like(xt, device=device) # xe starts as a zero tensor

        # Collection of hidden states for distillation
        all_hidden_states = []
        
        # Store the initial combined state (after embeddings)
        # This corresponds to hidden_states[0] in Hugging Face models
        all_hidden_states.append(xt + xe)
        
        # Pass xt and xe through the stack of FactoredPreLNBlocks
        for block in self.transformer.h:
            xt, xe = block(xt, xe)
            # Store the combined hidden state after each block
            all_hidden_states.append(xt + xe)

        # Final combination and normalization for the output head
        x_final_combined = xt + xe
        x_final_normed = self.transformer.ln_f(x_final_combined)

        # Language model head to get logits
        logits = self.lm_head(x_final_normed) # Shape (b, t, vocab_size)

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
            
        # Return results in a dictionary with hidden states
        return_dict = {
            'loss': loss, 
            'logits': logits,
            'hidden_states': all_hidden_states  # Important for distillation
        }
        return return_dict

# Delete unnecessary class definition
# The base FactoredPreLNBlock class from model_token_factored.py will be used instead
