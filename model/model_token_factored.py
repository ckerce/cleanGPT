# ./model/model_token_factored.py
"""
Factored Transformer model with Pre-Layer Normalization.
This model incorporates xt and xe embedding streams, where:
- xt is primarily updated by the attention mechanism and represents token-like symbolic states.
- xe is primarily updated by the MLP and represents embedding-like contextual states.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input_tensor):
        """
        Applies Layer Normalization.
        Handles potential dtype issues for stability.
        """
        orig_dtype = input_tensor.dtype
        # For lower precision inputs (float16, bfloat16),
        # cast to float32 for the normalization computation for stability, then cast back.
        input_for_norm = input_tensor
        if input_tensor.dtype == torch.bfloat16 or input_tensor.dtype == torch.float16:
            input_for_norm = input_tensor.float()
        
        # Ensure weight and bias are also float32 if input was cast
        weight = self.weight.float() if input_for_norm.dtype == torch.float32 else self.weight
        bias = self.bias.float() if self.bias is not None and input_for_norm.dtype == torch.float32 else self.bias
        
        output = F.layer_norm(input_for_norm, weight.shape, weight, bias, 1e-5)
        return output.to(orig_dtype)


class FactoredCausalSelfAttention(nn.Module):
    """
    Causal self-attention mechanism for the Factored Transformer.
    This preserves the symbolic structure while allowing contextual attention patterns.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.c_attn = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        # NO c_proj - would distort xt symbolic structure
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        # No resid_dropout since no c_proj
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Causal mask to ensure attention is only to the left
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x_norm_for_qk, xt_current_for_v):
        """
        Forward pass for FactoredCausalSelfAttention.
        Args:
            x_norm_for_qk (torch.Tensor): Normalized (xt + xe) used for Q and K computation.
            xt_current_for_v (torch.Tensor): The current xt stream, used directly as values.
        Returns:
            torch.Tensor: The output of the attention mechanism, to be added to xt.
        """
        B, T, C = x_norm_for_qk.size()
        
        q, k = self.c_attn(x_norm_for_qk).split(self.n_embd, dim=2)

        # Reshape Q and K for multi-head attention
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        v = xt_current_for_v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Scaled dot-product attention
        att_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply causal mask
        att_scores = att_scores.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        
        # Softmax to get attention weights
        att_weights = F.softmax(att_scores, dim=-1)
        att_weights = self.attn_dropout(att_weights)
        
        # Apply attention to the values (which are just xt)
        y = att_weights @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Re-assemble all head outputs - this is the final output (no projection)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return y


class FactoredMLP(nn.Module):
    """
    MLP for the Factored Transformer.
    Takes norm(xt_updated_by_attention + xe) as input.
    The output of this block updates xe.
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x_norm):
        """
        Forward pass for FactoredMLP.
        Args:
            x_norm (torch.Tensor): Normalized (xt_after_attention_update + xe_before_mlp_update).
        Returns:
            torch.Tensor: The output of the MLP, to be added to xe.
        """
        x = self.c_fc(x_norm)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class FactoredPreLNBlock(nn.Module):
    """
    Transformer block with Pre-Layer Normalization for the Factored Transformer.
    Manages the separate update paths for xt and xe.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias) # For attention input
        self.attn = FactoredCausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias) # For MLP input
        self.mlp = FactoredMLP(config)

    def forward(self, xt, xe):
        """
        Forward pass for FactoredPreLNBlock.
        Args:
            xt (torch.Tensor): The 'token-related' embedding stream.
            xe (torch.Tensor): The 'environment-related' embedding stream.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The updated xt and xe streams.
        """
        # Attention Path
        norm_for_attn_qk = self.ln_1(xt + xe)
        attn_output = self.attn(x_norm_for_qk=norm_for_attn_qk, xt_current_for_v=xt)
        xt = xt + attn_output

        # MLP Path
        # Input to MLP is norm(xt_updated_by_attention + xe_original_passed_to_block)
        # MLP output updates only xe.
        norm_for_mlp = self.ln_2(xt + xe) # Note: xt here is the one updated by attention
        mlp_output = self.mlp(norm_for_mlp)
        xe = xe + mlp_output

        return xt, xe


class FactoredTransformerModel(nn.Module):
    """
    Factored Transformer model using Pre-Layer Normalization and xt/xe streams.
    """
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None, "vocab_size must be specified in config"
        assert config.block_size is not None, "block_size must be specified in config"
        self.config = config
        self.padding_idx = getattr(config, 'padding_idx', None)

        # Model components dictionary
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=self.padding_idx),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([FactoredPreLNBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between token embeddings (wte) and the final linear layer (lm_head)
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled initialization to the output projection layers
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"FactoredTransformerModel initialized with {self.get_num_params()/1e6:.2f}M parameters")

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        If non_embedding is True, subtracts parameters of positional embeddings.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Initialize model weights according to common practices."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass for the FactoredTransformerModel.
        Args:
            input_ids (torch.Tensor): Input token IDs (batch_size, sequence_length).
            attention_mask (torch.Tensor, optional): Mask for padded tokens.
            labels (torch.Tensor, optional): Target token IDs for loss calculation.
        Returns:
            dict: A dictionary containing 'loss' (if labels provided) and 'logits'.
        """
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Positional indices
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        # Initial Embeddings
        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(pos)
        
        # Initialize xt and xe streams
        xt = tok_emb + pos_emb
        xt = self.transformer.drop(xt)
        xe = torch.zeros_like(xt, device=device)

        # Pass through transformer blocks
        for block in self.transformer.h:
            xt, xe = block(xt, xe)

        # Final combination and normalization
        x_final_combined = xt + xe
        x_final_normed = self.transformer.ln_f(x_final_combined)

        # Language model head
        logits = self.lm_head(x_final_normed)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return {'loss': loss, 'logits': logits}

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.
        Args:
            idx (torch.Tensor): Current context token IDs.
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Sampling temperature.
            top_k (int, optional): Top-k sampling.
        Returns:
            torch.Tensor: Generated sequence including initial context.
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Ensure context doesn't exceed block size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Get model outputs
            outputs = self(idx_cond)
            logits = outputs['logits']
            
            # Focus on last token's logits
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Convert to probabilities and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

        self.train()
        return idx
