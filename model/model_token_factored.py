# ./model/model_Factored.py
"""
Factored Transformer model with Pre-Layer Normalization.
This model incorporates xt and xe embedding streams, where:
- xt is primarily updated by the attention mechanism.
- xe is primarily updated by the MLP.
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
    - Q and K are derived from norm(xt + xe).
    - V is effectively xt_current * V_derived_from_norm(xt+xe).
    The output of this block updates xt.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Key, Query, Value projections from the combined normalized input
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout) # Dropout on the output of the projection
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # self.dropout = config.dropout # Stored for reference, used in self.attn_dropout

        # Causal mask to ensure attention is only to the left
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x_norm_for_qkv, xt_current_for_v):
        """
        Forward pass for FactoredCausalSelfAttention.
        Args:
            x_norm_for_qkv (torch.Tensor): Normalized (xt + xe) used for Q, K, and initial V.
            xt_current_for_v (torch.Tensor): The current xt stream, used to modulate V.
        Returns:
            torch.Tensor: The output of the attention mechanism, to be added to xt.
        """
        B, T, C = x_norm_for_qkv.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        # Calculate query, key, and an initial value (v_orig) from x_norm_for_qkv
        q, k, v_orig = self.c_attn(x_norm_for_qkv).split(self.n_embd, dim=2)

        # Reshape Q, K, V_orig for multi-head attention
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v_orig = v_orig.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Modulate v_orig using xt_current_for_v to get the effective V
        # Reshape xt_current_for_v to match v_orig's shape for element-wise multiplication
        xt_reshaped_for_v = xt_current_for_v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        effective_v = xt_reshaped_for_v * v_orig # Element-wise multiplication

        # Scaled dot-product attention
        att_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply causal mask
        att_scores = att_scores.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        
        # Softmax to get attention weights
        att_weights = F.softmax(att_scores, dim=-1)
        att_weights = self.attn_dropout(att_weights)
        
        # Apply attention to the effective_v
        y = att_weights @ effective_v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Re-assemble all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)
        
        # Output projection and dropout
        y = self.resid_dropout(self.c_proj(y))
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
        self.activation = nn.GELU() # Standard GELU activation

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
        # Input to attention's QKV derivation is norm(xt + xe)
        # The V used in attention is modulated by the original xt passed to this block.
        # Attention output updates only xt.
        norm_for_attn_qkv = self.ln_1(xt + xe)
        attn_output = self.attn(x_norm_for_qkv=norm_for_attn_qkv, xt_current_for_v=xt)
        xt = xt + attn_output  # xe remains unchanged by the attention output itself

        # MLP Path
        # Input to MLP is norm(xt_updated_by_attention + xe_original_passed_to_block)
        # MLP output updates only xe.
        norm_for_mlp = self.ln_2(xt + xe) # Note: xt here is the one updated by attention
        mlp_output = self.mlp(norm_for_mlp)
        xe = xe + mlp_output  # xt remains unchanged by the MLP output itself
        
        # NOTE for future consideration on normalization:
        # To potentially improve training stability, one might consider normalizing
        # xt and xe *after* their respective updates within this block, e.g.:
        # xt_new = self.ln_xt_out(xt + attn_output)
        # xe_new = self.ln_xe_out(xe + mlp_output)
        # This would require additional LayerNorm modules (e.g., self.ln_xt_out, self.ln_xe_out)
        # and careful consideration of where those norms are placed relative to the residual.
        # For now, adhering to the simpler residual addition before any such output normalization.

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
        self.padding_idx = getattr(config, 'padding_idx', None) # For embedding layer

        # Model components dictionary
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=self.padding_idx), # Token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # Positional embeddings
            drop = nn.Dropout(config.dropout), # Dropout for the sum of token and positional embeddings
            h = nn.ModuleList([FactoredPreLNBlock(config) for _ in range(config.n_layer)]), # Stack of FactoredPreLNBlocks
            ln_f = LayerNorm(config.n_embd, bias=config.bias), # Final layer norm before output head
        ))

        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between token embeddings (wte) and the final linear layer (lm_head)
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled initialization to the output projection layers in Attention and MLP
        # This is a common practice from GPT-2 to help with training stability.
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'): # Targets output projections in FactoredCausalSelfAttention and FactoredMLP
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"FactoredTransformerModel initialized with {self.get_num_params()/1e6:.2f}M parameters")

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        If non_embedding is True, subtracts parameters of positional embeddings.
        Token embeddings are tied to lm_head, so they are counted.
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
                with torch.no_grad(): # Ensure no gradient tracking during this in-place operation
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, LayerNorm):
            # LayerNorm weights (gamma) are initialized to 1 and biases (beta) to 0 by default
            # within the LayerNorm class's __init__ if bias is enabled.
            pass


    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass for the FactoredTransformerModel.
        Args:
            input_ids (torch.Tensor): Input token IDs (batch_size, sequence_length).
            attention_mask (torch.Tensor, optional): Mask for padded tokens. Not directly used by
                                                     FactoredCausalSelfAttention's causal mask but
                                                     can be relevant for padding in embeddings if padding_idx is set.
            labels (torch.Tensor, optional): Target token IDs for loss calculation (batch_size, sequence_length).
        Returns:
            dict: A dictionary containing 'loss' (if labels are provided), 'logits',
                  and optionally 'xt_final', 'xe_final' for debugging.
        """
        device = input_ids.device
        b, t = input_ids.size() # batch_size, sequence_length
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Positional indices (0, 1, ..., t-1)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # Shape (1, t)

        # Initial Embeddings
        tok_emb = self.transformer.wte(input_ids) # Token embeddings: (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)       # Positional embeddings: (1, t, n_embd) broadcastable to (b,t,n_embd)
        
        # Initialize xt and xe streams
        xt = tok_emb + pos_emb # xt starts as the sum of token and positional embeddings
        xt = self.transformer.drop(xt) # Apply dropout to the initial xt
        xe = torch.zeros_like(xt, device=device) # xe starts as a zero tensor

        # Pass xt and xe through the stack of FactoredPreLNBlocks
        for block in self.transformer.h:
            xt, xe = block(xt, xe)

        # Final combination and normalization for the output head
        x_final_combined = xt + xe
        x_final_normed = self.transformer.ln_f(x_final_combined)

        # Language model head to get logits
        logits = self.lm_head(x_final_normed) # Shape (b, t, vocab_size)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Flatten the logits and labels for CrossEntropyLoss
            # Logits: (batch_size * sequence_length, vocab_size)
            # Labels: (batch_size * sequence_length)
            # ignore_index=-100 is a common convention for padded tokens in labels.
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100) 
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Return results in a dictionary
        # Optionally, return final xt and xe for analysis or debugging
        return_dict = {'loss': loss, 'logits': logits}
        # if self.config.get('output_xt_xe', False): # Example: control via config
        #     return_dict['xt_final'] = xt
        #     return_dict['xe_final'] = xe
        return return_dict

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.
        Args:
            idx (torch.Tensor): Current context token IDs (batch_size, current_sequence_length).
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Sampling temperature. Higher values make output more random.
            top_k (int, optional): If set, restricts sampling to the top_k most likely tokens.
        Returns:
            torch.Tensor: The generated sequence of token IDs, including the initial context.
        """
        self.eval() # Set the model to evaluation mode

        # The generation loop re-evaluates the model with the growing sequence.
        # The forward pass handles the xt, xe initialization internally based on `idx`.
        for _ in range(max_new_tokens):
            # Ensure the context for the model does not exceed its block size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Get model outputs (logits) for the current context
            outputs = self(idx_cond) 
            logits = outputs['logits'] # Shape (b, t_cond, vocab_size)
            
            # Focus on the logits for the very last token in the sequence
            logits = logits[:, -1, :] / temperature # Shape (b, vocab_size)

            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Zero out logits for tokens not in the top k
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Convert logits to probabilities using softmax
            probs = F.softmax(logits, dim=-1) # Shape (b, vocab_size)

            # Sample the next token index from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1) # Shape (b, 1)

            # Append the sampled token to the current sequence
            idx = torch.cat((idx, idx_next), dim=1) # Shape (b, t_current + 1)

        self.train() # Set model back to training mode if it was in training before generate
        return idx

