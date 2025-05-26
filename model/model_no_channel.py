# -*- coding: utf-8 -*-
"""
Vanilla Transformer model with Pre-Layer Normalization and Modified Attention
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

    def forward(self, input_tensor): # Renamed 'input' to 'input_tensor' to avoid conflict
        orig_dtype = input_tensor.dtype
        # Determine the compute dtype, bfloat16 if supported, else float32
        input_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        
        # Cast input, weight, and bias to the compute dtype
        casted_input = input_tensor.to(input_dtype)
        casted_weight = self.weight.to(input_dtype)
        casted_bias = self.bias.to(input_dtype) if self.bias is not None else None
        
        # Perform layer normalization
        normalized_output = F.layer_norm(
            casted_input,
            casted_weight.shape, # Shape of the weights, e.g., (ndim,)
            casted_weight,
            casted_bias,
            1e-5  # eps, for numerical stability
        )
        
        # Cast the output back to the original dtype
        return normalized_output.to(orig_dtype)


class CausalSelfAttention(nn.Module):
    """
    Modified multi-head self-attention with causal mask.
    The embedding vector x is used directly as the value (V), avoiding channelization for V.
    The output is a weighted sum of attention-applied inputs over the heads:
    x_out = sum_h (w_h * (A_h @ x_in)), where w_h is a learnable vector per head.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads"

        # Key, Query projections (Value V is the input x itself)
        self.c_attn = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias) # Modified for Q, K only

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout) # Dropout for the final output of this block

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout # Storing dropout rate from config

        # Learnable weights for combining head outputs.
        # Each head h has a weight vector w_h of size n_embd.
        self.head_weights = nn.Parameter(torch.empty(self.n_head, config.n_embd))
        # Initialize weights (e.g., Xavier uniform)
        torch.nn.init.xavier_uniform_(self.head_weights)

        # Causal mask to ensure that attention is only applied to the left in the input sequence
        # Create it as a persistent buffer, so it's part of the model's state_dict but not a parameter
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)) # Use bool for masked_fill
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        # x shape: (B, T, C) where B=batch_size, T=sequence_length, C=embedding_dimensionality (n_embd)
        B, T, C = x.size() 

        # Calculate query and key for all heads in batch
        # Value (v) will be x itself.
        q_proj, k_proj = self.c_attn(x).split(self.n_embd, dim=2) # q_proj, k_proj: (B, T, C)

        # Reshape Q, K for multi-head attention
        # hs (head_size) = C // self.n_head
        hs = C // self.n_head
        k = k_proj.view(B, T, self.n_head, hs).transpose(1, 2) # (B, nh, T, hs)
        q = q_proj.view(B, T, self.n_head, hs).transpose(1, 2) # (B, nh, T, hs)
        # v is x, which is (B, T, C). We'll use it later.

        # Causal self-attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # Calculate attention scores (A_h for each head)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # k.size(-1) is hs
        
        # Apply the causal mask to prevent attending to future positions
        # The mask is (1, 1, block_size, block_size). We need to slice it for the current sequence length T.
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        
        # Apply softmax to get attention probabilities
        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        
        # Apply dropout to attention probabilities
        att = self.attn_dropout(att)

        # Apply attention to the input x (which serves as V)
        # x_in is x: (B, T, C)
        # A_h is att[:, h, :, :] : (B, T, T) for a specific head h in the batch
        # We want to compute sum_h (w_h * (A_h @ x_in))
        
        # To do this efficiently:
        # Expand x to be compatible with batched matrix multiplication with att
        # x_expanded shape: (B, 1, T, C) to allow broadcasting over the head dimension
        x_expanded = x.unsqueeze(1) 
        
        # Perform batched matrix multiplication: (B, nh, T, T) @ (B, 1, T, C) -> (B, nh, T, C)
        # Each attended_x_heads[b, h, :, :] is effectively A_h @ x[b, :, :]
        attended_x_heads = torch.matmul(att, x_expanded) # (B, nh, T, C)

        # Reshape head_weights for broadcasting: (1, nh, 1, C)
        # self.head_weights is (nh, C)
        weights_reshaped = self.head_weights.unsqueeze(0).unsqueeze(2) 

        # Element-wise multiplication by head-specific weights w_h (which are vectors of size C)
        # (B, nh, T, C) * (1, nh, 1, C) -> (B, nh, T, C)
        weighted_attended_heads = attended_x_heads * weights_reshaped
        
        # Sum over the heads dimension (dim=1)
        # (B, nh, T, C) -> (B, T, C)
        y = weighted_attended_heads.sum(dim=1)

        # Apply final residual dropout
        y = self.resid_dropout(y)
        
        return y


class MLP(nn.Module):
    """Simple MLP with GELU activation and dropout."""
    def __init__(self, config):
        super().__init__()
        # First fully connected layer, expanding dimensionality
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # GELU activation function
        self.activation = nn.GELU()
        # Second fully connected layer, projecting back to original dimensionality
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        # Dropout layer
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class PreLNBlock(nn.Module):
    """Transformer block with Pre-Layer Normalization."""
    def __init__(self, config):
        super().__init__()
        # First Layer Normalization
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        # Causal Self-Attention module
        self.attn = CausalSelfAttention(config) # Uses the modified attention
        # Second Layer Normalization
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        # MLP module
        self.mlp = MLP(config)

    def forward(self, x):
        # Pre-LN architecture: apply layer norm before each sub-block, then add residual connection
        # Attention sub-block
        x = x + self.attn(self.ln_1(x))
        # MLP sub-block
        x = x + self.mlp(self.ln_2(x))
        return x


class VanillaTransformerModel(nn.Module):
    """
    Standard Transformer model using Pre-Layer Normalization and the modified CausalSelfAttention.
    """
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None, "vocab_size must be defined in config"
        assert config.block_size is not None, "block_size must be defined in config"
        self.config = config
        self.padding_idx = getattr(config, 'padding_idx', None) # Optional padding index for embeddings

        # Model components dictionary
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=self.padding_idx), # Word token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # Word position embeddings
            drop = nn.Dropout(config.dropout), # Dropout after embedding sum
            h = nn.ModuleList([PreLNBlock(config) for _ in range(config.n_layer)]), # List of transformer blocks
            ln_f = LayerNorm(config.n_embd, bias=config.bias), # Final Layer Normalization
        ))

        # Output head for language modeling
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between token embeddings and the final linear layer
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights for the model
        self.apply(self._init_weights)
        
        # Special scaled initialization for output projection in MLP blocks
        # (and potentially other c_proj layers if they existed elsewhere)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'): # This targets MLP's c_proj and any other similarly named layers
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        # Note: The c_proj within CausalSelfAttention was removed, so no special init for it there.
        # The new head_weights in CausalSelfAttention are initialized with Xavier uniform.

        print(f"VanillaTransformerModel initialized with {self.get_num_params()/1e6:.2f}M parameters")

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model.
        If non_embedding is True, excludes position embeddings from the count.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Subtract word position embedding parameters if non_embedding is True
            # Note: wte is tied with lm_head, so its parameters are already counted.
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """Initialize model weights according to common practices."""
        if isinstance(module, nn.Linear):
            # Normal distribution for linear layer weights
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                # Zero initialization for biases
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Normal distribution for embedding weights
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                # Zero out the embedding for the padding index
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, LayerNorm):
            # LayerNorm weights (gamma) are initialized to 1 and biases (beta) to 0 by default in its __init__
            pass # Handled by LayerNorm's own parameter initialization

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass of the transformer model.
        Args:
            input_ids (Tensor): Input token IDs of shape (batch_size, sequence_length).
            attention_mask (Tensor, optional): Mask to avoid performing attention on padding token indices.
                                              Shape (batch_size, sequence_length). Not explicitly used in this Causal model
                                              as causality is handled by the mask in CausalSelfAttention.
            labels (Tensor, optional): Labels for language modeling. Shape (batch_size, sequence_length).
        Returns:
            dict: A dictionary containing 'loss' (if labels are provided) and 'logits'.
        """
        device = input_ids.device
        b, t = input_ids.size() # Batch size, sequence length
        # Ensure sequence length does not exceed configured block size
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Create positional indices (0, 1, ..., t-1)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # Shape (1, t)

        # Get token and position embeddings
        tok_emb = self.transformer.wte(input_ids) # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)       # (1, t, n_embd) -> broadcasts to (b, t, n_embd)
        
        # Sum token and position embeddings, then apply dropout
        x = self.transformer.drop(tok_emb + pos_emb) # (b, t, n_embd)

        # Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x) # (b, t, n_embd)

        # Final layer normalization
        x = self.transformer.ln_f(x) # (b, t, n_embd)

        # Language model head: project to vocabulary size
        logits = self.lm_head(x) # (b, t, vocab_size)

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # CrossEntropyLoss expects logits of shape (N, C) and labels of shape (N)
            # N = b * t (batch_size * sequence_length), C = vocab_size
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100) # -100 is a common ignore index for padding
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {'loss': loss, 'logits': logits}

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.
        Args:
            idx (Tensor): Context sequence of token IDs, shape (batch_size, current_sequence_length).
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Softmax temperature for sampling. Higher values make output more random.
            top_k (int, optional): If set, only sample from the top_k most likely tokens.
        Returns:
            Tensor: The input sequence concatenated with the generated tokens.
        """
        self.eval() # Set model to evaluation mode

        for _ in range(max_new_tokens):
            # Crop context if it exceeds block size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass to get logits for the next token
            outputs = self(idx_cond) # The forward method handles position embeddings
            logits = outputs['logits'] # (batch_size, current_sequence_length, vocab_size)
            
            # Focus on the logits for the last token in the sequence
            logits = logits[:, -1, :] / temperature # (batch_size, vocab_size)
            
            # Optional top-k sampling: set logits of non-top-k tokens to -infinity
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf') # Apply top-k filtering
                
            # Convert logits to probabilities using softmax
            probs = F.softmax(logits, dim=-1) # (batch_size, vocab_size)
            
            # Sample the next token ID from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (batch_size, 1)
            
            # Append the sampled token ID to the sequence
            idx = torch.cat((idx, idx_next), dim=1) # (batch_size, current_sequence_length + 1)
            
        self.train() # Set model back to training mode
        return idx

# Example Usage (requires a config object):
if __name__ == '__main__':
    # Define a mock config for testing
    class MockConfig:
        def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout, bias=True, padding_idx=None):
            self.vocab_size = vocab_size
            self.n_embd = n_embd
            self.n_head = n_head
            self.n_layer = n_layer
            self.block_size = block_size
            self.dropout = dropout
            self.bias = bias
            self.padding_idx = padding_idx


    # Configuration parameters
    vocab_size = 50257 # Example: GPT-2 vocab size
    n_embd = 64     # Embedding dimension (C)
    n_head = 4      # Number of attention heads
    n_layer = 2     # Number of transformer blocks/layers
    block_size = 32 # Max sequence length for model context
    dropout_rate = 0.1
    
    config = MockConfig(
        vocab_size=vocab_size, 
        n_embd=n_embd, 
        n_head=n_head, 
        n_layer=n_layer, 
        block_size=block_size, 
        dropout=dropout_rate
    )

    # Instantiate the model
    model = VanillaTransformerModel(config)
    print(f"Model '{type(model).__name__}' instantiated successfully.")

    # Create a dummy input tensor and labels
    batch_size = 2
    seq_length = 15 # Should be <= block_size
    assert seq_length <= config.block_size
    
    dummy_input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    dummy_labels = torch.randint(0, config.vocab_size, (batch_size, seq_length)) # Dummy labels for loss calculation

    # Perform a forward pass
    try:
        outputs = model(dummy_input_ids, labels=dummy_labels)
        print(f"Forward pass successful.")
        print(f"Logits shape: {outputs['logits'].shape}") # Expected: (batch_size, seq_length, vocab_size)
        assert outputs['logits'].shape == (batch_size, seq_length, config.vocab_size)
        if outputs['loss'] is not None:
            print(f"Loss: {outputs['loss'].item()}")
        else:
            print("Loss was not calculated (no labels provided).")
            
        # Test generation
        print("\nTesting generation...")
        generated_sequence = model.generate(dummy_input_ids[:, :5], max_new_tokens=5, temperature=0.8, top_k=10)
        print(f"Generated sequence shape: {generated_sequence.shape}") # Expected: (batch_size, 5 + 5)
        assert generated_sequence.shape == (batch_size, 5 + 5)
        print(f"Generated sequence example (first batch):\n{generated_sequence[0]}")

    except Exception as e:
        print(f"Error during forward pass or generation: {e}")
        import traceback
        traceback.print_exc()

    # Example of checking parameters (optional)
    # print("\nModel Parameters:")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")


