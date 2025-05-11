# ./model/model_SASPV.py 
"""
SASP Transformer Model Implementation
Based on the Simplified Attention Sub-Block with Projections and Value options (SAS-PV)
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class SASLayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        # Use bfloat16 if available and desired, otherwise default dtype
        self.weight.data = self.weight.data.to(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32)

    def forward(self, input):
        # Ensure input is in the correct dtype for layer norm
        orig_dtype = input.dtype
        input_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        return F.layer_norm(input.to(input_dtype), self.weight.shape, self.weight, self.bias, 1e-5).to(orig_dtype)

class SASMLP(nn.Module):
    """ MLP block as defined for SASP """
    def __init__(self, config):
        super().__init__()
        self.llama_mlp = getattr(config, 'llama_mlp', False)
        if self.llama_mlp:
            # Adjusted calculation for SwiGLU-like structure in LLaMA
            # N is often around 2/3 * (4 * n_embd), rounded to nearest multiple (e.g., of 256)
            # Here using the provided formula, ensure it's an integer
            N = int(2.1 / 3. * 4 * config.n_embd)
            # Ensure N is reasonable, maybe enforce multiple of 8 or more?
            N = (N + 7) // 8 * 8 # Example: round up to multiple of 8
            self.c_gate = nn.Linear(config.n_embd, N, bias=config.bias) # Renamed c_mask to c_gate for clarity
        else:
            N = int(4 * config.n_embd)

        self.c_fc    = nn.Linear(config.n_embd, N, bias=config.bias)
        self.actv    = nn.GELU() # GELU is common, Swish/SiLU used in LLaMA
        self.c_proj  = nn.Linear(N, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        if self.llama_mlp:
            # Implements SwiGLU-like activation: Swish(gate(x)) * fc(x)
            # GELU is used here as per original code, but Swish/SiLU is typical for SwiGLU
            x_fc = self.c_fc(x)
            x_gate = self.c_gate(x)
            x = self.actv(x_gate) * x_fc # Element-wise product
        else:
            x = self.c_fc(x)
            x = self.actv(x)

        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class CausalShapedAttention(nn.Module):
    """ Causal Shaped Attention block """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.use_proj = getattr(config, 'use_proj', False)
        self.use_v = getattr(config, 'use_v', False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.max_block_size = config.block_size # Use block_size from config
        self.bias = config.bias # Store bias setting

        # Allocate output projection, if designated
        if self.use_proj:
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=self.bias)

        # Allocate QKV or just QK parameters
        self.num_var_to_pack = (3 if self.use_v else 2)
        self.c_attn = nn.Linear(config.n_embd, self.num_var_to_pack * config.n_embd, bias=self.bias)

        # Attention dropout
        self.attn_dropout = nn.Dropout(config.dropout)

        # Use bfloat16 if supported for parameters and buffers
        param_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        buffer_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

        # Parameters specific to the shaped attention
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=param_dtype))
        self.beta = nn.Parameter(torch.tensor(0.1, dtype=param_dtype))
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=param_dtype))

        # Register causal mask and identity components as buffers
        tril = torch.tril(torch.ones(self.max_block_size, self.max_block_size, dtype=buffer_dtype))
        # Softmax for MC ensures rows sum to 1, focusing on cumulative effect
        mc_mask = F.softmax(1e4 * tril, dim=-1) # Use large factor for sharp cumulative mask
        self.register_buffer("MC", mc_mask.view(1, 1, self.max_block_size, self.max_block_size), persistent=False)
        # Softmax for Id ensures rows sum to 1, focusing on identity
        id_mask = F.softmax(torch.eye(self.max_block_size, dtype=buffer_dtype) * 1e4, dim=-1) # Sharp identity
        self.register_buffer("Id", id_mask.view(1, 1, self.max_block_size, self.max_block_size), persistent=False)

        # Initialize parameters according to SASP paper recommendations
        self.custom_variable_initialization()


    def custom_variable_initialization(self):
        # shaped attention has parameter initialization conditions
        with torch.no_grad():
            self.alpha.fill_(1.0)
            self.beta.fill_(0.1)
            self.gamma.fill_(0.1)
            # Initialize K weights (second chunk) to zero if not using V
            # If using V, K is the second chunk, V is the third.
            # Initialize K and V weights (if V exists) to zero.
            start_idx = self.n_embd # Start index for K weights
            end_idx = self.num_var_to_pack * self.n_embd # End index for K (or V if exists)
            if hasattr(self.c_attn, 'weight'): # Ensure weight exists (not meta init)
                 self.c_attn.weight.data[start_idx:end_idx, :].fill_(0.0)
                 if self.bias and self.c_attn.bias is not None:
                     self.c_attn.bias.data[start_idx:end_idx].fill_(0.0)


    def forward(self, x):
        B, T, C = x.size() # Batch size, sequence length, embedding dimensionality (n_embd)
        assert T <= self.max_block_size, f"Sequence length {T} exceeds maximum block size {self.max_block_size}"

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # q, k, v: [B, nh, T, hs] where hs = C // nh (head size)
        q, k, *v_packed = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Use x directly if no V is used in the attention, otherwise unpack v
        v = v_packed[0] if self.use_v else x
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # --- Shaped Attention Calculation ---
        # 1. Standard Scaled Dot-Product Attention Score
        att_scores = (q @ k.transpose(-2, -1)) * (k.size(-1)**-0.5) # More stable scaling

        # 2. Apply Causal Mask
        # Retrieve the buffers, slicing to the current sequence length T
        # The buffers are (1, 1, max_T, max_T)
        causal_mask = self.MC.logical_not()[:, :, :T, :T] # Get mask where MC is 0 (upper triangle)
        att_scores = att_scores.masked_fill(causal_mask, float('-inf')) # Fill upper triangle with -inf

        # 3. Apply SASP modifications (alpha, beta, gamma)
        # Retrieve Id and MC buffers sliced to current sequence length T
        Id_T = self.Id[:, :, :T, :T] # (1, 1, T, T)
        MC_T = self.MC[:, :, :T, :T] # (1, 1, T, T)

        # Softmax applied *before* adding Id and MC components, scaled by beta
        att_weights = F.softmax(att_scores, dim=-1) # (B, nh, T, T)
        shaped_att = self.beta * att_weights
        shaped_att = shaped_att + self.alpha * Id_T - self.gamma * MC_T # Add Id, subtract MC

        # Apply dropout to the final attention weights
        shaped_att = self.attn_dropout(shaped_att)

        # 4. Weighted sum of values
        y = shaped_att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)

        # Apply final output projection if enabled
        if self.use_proj:
            y = self.c_proj(y)

        return y

class SimplifiedTransformerBlock(nn.Module):
    """ Simplified Transformer block using SASP components """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = SASLayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalShapedAttention(config)
        self.mlp = SASMLP(config)
        # Use bfloat16 if supported for parameters
        param_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.beta_SA = nn.Parameter(torch.tensor(1.0, dtype=param_dtype))
        self.beta_FF = nn.Parameter(torch.tensor(0.2, dtype=param_dtype))
        # Perform initialization after parameters are created
        self.initialize_parameters()

    def initialize_parameters(self):
        # Initialization specific to SASP block structure
        with torch.no_grad():
            self.beta_SA.fill_(1.0)
            self.beta_FF.fill_(0.2)
        # Attention layer has its own custom init called within its __init__

    def forward(self, x):
        # Apply LayerNorm first (as suggested by the structure, ln_1 before attn/mlp)
        norm_x = self.ln_1(x)
        # Calculate attention and mlp outputs based on the normalized input
        attn_output = self.attn(norm_x)
        mlp_output = self.mlp(norm_x)
        # Combine using learned beta parameters - This is the SASP formulation
        # Note: This differs from standard residual connection (x = x + attn_output + mlp_output)
        #x = self.beta_SA * attn_output + self.beta_FF * mlp_output

        # Implementing standard skip connection for the shaped attention for convergence excursion study
        x = x + self.beta_SA * attn_output + self.beta_FF * mlp_output
        return x


class SASPTransformerModel(nn.Module):
    """
    Transformer model using Simplified Attention Sub-Block (SASP) architecture
    """
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.padding_idx = getattr(config, 'padding_idx', None)

        # Define the model architecture using nn.ModuleDict
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=self.padding_idx),
            wpe = nn.Embedding(config.block_size, config.n_embd), # Positional embeddings
            drop = nn.Dropout(config.dropout),
            # Create a list of transformer blocks
            h = nn.ModuleList([SimplifiedTransformerBlock(config) for _ in range(config.n_layer)]),
            # Final LayerNorm
            ln_f = SASLayerNorm(config.n_embd, bias=config.bias),
        ))

        # Output layer (language modeling head)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Tying: Share weights between token embeddings and final output layer
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Apply special scaled init to residual projections (if they exist in MLP/Attention)
        for pn, p in self.named_parameters():
            # Target projection layers within MLP and Attention (if use_proj=True)
            if pn.endswith('c_proj.weight'):
                # Scale initialization variance based on depth (GPT-2 paper recommendation)
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"SASPTransformerModel initialized with {self.get_num_params()/1e6:.2f}M parameters")


    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Subtract positional embedding parameters if not counting embeddings
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """ Initialize weights according to common practices """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # Zero out padding embedding if it exists
            if module.padding_idx is not None:
                 with torch.no_grad():
                     module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, SASLayerNorm):
             # Bias is initialized to zeros, weight to ones by default in SASLayerNorm
             pass


    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass for the SASP Transformer model.
        input_ids: (batch, seq_len)
        labels: (batch, seq_len) - Typically input_ids shifted right, for loss calculation.
        attention_mask: (batch, seq_len) - Optional, used for padding.
                        (Not explicitly used by CausalShapedAttention's masking,
                         but could be used by embedding padding_idx)
        """
        device = input_ids.device
        b, t = input_ids.size() # Batch size, sequence length
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Create positional indices: 0, 1, 2, ..., t-1
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # --- Forward through transformer layers ---
        # 1. Get token embeddings and positional embeddings
        tok_emb = self.transformer.wte(input_ids) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)

        # 2. Combine embeddings and apply dropout
        x = self.transformer.drop(tok_emb + pos_emb)

        # 3. Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # 4. Final LayerNorm
        x = self.transformer.ln_f(x)

        # 5. Language Modeling Head (Output Layer)
        # Project final embeddings to vocabulary size to get logits
        logits = self.lm_head(x) # shape (b, t, vocab_size)

        # --- Calculate Loss ---
        loss = None
        if labels is not None:
            # Calculate Cross Entropy Loss
            # Flatten the logits and labels for the loss function
            # Logits: (batch * seq_len, vocab_size)
            # Labels: (batch * seq_len)
            # Use ignore_index=-100, which is the standard for Hugging Face collators
            # when padding labels.
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Return dictionary compatible with Trainer
        return {'loss': loss, 'logits': logits}

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.
        idx: (b, t) tensor of indices in the current context.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Scaling factor for logits before sampling ( > 1.0 = more random, < 1.0 = more deterministic).
        top_k: If set, only sample from the top k most likely tokens.
        """
        self.eval() # Set model to evaluation mode

        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens if sequence gets too long
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass to get logits for the *next* token
            outputs = self(idx_cond) # Pass current sequence
            logits = outputs['logits'] # Shape (b, t, vocab_size)

            # Focus only on the logits for the last time step
            logits = logits[:, -1, :] / temperature # Shape (b, vocab_size)

            # Optional: Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Set logits not in the top k to -inf
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply softmax to convert logits to probabilities
            probs = F.softmax(logits, dim=-1) # Shape (b, vocab_size)

            # Sample the next token index from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1) # Shape (b, 1)

            # Append the sampled token index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # Shape (b, t+1)

        self.train() # Set model back to training mode if needed elsewhere
        return idx # Return the generated sequence including the initial context
