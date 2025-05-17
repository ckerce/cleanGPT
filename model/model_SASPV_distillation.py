# ./model/model_SASPV.py
"""
SASP Transformer Model Implementation
Based on the Simplified Attention Sub-Block with Projections and Value options (SAS-PV)
MODIFIED FOR DISTILLATION: Added output_hidden_states capability.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# Assuming SASLayerNorm, SASMLP, CausalShapedAttention, SimplifiedTransformerBlock
# are defined as in your original file. For brevity, skipping their re-definition.
# Make sure they are present in your actual file.

class SASLayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        # It's generally better to let the main model training loop handle dtype conversions (e.g. autocast)
        # rather than casting parameters directly during init, unless there's a specific reason.
        # For now, removing direct .to(bfloat16) here.

    def forward(self, input_tensor):
        orig_dtype = input_tensor.dtype
        # Promote to float32 for layernorm calculation if input is float16/bfloat16 for stability
        input_for_norm = input_tensor
        if input_tensor.dtype == torch.bfloat16 or input_tensor.dtype == torch.float16:
            input_for_norm = input_tensor.float()

        # Ensure weights/bias are also float32 for the calculation if input was promoted
        weight = self.weight.float() if input_for_norm.dtype == torch.float32 else self.weight
        bias = self.bias.float() if self.bias is not None and input_for_norm.dtype == torch.float32 else self.bias

        output = F.layer_norm(input_for_norm, weight.shape, weight, bias, 1e-5)
        return output.to(orig_dtype) # Cast back to original dtype


class SASMLP(nn.Module):
    """ MLP block as defined for SASP """
    def __init__(self, config):
        super().__init__()
        self.config = config # Store config
        self.llama_mlp = getattr(config, 'llama_mlp', False) # LLaMA-style MLP (SwiGLU-like)
        
        hidden_dim_multiplier = 4 # Standard for many transformers
        if self.llama_mlp:
            # For SwiGLU, the hidden dimension is often 2/3 * (4 * n_embd), rounded.
            # Or a common factor is 2.666 * n_embd. Let's use a common approximation:
            # hidden_dim = int( (4 * config.n_embd * 2 / 3) / 256) * 256 # Multiple of 256
            # A simpler approach often used:
            hidden_dim = int(8 / 3 * config.n_embd) # ~2.66x
            hidden_dim = (hidden_dim + 7) // 8 * 8 # Make it multiple of 8 for efficiency
            
            self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias) # Equivalent to c_fc
            self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias) # Equivalent to c_gate
            self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=config.bias) # Equivalent to c_proj
            self.actv_fn = F.silu # Swish/SiLU activation for LLaMA style
        else:
            # Standard MLP
            hidden_dim = hidden_dim_multiplier * config.n_embd
            self.c_fc    = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
            self.actv    = nn.GELU(approximate='tanh' if hasattr(config, 'gelu_approximate') and config.gelu_approximate else 'none')
            self.c_proj  = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        if self.llama_mlp:
            # SwiGLU: F.silu(w1(x)) * w3(x), then projected by w2
            output = self.w2(self.actv_fn(self.w1(x)) * self.w3(x))
        else:
            # Standard MLP: proj(actv(fc(x)))
            output = self.c_proj(self.actv(self.c_fc(x)))
            
        output = self.dropout(output)
        return output


class CausalShapedAttention(nn.Module):
    """ Causal Shaped Attention block """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config # Store config
        self.use_proj = getattr(config, 'use_proj', False) # Use projection in attention output
        self.use_v = getattr(config, 'use_v', False)       # Use separate Value vector in attention
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout_rate = config.dropout # Renamed from self.dropout to avoid conflict with nn.Dropout module
        self.max_block_size = config.block_size 
        self.bias = config.bias 

        # Output projection, if designated
        if self.use_proj:
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=self.bias)
        else:
            self.c_proj = nn.Identity() # No projection

        # QKV or just QK parameters
        # If use_v is True, we project for Q, K, and V. Otherwise, just Q, K.
        self.num_var_to_pack = (3 if self.use_v else 2)
        self.c_attn = nn.Linear(config.n_embd, self.num_var_to_pack * config.n_embd, bias=self.bias)

        # Attention dropout layer
        self.attn_dropout = nn.Dropout(self.dropout_rate) # Use the stored rate

        # Parameters specific to the shaped attention (alpha, beta, gamma)
        # Using float32 for these scalar parameters is generally fine.
        param_dtype = torch.float32
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=param_dtype))
        self.beta = nn.Parameter(torch.tensor(0.1, dtype=param_dtype))
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=param_dtype))

        # Register causal mask (MC) and identity components (Id) as buffers
        # These are fixed based on max_block_size.
        # Using bool for tril and then converting to float for softmax might be slightly cleaner.
        tril_ones = torch.tril(torch.ones(self.max_block_size, self.max_block_size, dtype=torch.float32))
        
        # MC mask: Cumulative sum, softmax normalized. Should emphasize earlier positions.
        # A sharp softmax for MC (cumulative mask)
        mc_mask_unnorm = tril_ones * 1e4 # Scale before softmax to make it sharp
        mc_mask = F.softmax(mc_mask_unnorm, dim=-1)
        self.register_buffer("MC", mc_mask.view(1, 1, self.max_block_size, self.max_block_size), persistent=False)

        # Id mask: Identity matrix, softmax normalized. Should emphasize current position.
        eye_matrix = torch.eye(self.max_block_size, dtype=torch.float32)
        id_mask_unnorm = eye_matrix * 1e4 # Scale before softmax for sharpness
        id_mask = F.softmax(id_mask_unnorm, dim=-1)
        self.register_buffer("Id", id_mask.view(1, 1, self.max_block_size, self.max_block_size), persistent=False)
        
        # Standard causal mask for attention scores (before SASP components)
        # This is the typical upper-triangular mask.
        causal_mask_bool = torch.tril(torch.ones(self.max_block_size, self.max_block_size, dtype=torch.bool)).logical_not()
        self.register_buffer("causal_mask_att", causal_mask_bool.view(1, 1, self.max_block_size, self.max_block_size), persistent=False)


        # Initialize parameters according to SASP paper recommendations
        self.custom_variable_initialization()


    def custom_variable_initialization(self):
        # Shaped attention has parameter initialization conditions
        with torch.no_grad():
            self.alpha.fill_(1.0)
            self.beta.fill_(0.1)
            self.gamma.fill_(0.1)
            
            # Initialize K weights (second chunk) and V weights (third chunk, if use_v) to zero.
            # Q is the first chunk.
            start_idx_k = self.n_embd  # Start index for K weights
            end_idx_k = 2 * self.n_embd # End index for K weights
            
            if hasattr(self.c_attn, 'weight') and self.c_attn.weight is not None: # Ensure weight exists (not meta init)
                 self.c_attn.weight.data[start_idx_k:end_idx_k, :].fill_(0.0)
                 if self.bias and self.c_attn.bias is not None:
                     self.c_attn.bias.data[start_idx_k:end_idx_k].fill_(0.0)
                 
                 if self.use_v:
                     start_idx_v = 2 * self.n_embd # Start index for V weights
                     end_idx_v = 3 * self.n_embd   # End index for V weights
                     self.c_attn.weight.data[start_idx_v:end_idx_v, :].fill_(0.0)
                     if self.bias and self.c_attn.bias is not None:
                         self.c_attn.bias.data[start_idx_v:end_idx_v].fill_(0.0)


    def forward(self, x):
        B, T, C = x.size() # Batch size, sequence length, embedding dimensionality (n_embd)
        assert T <= self.max_block_size, f"Sequence length {T} exceeds maximum block size {self.max_block_size}"

        # Calculate query, key, values for all heads in batch
        # q, k, v are [B, nh, T, hs] where hs = C // nh (head size)
        
        if self.use_v:
            q_proj, k_proj, v_proj = self.c_attn(x).split(self.n_embd, dim=2)
            v = v_proj.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        else:
            q_proj, k_proj = self.c_attn(x).split(self.n_embd, dim=2)
            # If not using a separate V projection, V is effectively x (the input to the attention)
            # This is a common setup for attention without a distinct value projection.
            v = x.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q = q_proj.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k_proj.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)


        # --- Shaped Attention Calculation ---
        # 1. Standard Scaled Dot-Product Attention Score
        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att_scores = (q @ k.transpose(-2, -1)) * (k.size(-1)**-0.5) 

        # 2. Apply Causal Mask for standard attention
        # self.causal_mask_att is (1,1,max_T,max_T), True for elements to mask
        current_causal_mask = self.causal_mask_att[:, :, :T, :T]
        att_scores = att_scores.masked_fill(current_causal_mask, float('-inf'))

        # 3. Softmax for standard attention weights
        att_weights_standard = F.softmax(att_scores, dim=-1) # (B, nh, T, T)
        
        # 4. Apply SASP modifications (alpha, beta, gamma)
        # Retrieve Id and MC buffers sliced to current sequence length T
        Id_T = self.Id[:, :, :T, :T] # (1, 1, T, T)
        MC_T = self.MC[:, :, :T, :T] # (1, 1, T, T)

        # Combine components: beta * standard_attention_weights + alpha * Identity - gamma * Cumulative
        shaped_att_weights = self.beta * att_weights_standard + \
                             self.alpha * Id_T - \
                             self.gamma * MC_T
        
        # Apply dropout to the final shaped attention weights
        shaped_att_weights_dropped = self.attn_dropout(shaped_att_weights)

        # 5. Weighted sum of values using the shaped attention weights
        y = shaped_att_weights_dropped @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)

        # Apply final output projection if enabled
        y = self.c_proj(y) # c_proj is nn.Identity if use_proj is False

        return y


class SimplifiedTransformerBlock(nn.Module):
    """ Simplified Transformer block using SASP components. Pre-LN structure. """
    def __init__(self, config):
        super().__init__()
        self.config = config # Store config
        self.ln_1 = SASLayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalShapedAttention(config)
        # This dropout is often applied after the sum of attention/mlp output and residual.
        # If it's meant to be a residual dropout, it should be applied after x = x + ...
        # self.dropout = nn.Dropout(config.dropout) # Consider if this is needed here or after residual
        
        self.ln_2 = SASLayerNorm(config.n_embd, bias=config.bias) # Added for Pre-LN MLP
        self.mlp = SASMLP(config)
        
        # Parameters for scaling attention and MLP contributions (as in SASP paper)
        # Using float32 for these scalar parameters is fine.
        param_dtype = torch.float32
        self.beta_SA = nn.Parameter(torch.tensor(1.0, dtype=param_dtype)) # Scale for Self-Attention
        self.beta_FF = nn.Parameter(torch.tensor(0.2, dtype=param_dtype)) # Scale for Feed-Forward

        self.initialize_parameters()

    def initialize_parameters(self):
        # Initialization specific to SASP block structure (beta_SA, beta_FF)
        with torch.no_grad():
            self.beta_SA.fill_(1.0)
            self.beta_FF.fill_(0.2)
        # Attention layer (CausalShapedAttention) has its own custom_variable_initialization

    def forward(self, x):
        # Pre-LN structure: LN -> Attention -> Residual -> LN -> MLP -> Residual
        
        # Attention part
        # The original SASP paper implies a slightly different structure where ln_1 normalizes input to both attn and mlp
        # and then their outputs are scaled and summed.
        # x_norm = self.ln_1(x)
        # attn_output = self.attn(x_norm)
        # mlp_output = self.mlp(x_norm)
        # x = x + self.beta_SA * attn_output + self.beta_FF * mlp_output # Original SASP formulation
        
        # Implementing a more standard Pre-LN block structure with SASP components:
        # x = x + dropout(attn(ln1(x)))
        # x = x + dropout(mlp(ln2(x)))
        # Here, beta_SA and beta_FF act as scaling factors for the residual additions.

        attn_output = self.attn(self.ln_1(x))
        x = x + self.beta_SA * attn_output # Scaled residual connection for attention

        mlp_output = self.mlp(self.ln_2(x))
        x = x + self.beta_FF * mlp_output # Scaled residual connection for MLP
        
        return x


class SASPTransformerModel(nn.Module):
    """
    Transformer model using Simplified Attention Sub-Block (SASP) architecture
    """
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None, "vocab_size must be specified in config"
        assert config.block_size is not None, "block_size must be specified in config"
        self.config = config # Store config
        self.padding_idx = getattr(config, 'padding_idx', None)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=self.padding_idx),
            wpe = nn.Embedding(config.block_size, config.n_embd), # Positional embeddings
            drop = nn.Dropout(config.dropout), # Dropout for embeddings
            # Create a list of transformer blocks
            h = nn.ModuleList([SimplifiedTransformerBlock(config) for _ in range(config.n_layer)]),
            # Final LayerNorm after all blocks
            ln_f = SASLayerNorm(config.n_embd, bias=config.bias),
        ))

        # Output layer (language modeling head)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Tying: Share weights between token embeddings and final output layer
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Apply special scaled init to residual projections (if they exist in MLP/Attention c_proj)
        # This is a common practice (e.g., GPT-2)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'): # Targets c_proj in SASMLP and CausalShapedAttention
                module_name = '.'.join(pn.split('.')[:-1])
                module = self.get_submodule(module_name)
                if isinstance(module, nn.Linear): # Ensure it's a Linear layer
                    torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        
        num_params = self.get_num_params()
        num_params_non_emb = self.get_num_params(non_embedding=True)
        print(f"SASPTransformerModel initialized with {num_params/1e6:.2f}M parameters ({num_params_non_emb/1e6:.2f}M non-embedding)")


    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        Token embeddings are included due to weight tying with lm_head.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
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
            if module.padding_idx is not None:
                 with torch.no_grad():
                     module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, SASLayerNorm):
             # Bias is initialized to zeros, weight to ones by default in SASLayerNorm init
             if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
             torch.nn.init.ones_(module.weight)
        # CausalShapedAttention and SimplifiedTransformerBlock handle their specific params (alpha, beta, gamma, beta_SA, beta_FF)
        # in their own initialization methods.


    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass for the SASP Transformer model.
        input_ids: (batch, seq_len)
        labels: (batch, seq_len) - Typically input_ids shifted right, for loss calculation.
        attention_mask: (batch, seq_len) - Optional, primarily for embedding padding_idx.
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

        # --- MODIFICATION FOR DISTILLATION ---
        all_hidden_states = []
        if self.config.output_hidden_states:
            # Store initial embedding output (hidden_states[0] in HF)
            all_hidden_states.append(x.clone())
        # --- END MODIFICATION ---

        # 3. Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)
            # --- MODIFICATION FOR DISTILLATION ---
            if self.config.output_hidden_states:
                all_hidden_states.append(x.clone()) # Store state *after* each block
            # --- END MODIFICATION ---

        # 4. Final LayerNorm
        x = self.transformer.ln_f(x)

        # 5. Language Modeling Head (Output Layer)
        logits = self.lm_head(x) # shape (b, t, vocab_size)

        # --- Calculate Loss ---
        loss = None
        if labels is not None:
            # Standard causal LM loss, assuming labels are correctly shifted by the dataloader
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100) # -100 for padding
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # --- MODIFICATION FOR DISTILLATION ---
        return_dict = {'loss': loss, 'logits': logits}
        if self.config.output_hidden_states:
            return_dict['hidden_states'] = all_hidden_states # Generic key
        # --- END MODIFICATION ---
        return return_dict

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.
        idx: (b, t) tensor of indices in the current context.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Scaling factor for logits before sampling.
        top_k: If set, only sample from the top k most likely tokens.
        """
        self.eval() 

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            original_output_hs_config = self.config.output_hidden_states
            self.config.output_hidden_states = False # Disable for generation efficiency
            
            outputs = self(idx_cond)
            
            self.config.output_hidden_states = original_output_hs_config # Restore

            logits = outputs['logits']
            logits = logits[:, -1, :] / temperature 
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

