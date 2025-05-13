# ./model/model_token_factored.py
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

# --- Add the new ChannelFactoredLinear class ---
class ChannelFactoredLinear(nn.Module):
    """
    A linear layer where the weights are channel-factored.
    The n_embd is treated as n_head * head_size.
    The projection is done by a (n_head x n_head) matrix that acts on the head dimension.
    """
    def __init__(self, n_embd, n_head, bias=True, config=None):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.config = config

        self.weight_channel = nn.Parameter(torch.empty(n_head, n_head))
        if bias:
            self.bias = nn.Parameter(torch.empty(n_embd))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights similar to the main model's _init_weights
        torch.nn.init.normal_(self.weight_channel, mean=0.0, std=0.02)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: (B, T, C) or (B, C)
        original_shape = x.shape
        B = original_shape[0]
        
        if x.ndim == 3: # (B, T, C)
            T = original_shape[1]
            x_reshaped = x.view(B, T, self.n_head, self.head_size)
            # Use transpose for matrix multiplication
            out_reshaped = torch.einsum('bthc, oh -> btoc', x_reshaped, self.weight_channel.T)
            out = out_reshaped.contiguous().view(B, T, self.n_embd)
        elif x.ndim == 2: # (B,C)
            x_reshaped = x.view(B, self.n_head, self.head_size)
            out_reshaped = torch.einsum('bhc, oh -> boc', x_reshaped, self.weight_channel.T)
            out = out_reshaped.contiguous().view(B, self.n_embd)
        else:
            raise ValueError("Input tensor must have 2 or 3 dimensions")

        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self):
        return f'n_embd={self.n_embd}, n_head={self.n_head}, bias={self.bias is not None}'


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input_tensor):
        orig_dtype = input_tensor.dtype
        input_for_norm = input_tensor
        if input_tensor.dtype == torch.bfloat16 or input_tensor.dtype == torch.float16:
            input_for_norm = input_tensor.float()
        
        weight = self.weight.float() if input_for_norm.dtype == torch.float32 else self.weight
        bias = self.bias.float() if self.bias is not None and input_for_norm.dtype == torch.float32 else self.bias
        
        output = F.layer_norm(input_for_norm, weight.shape, weight, bias, 1e-5)
        return output.to(orig_dtype)


class FactoredCausalSelfAttention(nn.Module):
    """
    Causal self-attention mechanism for the Factored Transformer.
    - Q and K are derived from norm(xt + xe).
    - V is effectively xt_current * V_base.
      - V_base is derived from norm(xt + xe) if use_v=True.
      - V_base can be standard projection or channel-factored.
    - Output projection can be standard or channel-factored if use_proj=True.
    The output of this block updates xt.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # Use standardized flags
        self.use_proj = config.use_proj
        self.use_v = config.use_v
        self.use_channel_factor_v = config.use_channel_factor_v
        self.use_channel_factor_proj = config.use_channel_factor_proj

        # Output projection
        if self.use_proj:
            if self.use_channel_factor_proj:
                self.c_proj = ChannelFactoredLinear(config.n_embd, config.n_head, bias=config.bias, config=config)
            else:
                self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        else:
            self.c_proj = nn.Identity() # No projection

        # V-specific projection layer (if use_channel_factor_v is True and use_v is True)
        self.v_fc_layer = None
        num_qkv_projs = 2 # For Q, K by default

        if self.use_v:
            if self.use_channel_factor_v:
                self.v_fc_layer = ChannelFactoredLinear(config.n_embd, config.n_head, bias=config.bias, config=config)
                # Q, K still come from c_attn
            else:
                num_qkv_projs = 3 # Q, K, V_orig from c_attn
        
        # Key, Query, and potentially V_orig projections
        self.c_attn = nn.Linear(config.n_embd, num_qkv_projs * config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        # resid_dropout is applied *after* c_proj (or identity if not used)
        self.resid_dropout = nn.Dropout(config.dropout) 
        
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x_norm_for_qkv, xt_current_for_v):
        B, T, C = x_norm_for_qkv.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        v_orig_from_c_attn = None
        if self.use_v and not self.use_channel_factor_v: # Q, K, V_orig from c_attn
            q, k, v_orig_from_c_attn = self.c_attn(x_norm_for_qkv).split(self.n_embd, dim=2)
        else: # Q, K from c_attn; V handled separately or not used
            q, k = self.c_attn(x_norm_for_qkv).split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Prepare xt_current_for_v for modulation or as direct V
        xt_reshaped_for_v = xt_current_for_v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)

        if self.use_v:
            if self.use_channel_factor_v:
                assert self.v_fc_layer is not None
                base_v = self.v_fc_layer(x_norm_for_qkv) # (B, T, C)
            else:
                assert v_orig_from_c_attn is not None
                base_v = v_orig_from_c_attn # (B, T, C)
            
            base_v_reshaped = base_v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            effective_v = xt_reshaped_for_v * base_v_reshaped # Element-wise multiplication
        else:
            # If not using a separate V projection, V is effectively xt_current_for_v
            effective_v = xt_reshaped_for_v

        att_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att_scores = att_scores.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att_weights = F.softmax(att_scores, dim=-1)
        att_weights = self.attn_dropout(att_weights)
        
        y = att_weights @ effective_v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)
        
        # Output projection (if enabled) and dropout
        y = self.c_proj(y) # c_proj is nn.Identity() if use_proj is False
        y = self.resid_dropout(y)
        return y


class FactoredMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x_norm):
        x = self.c_fc(x_norm)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class FactoredPreLNBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = FactoredCausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = FactoredMLP(config)

    def forward(self, xt, xe):
        norm_for_attn_qkv = self.ln_1(xt + xe)
        attn_output = self.attn(x_norm_for_qkv=norm_for_attn_qkv, xt_current_for_v=xt)
        xt = xt + attn_output

        norm_for_mlp = self.ln_2(xt + xe) 
        mlp_output = self.mlp(norm_for_mlp)
        xe = xe + mlp_output
        
        return xt, xe


class FactoredTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None, "vocab_size must be specified in config"
        assert config.block_size is not None, "block_size must be specified in config"
        self.config = config
        self.padding_idx = getattr(config, 'padding_idx', None) 

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=self.padding_idx), 
            wpe = nn.Embedding(config.block_size, config.n_embd), 
            drop = nn.Dropout(config.dropout), 
            h = nn.ModuleList([FactoredPreLNBlock(config) for _ in range(config.n_layer)]), 
            ln_f = LayerNorm(config.n_embd, bias=config.bias), 
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # Weight tying
        
        self.apply(self._init_weights)
        
        # Special init for projection weights in MLP and Attention
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'): # This applies to nn.Linear c_proj
                # For ChannelFactoredLinear, its main param is weight_channel, initialized in reset_parameters.
                if isinstance(p.module if hasattr(p, 'module') else self.get_submodule('.'.join(pn.split('.')[:-1])), nn.Linear):
                     torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"FactoredTransformerModel initialized with {self.get_num_params()/1e6:.2f}M parameters")

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad(): 
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, LayerNorm):
            # Weights are 1s, bias 0s by default in LayerNorm init
            pass
        elif isinstance(module, ChannelFactoredLinear):
            # Already handled by its own reset_parameters
            pass

    def forward(self, input_ids, attention_mask=None, labels=None):
        device = input_ids.device
        b, t = input_ids.size() 
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) 

        tok_emb = self.transformer.wte(input_ids) 
        pos_emb = self.transformer.wpe(pos)       
        
        xt = tok_emb + pos_emb 
        xt = self.transformer.drop(xt) 
        xe = torch.zeros_like(xt, device=device) 

        for block in self.transformer.h:
            xt, xe = block(xt, xe)

        x_final_combined = xt + xe
        x_final_normed = self.transformer.ln_f(x_final_combined)

        logits = self.lm_head(x_final_normed) 

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100) 
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return_dict = {'loss': loss, 'logits': logits}
        return return_dict

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval() 
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            outputs = self(idx_cond) 
            logits = outputs['logits'] 
            logits = logits[:, -1, :] / temperature 
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1) 
        self.train() 
        return idx
