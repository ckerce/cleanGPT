#!/usr/bin/env python
# ./train_modified_attention_transformer.py
"""
Example script demonstrating how to train a Vanilla Transformer model
with a MODIFIED CausalSelfAttention mechanism, using a GPT-2 tokenizer.
The modified attention uses the input embedding x directly as Value (V)
and combines head outputs with learnable weights.


Example calls:

# (1) Minimal implementation
python examples/train_dechannelized_vanilla_gpt2.py \
  --num_epochs 1 \
  --output_dir "output/test_dechannelize/"

# (2) A call example with 6 heads and 6 layers, n_embd=192, and num_epochs=3
python examples/train_dechannelized_vanilla_gpt2.py \
  --output_dir "output/dechannelized_out/"
  --n_layer 6 \
  --n_head 6 \
  --n_embd 192 \
  --num_epochs 3 \
  --max_samples 5000


python examples/train_dechannelized_vanilla_gpt2.py \
 --dataset "roneneldan/TinyStories"  \  
 --n_layer 6   --n_head 6   --n_embd 288   --block_size 256   \
 --batch_size 64   --num_epochs 5   -loutput_dir "./outputs/dechannelized_out" \   
 --max_samples 400000   

"""

import argparse
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import math
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer # Using AutoTokenizer for GPT-2

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration Class
# -----------------------------------------------------------------------------
class GPTConfig:
    """Configuration class to store the configuration of a `VanillaTransformerModel`."""
    def __init__(self, model_type="VanillaModifiedAttention", vocab_size=50257, block_size=256, 
                 n_layer=4, n_head=4, n_embd=128, dropout=0.1, bias=False,
                 batch_size=32, num_epochs=3, learning_rate=2.5e-4, weight_decay=1e-2,
                 padding_idx=None, **kwargs):
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias # Bias for Linear layers and LayerNorm
        self.padding_idx = padding_idx

        # Training specific parameters (can be part of a separate TrainerConfig)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Update with any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update_from_tokenizer(self, tokenizer):
        """Updates vocab_size and padding_idx from the tokenizer."""
        self.vocab_size = tokenizer.vocab_size
        if tokenizer.pad_token_id is not None:
            self.padding_idx = tokenizer.pad_token_id
        logger.info(f"Updated config from tokenizer: vocab_size={self.vocab_size}, padding_idx={self.padding_idx}")

def print_config(config, **kwargs):
    """Prints the configuration."""
    logger.info("Model Configuration:")
    for key, value in vars(config).items():
        logger.info(f"  {key}: {value}")
    if kwargs:
        logger.info("Additional Training Info:")
        for key, value in kwargs.items():
            logger.info(f"  {key}: {value}")


# -----------------------------------------------------------------------------
# Model Definition (Copied from transformer_model_updated_attention)
# -----------------------------------------------------------------------------
class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input_tensor):
        orig_dtype = input_tensor.dtype
        input_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        casted_input = input_tensor.to(input_dtype)
        casted_weight = self.weight.to(input_dtype)
        casted_bias = self.bias.to(input_dtype) if self.bias is not None else None
        normalized_output = F.layer_norm(
            casted_input, casted_weight.shape, casted_weight, casted_bias, 1e-5
        )
        return normalized_output.to(orig_dtype)

class CausalSelfAttention(nn.Module):
    """Modified multi-head self-attention with causal mask."""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_weights = nn.Parameter(torch.empty(self.n_head, config.n_embd))
        torch.nn.init.xavier_uniform_(self.head_weights)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()
        q_proj, k_proj = self.c_attn(x).split(self.n_embd, dim=2)
        hs = C // self.n_head
        k = k_proj.view(B, T, self.n_head, hs).transpose(1, 2)
        q = q_proj.view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        x_expanded = x.unsqueeze(1)
        attended_x_heads = torch.matmul(att, x_expanded)
        weights_reshaped = self.head_weights.unsqueeze(0).unsqueeze(2)
        weighted_attended_heads = attended_x_heads * weights_reshaped
        y = weighted_attended_heads.sum(dim=1)
        y = self.resid_dropout(y)
        return y

class MLP(nn.Module):
    """Simple MLP with GELU activation and dropout."""
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.activation = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
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
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class VanillaTransformerModel(nn.Module):
    """Transformer model with modified CausalSelfAttention."""
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.padding_idx = config.padding_idx

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=self.padding_idx),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([PreLNBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        logger.info(f"VanillaTransformerModel (Modified Attention) initialized with {self.get_num_params()/1e6:.2f}M parameters")

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

    def forward(self, input_ids, labels=None): # Removed attention_mask as it's not used by this causal model
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.config.block_size, f"Seq len {t} > block_size {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {'loss': loss, 'logits': logits}

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
# -----------------------------------------------------------------------------

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Transformer with Modified Attention")
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories", help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default=None, help="Dataset configuration")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max samples from dataset")
    parser.add_argument("--n_layer", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--block_size", type=int, default=256, help="Context window size")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--bias", action="store_true", default=False, help="Use bias")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--output_dir", type=str, default="./outputs/modified_attention_model", help="Output directory")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"], help="Device (auto-detect if None)")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="Tokenizer name (e.g., gpt2)")
    parser.add_argument("--use_fast_tokenizer", action="store_true", default=True, help="Use fast tokenizer")

    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # 1. Load Dataset
    logger.info(f"Loading dataset: {args.dataset}")
    try:
        dataset_args = {'path': args.dataset}
        if args.dataset_config:
            dataset_args['name'] = args.dataset_config
        dataset = load_dataset(**dataset_args, split=f"train[:{args.max_samples}]", trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    text_field = 'text' if 'text' in dataset.column_names else \
                 'story' if 'story' in dataset.column_names else \
                 next((col for col in dataset.column_names if dataset.features[col].dtype == 'string'), None)
    if not text_field:
        logger.error(f"Could not find a text column. Available: {dataset.column_names}")
        return
    logger.info(f"Using text column: '{text_field}'. Loaded {len(dataset)} samples.")

    # 2. Create Tokenizer
    logger.info(f"Creating {args.tokenizer_name} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=args.use_fast_tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Tokenizer pad_token set to eos_token ({tokenizer.eos_token}).")
    logger.info(f"Tokenizer initialized. Vocab size: {tokenizer.vocab_size}")

    # 3. Create Model Configuration
    model_config = GPTConfig(
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        block_size=args.block_size, dropout=args.dropout, bias=args.bias,
        batch_size=args.batch_size, num_epochs=args.num_epochs,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
    )
    model_config.update_from_tokenizer(tokenizer)
    print_config(model_config, dataset_name=args.dataset, max_samples=args.max_samples)

    # 4. Prepare Training Data
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    class CustomDataCollator:
        def __init__(self, tokenizer, block_size):
            self.tokenizer = tokenizer
            self.block_size = block_size
            self.pad_token_id = tokenizer.pad_token_id
            if self.pad_token_id is None:
                raise ValueError("Tokenizer must have pad_token_id.")

        def __call__(self, examples: list[dict]):
            batch_input_ids, batch_labels = [], []
            for ex in examples:
                ids = ex['input_ids']
                padding_length = self.block_size - len(ids)
                if padding_length < 0: ids = ids[:self.block_size]; padding_length = 0
                padded_ids = ids + [self.pad_token_id] * padding_length
                labels = padded_ids[1:] + [-100] # Shifted labels, -100 for padding
                batch_input_ids.append(padded_ids)
                batch_labels.append(labels)
            return {
                "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                "labels": torch.tensor(batch_labels, dtype=torch.long),
            }

    def tokenize_function(examples):
        return tokenizer(
            examples[text_field], add_special_tokens=True, truncation=True,
            max_length=model_config.block_size, padding=False, # Collator handles padding
        )

    logger.info(f"Tokenizing dataset (max_length={model_config.block_size})...")
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names, desc="Tokenizing"
    )
    data_collator = CustomDataCollator(tokenizer, model_config.block_size)
    dataloader = DataLoader(
        tokenized_dataset, batch_size=model_config.batch_size, collate_fn=data_collator, shuffle=True
    )
    logger.info(f"DataLoader created. Batches per epoch: {len(dataloader)}")

    # 5. Initialize Model
    logger.info("Initializing Vanilla Transformer Model (Modified Attention)...")
    model = VanillaTransformerModel(config=model_config)
    model = model.to(device)

    # 6. Setup Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=model_config.learning_rate, weight_decay=model_config.weight_decay
    )

    # 7. Training Loop
    logger.info("Starting training...")
    model.train()
    for epoch in range(model_config.num_epochs):
        logger.info(f"Epoch {epoch+1}/{model_config.num_epochs}")
        total_loss = 0
        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, labels=labels)
            loss = outputs['loss']
            
            if loss is not None:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if step % 100 == 0: # Log every 100 steps
                    logger.info(f"  Step {step}/{len(dataloader)}, Loss: {loss.item():.4f}")
            else:
                logger.warning(f"  Step {step}/{len(dataloader)}, Loss was None. Skipping batch.")

        avg_epoch_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")

    # 8. Save Model
    model_path = os.path.join(args.output_dir, "modified_attention_transformer.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Tokenizer is pre-trained, usually not saved unless fine-tuned.
    # tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))


    # 9. Generate Sample Text
    logger.info("Generating sample text with the trained model...")
    test_prompts = [
       "Once upon a time, in a land filled with sparkling rivers",
       "The little blue boat bobbed gently on the shimmering lake",
       "Deep in the whispering woods, lived a tiny, brave mouse named",
    ] 
    model.eval()
    for i, prompt_text in enumerate(test_prompts):
        logger.info(f"\nGenerating for prompt: '{prompt_text}'")
        input_ids = tokenizer.encode(prompt_text, add_special_tokens=True, return_tensors="pt").to(device)
        try:
            output_ids_tensor = model.generate(
                idx=input_ids, max_new_tokens=50, temperature=0.7, top_k=40
            )
            generated_text = tokenizer.decode(output_ids_tensor[0].tolist(), skip_special_tokens=True)
            logger.info(f"Generated text: {generated_text}")
            with open(os.path.join(args.output_dir, f"generation_{i+1}.txt"), "w", encoding="utf-8") as f:
                f.write(f"Prompt: {prompt_text}\n\nGenerated:\n{generated_text}\n")
        except Exception as e:
            logger.error(f"Error generating text for prompt '{prompt_text}': {e}", exc_info=True)

    logger.info("Training and generation example completed!")

if __name__ == "__main__":
    main()

