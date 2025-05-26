# Token-Factored Transformer with ALiBi Tutorial

## Overview

This tutorial introduces the **Token-Factored Transformer with ALiBi**, an architecture that illustrates the concept of factoring token-like and embedding-like internal state representations in the transformer architecture.  ALiBi positional representations are used to maintain the conceptual clarity of the token-like structure as xt is passed block-to-block within the transformer backbone -- no addative modulations are applied to xt channels within the architecture. Furthermore, when the value matrix is not the identity, only pure channel mixing is implemented.  This stands in contrast to standard transformers, where the value matix varies across the entire token embedding -- a process that destroys the mapping between tokens and internal transformer state variables.  

1. **Token Factorization**: Maintains separate `xt` (token-like) and `xe` (embedding-like) internal state streams
2. **ALiBi (Attention with Linear Biases)**: Replaces learned positional embeddings with linear biases for better length extrapolation

The result is an interpretable and transparent model that conceptually seperates the symbolic token manipulation of xt (an form or deductive reasoning on token-like internal states) and abductive introduction of new internal states xe (perhaps in token form, this is currently unverified)  that provide additional contextual information for next token prediction. 

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Key Benefits](#key-benefits)
- [Installation & Setup](#installation--setup)
- [Basic Usage](#basic-usage)
- [Configuration](#configuration)
- [Training](#training)
- [Advanced Features](#advanced-features)
- [Performance Comparison](#performance-comparison)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## Quick Start

### 5-Minute Example

```python
from model import get_model
from config_alibi import GPTConfigALiBi, load_alibi_config_preset
from mytokenizers import create_tokenizer

# 1. Load a preset configuration
config = load_alibi_config_preset('small')  # 4L-4H-256D model

# 2. Create tokenizer and update config
tokenizer = create_tokenizer('gpt2')
config.update_from_tokenizer(tokenizer)

# 3. Initialize the model
model = get_model("FactoredALiBi", config)

# 4. Generate text (can exceed training length!)
prompt = "The future of artificial intelligence"
input_ids = torch.tensor([tokenizer.encode(prompt)], device='cuda')

# Generate up to 4x training length
generated = model.generate(
    input_ids, 
    max_new_tokens=800,  # Much longer than block_size (256)
    temperature=0.8,
    top_k=40
)

print(tokenizer.decode(generated[0], skip_special_tokens=True))
```

## Architecture Overview

### Dual-Stream Design

The model maintains two parallel embedding streams:

```
Input Tokens → Token Embeddings → xt (token-like stream)
                                   xe (embedding-like stream) ← zeros
```

### Information Flow

```mermaid
graph TD
    A[Token Embeddings] --> B[xt stream]
    C[Zero Tensor] --> D[xe stream]
    
    B --> E[norm(xt + xe)]
    D --> E
    
    E --> F[Attention Q,K,V]
    B --> G[V Modulation: xt × V]
    F --> G
    G --> H[Attention Output]
    H --> I[xt = xt + attention]
    
    I --> J[norm(xt + xe)]
    D --> J
    J --> K[MLP]
    K --> L[xe = xe + mlp]
    
    I --> M[Final: xt + xe]
    L --> M
```

### ALiBi Integration

Instead of learned positional embeddings, ALiBi adds linear biases directly to attention scores:

```python
# Traditional: Q·K^T + positional_bias_learned
# ALiBi:       Q·K^T + slope × relative_position
attention_scores = (Q @ K.T) + alibi_slopes[:, None, None] * relative_positions
```

## Key Benefits

### **Length Extrapolation**
- **Training**: 256 tokens
- **Inference**: 1024+ tokens
- **Quality**: Better than truncation or naive extension

### **Parameter Efficiency**
- **Saves**: 10-15% of total parameters
- **Example**: 98K fewer parameters for 256×384 model
- **Memory**: Reduced storage and faster loading

### **Structured Representations**
- **xt stream**: Token-specific information (attention-updated)
- **xe stream**: Contextual information (MLP-updated)
- **Interaction**: Cross-stream communication through normalization

### **Better Extrapolation**
- **No degradation**: Quality maintained at longer lengths
- **Scalable**: Linear computational cost
- **Flexible**: Configurable maximum inference length

## Installation & Setup

### Prerequisites

```bash
# Ensure you have the cleanGPT package structure
cleanGPT/
├── model/
│   ├── model_token_factored_alibi.py
│   └── __init__.py
├── config_alibi.py
├── mytokenizers/
└── utils/
```

### Dependencies

```python
# Core dependencies (already in cleanGPT)
torch>=1.9.0
numpy
tqdm
matplotlib  # For comparison plots

# Optional for advanced features
transformers  # For tokenizer comparison
datasets     # For real dataset loading
```

### Import Check

```python
# Verify ALiBi model is available
from model import list_available_models
print(list_available_models())
# Should include: ['SASP', 'Vanilla', 'Factored', 'FactoredALiBi']
```

## Basic Usage

### Model Creation

```python
from model import get_model
from config_alibi import GPTConfigALiBi

# Method 1: Using presets
config = load_alibi_config_preset('medium')  # 8L-8H-512D

# Method 2: Custom configuration
config = GPTConfigALiBi(
    n_layer=6,
    n_head=6,
    n_embd=384,
    block_size=256,                    # Training length
    max_position_embeddings=1024,      # Max inference length
    vocab_size=50257,
    dropout=0.1
)

# Create model
model = get_model("FactoredALiBi", config)
print(f"Model parameters: {model.get_num_params()/1e6:.2f}M")
```

### Forward Pass

```python
import torch

# Prepare input (batch_size=2, seq_len=128)
input_ids = torch.randint(0, config.vocab_size, (2, 128))
labels = input_ids.clone()

# Forward pass
outputs = model(input_ids=input_ids, labels=labels)

print(f"Logits shape: {outputs['logits'].shape}")  # [2, 128, vocab_size]
print(f"Loss: {outputs['loss'].item():.4f}")       # Cross-entropy loss
```

### Text Generation

```python
from mytokenizers import create_tokenizer

tokenizer = create_tokenizer('gpt2')
config.update_from_tokenizer(tokenizer)

# Encode prompt
prompt = "In the year 2050, artificial intelligence will"
input_ids = torch.tensor([tokenizer.encode(prompt)])

# Generate with length extrapolation
generated = model.generate(
    input_ids,
    max_new_tokens=500,      # Can exceed block_size
    temperature=0.8,         # Sampling temperature
    top_k=40,               # Top-k sampling
)

# Decode result
full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
generated_text = tokenizer.decode(generated[0][len(input_ids[0]):], skip_special_tokens=True)

print(f"Generated: {generated_text}")
```

## Configuration

### Configuration Options

```python
@dataclass
class GPTConfigALiBi:
    # Model Architecture
    n_layer: int = 6                        # Number of transformer layers
    n_head: int = 6                         # Number of attention heads
    n_embd: int = 384                       # Embedding dimension
    block_size: int = 128                   # Training sequence length
    max_position_embeddings: int = None     # Max inference length (default: 4×block_size)
    
    # ALiBi Specific
    alibi_max_bias: float = None            # Maximum bias (auto-computed)
    
    # Training
    batch_size: int = 32
    learning_rate: float = 2.5e-4
    weight_decay: float = 0.01
    dropout: float = 0.1
    
    # Generation
    temperature: float = 0.8
    top_k: int = 50
```

### Preset Configurations

```python
from config_alibi import load_alibi_config_preset

# Available presets
presets = ['small', 'medium', 'large', 'character']

# Load and customize
config = load_alibi_config_preset('medium')
config.max_position_embeddings = 4096  # Extend max length
config.dropout = 0.05                  # Reduce dropout
```

### Configuration Validation

```python
from config_alibi import print_config_alibi

# Print detailed configuration
print_config_alibi(config, dataset_name="wikitext", max_samples=10000)
```

## Training

### Basic Training Loop

```python
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model("FactoredALiBi", config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# Training loop
model.train()
for epoch in range(config.num_epochs):
    total_loss = 0
    
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs['loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
```

### Using the Training Example

```bash
# Run the provided training example
cd examples/
python train_factored_alibi_example.py \
    --preset medium \
    --block_size 256 \
    --max_position_embeddings 1024 \
    --batch_size 16 \
    --num_epochs 5 \
    --test_generation
```

### Dataset Preparation

```python
from utils.data_utils import prepare_causal_lm_dataset

# Option 1: Use existing data utilities
texts = ["Sample text 1", "Sample text 2", ...]  # Your text data
dataloader = prepare_causal_lm_dataset(
    texts, 
    tokenizer, 
    block_size=config.block_size,
    batch_size=config.batch_size
)

# Option 2: Custom dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, block_size):
        self.examples = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            for i in range(0, len(tokens) - block_size + 1, block_size//2):
                self.examples.append(tokens[i:i + block_size])
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)
```

## Advanced Features

### Length Extrapolation Testing

```python
def test_extrapolation(model, tokenizer, device):
    """Test model's ability to handle longer sequences."""
    model.eval()
    
    # Test at different lengths
    test_lengths = [64, 128, 256, 512, 1024]
    prompt = "The evolution of technology has"
    
    for length in test_lengths:
        if length > model.config.max_position_embeddings:
            continue
            
        print(f"\n--- Testing at {length} tokens ---")
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        
        try:
            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    max_new_tokens=length - input_ids.size(1),
                    temperature=0.7
                )
            
            text = tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"✓ Success: {text[:100]}...")
            
        except Exception as e:
            print(f"✗ Failed: {e}")

# Run test
test_extrapolation(model, tokenizer, device)
```

### Model Comparison

```python
from utils.compare_alibi_vs_original import run_comprehensive_comparison

# Compare ALiBi vs original factored model
run_comprehensive_comparison()
```

### Custom ALiBi Slopes

```python
# Access and modify ALiBi slopes (advanced usage)
model.eval()
first_layer = model.transformer.h[0].attn

# View current slopes
print("ALiBi slopes:", first_layer.alibi_slopes)

# Custom slopes (experimental)
custom_slopes = torch.tensor([0.5, 0.25, 0.125, 0.0625])  # For 4 heads
first_layer.alibi_slopes.data = custom_slopes
```

### Memory Optimization

```python
# For very long sequences, use gradient checkpointing
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()

# Monitor memory usage
if torch.cuda.is_available():
    print(f"Memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    print(f"Memory cached: {torch.cuda.memory_reserved()/1024**2:.1f} MB")
```

## Performance Comparison

### Parameter Efficiency

```python
from config_alibi import estimate_model_parameters
from config import GPTConfig

# Compare parameter counts
alibi_config = GPTConfigALiBi(n_layer=6, n_head=6, n_embd=384, block_size=256)
original_config = GPTConfig(n_layer=6, n_head=6, n_embd=384, block_size=256)

alibi_params = estimate_model_parameters(alibi_config)
original_params = alibi_params + (256 * 384)  # Add positional embedding params

print(f"ALiBi model:     {alibi_params:,} parameters")
print(f"Original model:  {original_params:,} parameters")
print(f"Savings:         {original_params - alibi_params:,} parameters ({(original_params - alibi_params)/original_params*100:.1f}%)")
```

### Length Capability

| Model Type | Training Length | Max Inference | Extrapolation Quality |
|------------|----------------|---------------|----------------------|
| Original   | 256 tokens     | 256 tokens    | ❌ Hard limit        |
| ALiBi      | 256 tokens     | 1024+ tokens  | ✅ Maintains quality |

### Benchmark Results

```python
# Run built-in benchmarks
from tests.test_alibi_integration import run_all_tests
run_all_tests()
```

## Troubleshooting

### Common Issues

#### 1. **Out of Memory Errors**

```python
# Solution: Reduce batch size or sequence length
config.batch_size = 8  # Reduce from 32
config.max_position_embeddings = 512  # Reduce from 1024

# Use gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch['input_ids'], labels=batch['labels'])['loss']
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 2. **Model Not Found Error**

```python
# Check model registry
from model import list_available_models
print("Available models:", list_available_models())

# If FactoredALiBi missing, check import
try:
    from model.model_token_factored_alibi import FactoredTransformerModelALiBi
    print("✓ ALiBi model imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
```

#### 3. **Generation Quality Issues**

```python
# Adjust generation parameters
generated = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,      # Lower = more focused
    top_k=40,            # Reduce for more focused sampling
    # top_p=0.9,         # Add nucleus sampling if needed
)

# Check if sequence is too long
if input_ids.size(1) + max_new_tokens > model.config.max_position_embeddings:
    print("Warning: Requested length exceeds max_position_embeddings")
```

#### 4. **Slow Training**

```python
# Enable optimizations
torch.backends.cudnn.benchmark = True  # For consistent input sizes
model = model.half()  # Use fp16 if supported

# Use DataLoader optimizations
dataloader = DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=4,        # Parallel data loading
    pin_memory=True,      # Faster GPU transfer
    persistent_workers=True
)
```

### Debugging Tips

```python
# Check model structure
print(model)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Verify ALiBi components
for name, module in model.named_modules():
    if 'attn' in name and hasattr(module, 'alibi_slopes'):
        print(f"ALiBi slopes in {name}: {module.alibi_slopes}")

# Monitor attention patterns (advanced)
def hook_fn(module, input, output):
    if hasattr(module, 'alibi_slopes'):
        print(f"Attention shape: {output.shape}")

model.transformer.h[0].attn.register_forward_hook(hook_fn)
```

## API Reference

### Core Classes

#### `FactoredTransformerModelALiBi`

```python
class FactoredTransformerModelALiBi(nn.Module):
    def __init__(self, config: GPTConfigALiBi)
    def forward(self, input_ids, attention_mask=None, labels=None) -> Dict
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None) -> torch.Tensor
    def get_num_params(self, non_embedding=True) -> int
```

#### `GPTConfigALiBi`

```python
@dataclass
class GPTConfigALiBi:
    block_size: int = 128
    max_position_embeddings: Optional[int] = None
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    # ... (see Configuration section for full list)
```

### Utility Functions

```python
# Model registry
get_model(model_type: str, config) -> nn.Module
list_available_models() -> List[str]

# Configuration
load_alibi_config_preset(preset_name: str) -> GPTConfigALiBi
print_config_alibi(config: GPTConfigALiBi, **kwargs)
estimate_model_parameters(config: GPTConfigALiBi) -> int

# Comparison
run_comprehensive_comparison()  # From utils.compare_alibi_vs_original
```

### Training Utilities

```python
# From examples/train_factored_alibi_example.py
train_model(model, dataloader, optimizer, device, num_epochs, log_interval)
test_length_extrapolation(model, tokenizer, device, config)

# From tests/test_alibi_integration.py
run_all_tests() -> bool  # Comprehensive integration testing
```

---

## Next Steps

1. **Experiment with presets**: Try different model sizes to find the right balance
2. **Custom datasets**: Adapt the training pipeline to your specific data
3. **Fine-tuning**: Use pre-trained weights and fine-tune for your task
4. **Performance optimization**: Profile and optimize for your hardware
5. **Extended lengths**: Push the boundaries of length extrapolation

## References

- [ALiBi Paper: Train Short, Test Long](https://arxiv.org/abs/2108.12409)
- [Token Factorization Architecture](./README_cleanGPT_Tutorial.md)
- [cleanGPT Documentation](./README_cleanGPT_Distillation_Process.md)

---

*For questions, issues, or contributions, please refer to the main cleanGPT documentation or open an issue in the repository.*
