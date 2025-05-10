# cleanGPT Tutorial

This tutorial provides a comprehensive overview of the cleanGPT framework, explaining its components, architecture, and how to extend it for your own research and experimentation.

## Table of Contents

1. [Overview and Philosophy](#overview-and-philosophy)
2. [Core Components](#core-components)
   - [Tokenizers](#tokenizers)
   - [Models](#models)
   - [Trainers](#trainers)
   - [Inference](#inference)
   - [Utilities](#utilities)
3. [Creating Custom Tokenizers](#creating-custom-tokenizers)
4. [Building Modified Transformer Architectures](#building-modified-transformer-architectures)
5. [Implementing Custom Training Strategies](#implementing-custom-training-strategies)
6. [Advanced Text Generation](#advanced-text-generation)
7. [Token Analysis and Vocabulary Optimization](#token-analysis-and-vocabulary-optimization)
8. [End-to-End Examples](#end-to-end-examples)

## Overview and Philosophy

cleanGPT is designed around the principle of separation of concerns, enabling researchers to focus on experimenting with specific aspects of the language model pipeline without needing to reimplement the entire system. The architecture follows a modular design where components have well-defined interfaces and can be swapped or extended easily.

Key design philosophies:

- **Modularity**: Components are interchangeable through consistent interfaces
- **Readability**: Code prioritizes clarity and maintainability
- **Extensibility**: Easy to add new variants without altering existing code
- **Configurability**: Extensive configuration options without code changes
- **Single Responsibility**: Each component handles one aspect of the system

## Core Components

### Tokenizers

The tokenizer system converts text to and from token IDs for processing by the model.

**Key Files**:
- `tokenizers/base_tokenizer.py`: Abstract base class defining the tokenizer interface
- `tokenizers/character_tokenizer.py`: Simple character-level tokenizer implementation
- `tokenizers/gpt2_tokenizer.py`: Wrapper for Hugging Face's GPT-2 tokenizer
- `tokenizers/factory.py`: Factory pattern for creating tokenizers

**Usage Example**:

```python
from tokenizers import create_tokenizer

# Create a character-level tokenizer
char_tokenizer = create_tokenizer("character")

# Create a GPT-2 tokenizer
gpt2_tokenizer = create_tokenizer("gpt2")

# Encode and decode text
text = "Hello, world!"
tokens = gpt2_tokenizer.encode(text)
decoded_text = gpt2_tokenizer.decode(tokens)

# Use for model input (returns tensors ready for the model)
inputs = gpt2_tokenizer(
    ["First sentence", "Second sentence"], 
    padding=True, 
    truncation=True,
    return_tensors="pt"
)
```

### Models

The model system provides implementations of different transformer architectures.

**Key Files**:
- `model/__init__.py`: Contains the model registry and factory function
- `model/model_Vanilla.py`: Standard transformer implementation
- `model/model_SASPV.py`: Simplified Attention Sub-Block with Projections (SASP) implementation

**Usage Example**:

```python
from model import get_model
from config import GPTConfig

# Create model configuration
config = GPTConfig(
    model_type="SASP",    # Or "Vanilla"
    vocab_size=50257,     # Size of tokenizer vocabulary
    n_layer=6,            # Number of transformer layers
    n_head=6,             # Number of attention heads
    n_embd=384,           # Embedding dimension
    block_size=128        # Maximum sequence length
)

# Get model instance
model = get_model(config.model_type, config=config)

# Forward pass (assuming properly formatted inputs)
outputs = model(input_ids=input_ids, labels=labels)

# Access loss and logits
loss = outputs['loss']
logits = outputs['logits']
```

### Trainers

The trainer system handles the training loop and optimization.

**Key Files**:
- `trainers/base_trainer.py`: Abstract base class defining the trainer interface
- `trainers/simple_trainer.py`: Basic training loop implementation
- `trainers/__init__.py`: Trainer registry and factory function

**Usage Example**:

```python
import torch
from trainers import get_trainer

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Create trainer
trainer = get_trainer(
    trainer_type="simple",
    model=model,
    dataloader=train_dataloader,
    optimizer=optimizer,
    device=torch.device("cuda"),
    num_epochs=5
)

# Train the model
training_stats = trainer.train()

# Evaluate on validation data
eval_stats = trainer.evaluate(eval_dataloader)
```

### Inference

The inference system handles text generation with different sampling strategies.

**Key Files**:
- `inference/generation.py`: Main generation functions
- `inference/sampling_strategies.py`: Different token sampling methods
- `inference/__init__.py`: Exports key functions

**Usage Example**:

```python
from inference.generation import run_generation, batch_generate

# Generate from a single prompt
generated_ids, generated_text = run_generation(
    model=model,
    tokenizer=tokenizer,
    prompt_text="Once upon a time",
    device=torch.device("cuda"),
    max_new_tokens=50,
    temperature=0.8,
    top_k=50
)

# Generate from multiple prompts
prompts = [
    "The history of",
    "Scientists discovered that",
    "According to research"
]

results = batch_generate(
    model=model,
    tokenizer=tokenizer,
    prompts=prompts,
    device=torch.device("cuda"),
    max_new_tokens=30,
    temperature=0.7
)
```

### Utilities

Utility functions for data loading, token analysis, and more.

**Key Files**:
- `utils/data_utils.py`: Functions for loading and preparing data
- `utils/token_statistics.py`: Tools for analyzing token usage

**Usage Example**:

```python
from utils import load_and_prepare_data
from utils.token_statistics import TokenUsageAnalyzer

# Load dataset
dataloader, tokenizer = load_and_prepare_data(
    dataset_name="wikitext",
    dataset_config="wikitext-103-v1",
    tokenizer=tokenizer,
    max_samples=1000,
    max_seq_length=128,
    batch_size=32
)

# Analyze token usage
analyzer = TokenUsageAnalyzer(tokenizer)
results = analyzer.analyze_texts(text_samples, max_samples=1000)

# Generate plots
analyzer.plot_token_distribution(top_n=100)
analyzer.plot_coverage_curve()

# Create reduced vocabulary
reduced_vocab = analyzer.create_reduced_vocab(coverage=0.95)
```

## Creating Custom Tokenizers

To create a custom tokenizer, follow these steps:

1. **Create a new tokenizer class** that inherits from `BaseTokenizer`:

```python
# ./tokenizers/byte_level_tokenizer.py
from .base_tokenizer import BaseTokenizer

class ByteLevelTokenizer(BaseTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize tokenizer-specific attributes
        
    def tokenize(self, text):
        # Implement tokenization logic
        return [byte for byte in text.encode('utf-8')]
        
    def encode(self, text, **kwargs):
        # Implement encoding logic
        tokens = self.tokenize(text)
        return self.convert_tokens_to_ids(tokens)
        
    def decode(self, token_ids, **kwargs):
        # Implement decoding logic
        tokens = self.convert_ids_to_tokens(token_ids)
        return bytes(tokens).decode('utf-8', errors='replace')
        
    def convert_tokens_to_ids(self, tokens):
        # Implement token to ID conversion
        return tokens  # For byte tokenizer, bytes are already 0-255
        
    def convert_ids_to_tokens(self, ids):
        # Implement ID to token conversion
        return ids  # For byte tokenizer, IDs are already 0-255 bytes
```

2. **Register the tokenizer** in the factory:

```python
# ./tokenizers/factory.py
from .byte_level_tokenizer import ByteLevelTokenizer

class TokenizerFactory:
    TOKENIZER_TYPES = {
        'character': CharacterTokenizer,
        'gpt2': GPT2Tokenizer,
        'byte': ByteLevelTokenizer,  # Add your new tokenizer
    }
    # Rest of the factory class
```

3. **Import in the module's `__init__.py`**:

```python
# ./tokenizers/__init__.py
from .byte_level_tokenizer import ByteLevelTokenizer

__all__ = [
    'BaseTokenizer',
    'CharacterTokenizer', 
    'GPT2Tokenizer',
    'ByteLevelTokenizer',  # Add your new tokenizer
    'TokenizerFactory',
    'create_tokenizer',
    'from_pretrained'
]
```

## Building Modified Transformer Architectures

To create a new transformer architecture:

1. **Create a new model file** in the `model/` directory:

```python
# ./model/model_MyTransformer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Implement your custom attention mechanism
        
    def forward(self, x):
        # Implement forward pass
        return x

class MyTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MyAttentionBlock(config)
        # Other layers
        
    def forward(self, x):
        # Implement forward pass with your custom attention
        return x

class MyTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Implement model architecture with your blocks
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([MyTransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Implement the forward pass
        # Return dict with 'loss' and 'logits' keys
        
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        # Implement the generation method
        return generated_sequence
```

2. **Register your model** in the model registry:

```python
# ./model/__init__.py
from .model_MyTransformer import MyTransformerModel

MODEL_REGISTRY = {
    "SASP": SASPTransformerModel,
    "Vanilla": VanillaTransformerModel,
    "MyTransformer": MyTransformerModel,  # Add your model
}
```

3. **Update configuration parameters** if needed:

```python
# ./config.py
@dataclass
class GPTConfig:
    # Existing parameters
    
    # Your architecture-specific parameters
    my_parameter: float = 0.5
    use_custom_feature: bool = False
```

Key areas to experiment with in transformer architectures:

- **Attention mechanisms**: Modify how attention weights are calculated
- **Feed-forward networks**: Change activation functions or network structure
- **Normalization**: Pre-norm vs. post-norm, different normalization methods
- **Residual connections**: Change how information flows between blocks
- **Positional encoding**: Alternative position representations

## Implementing Custom Training Strategies

To create a custom training strategy:

1. **Create a new trainer class** that inherits from `BaseTrainer`:

```python
# ./trainers/advanced_trainer.py
import time
import torch
from tqdm.auto import tqdm
from .base_trainer import BaseTrainer

class AdvancedTrainer(BaseTrainer):
    def __init__(self, 
                 model, 
                 dataloader, 
                 optimizer, 
                 device,
                 num_epochs=5,
                 output_dir=None,
                 lr_scheduler=None,
                 gradient_accumulation_steps=1,
                 mixed_precision=False):
        super().__init__(model, dataloader, optimizer, device, output_dir)
        self.num_epochs = num_epochs
        self.lr_scheduler = lr_scheduler
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        
    def train(self):
        # Implement advanced training loop with features like:
        # - Gradient accumulation
        # - Mixed precision training
        # - Learning rate scheduling
        # - Early stopping
        # Return training statistics
        
    def evaluate(self, eval_dataloader=None):
        # Implement evaluation
        # Return evaluation metrics
```

2. **Register your trainer** in the trainer registry:

```python
# ./trainers/__init__.py
from .advanced_trainer import AdvancedTrainer

TRAINER_REGISTRY = {
    'simple': SimpleTrainer,
    'advanced': AdvancedTrainer,  # Add your trainer
}
```

Advanced training features to consider:

- **Gradient accumulation**: Update weights after multiple forward/backward passes
- **Mixed precision training**: Use lower precision for speed with minimal accuracy loss
- **Learning rate scheduling**: Adjust learning rate dynamically during training
- **Early stopping**: Stop training when validation metrics plateau
- **Checkpoint averaging**: Average weights from multiple checkpoints
- **Gradient clipping**: Prevent exploding gradients
- **Distributed training**: Train across multiple GPUs or machines

## Advanced Text Generation

cleanGPT provides several sampling strategies for text generation. Here's how to use them:

1. **Basic generation** with top-k sampling:

```python
from inference.generation import run_generation

generated_ids, generated_text = run_generation(
    model=model,
    tokenizer=tokenizer,
    prompt_text="The future of AI is",
    device=device,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50
)
```

2. **Using different sampling strategies**:

```python
from inference.sampling_strategies import get_sampling_fn

# Get specific sampling function
top_p_sampling = get_sampling_fn("top_p")

# Use it in a custom generation loop
input_ids = tokenizer.encode("The future of AI is", return_tensors="pt").to(device)
for _ in range(50):  # Generate 50 new tokens
    with torch.no_grad():
        outputs = model(input_ids)
    
    # Apply top-p sampling to logits
    next_token = top_p_sampling(
        outputs['logits'][:, -1, :],  # Only consider last token position
        p=0.9,                        # 90% cumulative probability threshold
        temperature=0.8               # Temperature for controlled randomness
    )
    
    # Append token and continue
    input_ids = torch.cat([input_ids, next_token], dim=1)

# Decode the result
generated_text = tokenizer.decode(input_ids[0].tolist())
```

3. **Combined sampling strategies**:

```python
from inference.sampling_strategies import combined_sampling

next_token = combined_sampling(
    logits,
    temperature=0.8,
    top_k=50,
    top_p=0.9
)
```

Experiment with these parameters to control generation quality:

- **Temperature**: Higher values (>1.0) produce more random/creative outputs, while lower values (<1.0) produce more focused/deterministic outputs
- **Top-k**: Limits sampling to k most likely tokens; smaller values are more focused, larger values allow more diversity
- **Top-p/nucleus**: Only considers tokens that exceed cumulative probability threshold; adapts dynamically to the token distribution

## Token Analysis and Vocabulary Optimization

cleanGPT includes tools for analyzing tokenizer performance and optimizing vocabularies:

1. **Basic token analysis**:

```python
from utils.token_statistics import TokenUsageAnalyzer

# Create analyzer with your tokenizer
analyzer = TokenUsageAnalyzer(tokenizer)

# Analyze a dataset
results = analyzer.analyze_texts(text_samples)

# Print statistics
print(f"Total tokens: {results['total_tokens']}")
print(f"Unique tokens: {results['unique_tokens']}")
print(f"Vocabulary coverage: {results['vocab_coverage']:.2%}")
print(f"Average sequence length: {results['avg_sequence_length']:.2f}")

# View most common tokens
for token, count in results['most_common_tokens'][:10]:
    print(f"{repr(token)}: {count}")
```

2. **Token distribution visualization**:

```python
# Plot token frequency distribution
analyzer.plot_token_distribution(top_n=100)

# Plot vocabulary coverage curve
analyzer.plot_coverage_curve()
```

3. **Create reduced vocabulary** for more efficient training:

```python
# Create a vocabulary that covers 95% of the tokens
reduced_vocab = analyzer.create_reduced_vocab(coverage=0.95)

# Save it for later use
import json
with open("reduced_vocab.json", "w") as f:
    json.dump(reduced_vocab, f)
```

4. **Create a specialized tokenizer** with the reduced vocabulary:

```python
from tokenizers import CharacterTokenizer

# Create tokenizer with reduced vocabulary
specialized_tokenizer = CharacterTokenizer(vocab=reduced_vocab)

# Save for future use
specialized_tokenizer.save_pretrained("./specialized_tokenizer")
```

Vocabulary optimization can significantly improve training efficiency and model performance, especially for domain-specific tasks.

## End-to-End Examples

### Example 1: Train a SASP Transformer with Character Tokenizer

```python
import torch
from tokenizers import create_tokenizer
from utils import load_and_prepare_data
from model import get_model
from trainers import get_trainer
from config import GPTConfig

# 1. Create tokenizer
tokenizer = create_tokenizer("character")

# 2. Prepare dataset (example using a list of texts)
texts = ["Example text 1", "Example text 2", ...]
from utils import prepare_causal_lm_dataset
dataloader = prepare_causal_lm_dataset(
    texts=texts,
    tokenizer=tokenizer,
    block_size=128,
    batch_size=32
)

# 3. Create model configuration
config = GPTConfig(
    model_type="SASP",
    vocab_size=tokenizer.vocab_size,
    n_layer=4,
    n_head=4,
    n_embd=256,
    block_size=128
)

# 4. Initialize model
model = get_model(config.model_type, config=config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 5. Setup optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# 6. Create trainer and train
trainer = get_trainer(
    trainer_type="simple",
    model=model,
    dataloader=dataloader,
    optimizer=optimizer,
    device=device,
    num_epochs=3
)
trainer.train()

# 7. Generate text with the trained model
from inference.generation import run_generation
generated_ids, generated_text = run_generation(
    model=model,
    tokenizer=tokenizer,
    prompt_text="Example",
    device=device,
    max_new_tokens=20,
    temperature=0.7
)
print(generated_text)
```

### Example A2: Custom Transformer Experiment

```python
import torch
from tokenizers import create_tokenizer
from datasets import load_dataset
from utils import load_and_prepare_data
from model import get_model
from trainers import get_trainer
from config import GPTConfig

# 1. Load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = dataset["text"]

# 2. Create and analyze tokenizer
tokenizer = create_tokenizer("gpt2")
from utils.token_statistics import TokenUsageAnalyzer
analyzer = TokenUsageAnalyzer(tokenizer)
analyzer.analyze_texts(texts, max_samples=1000)
analyzer.plot_coverage_curve()  # Visualize vocabulary coverage

# 3. Prepare data
dataloader, _ = load_and_prepare_data(
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    tokenizer=tokenizer,
    max_samples=10000,
    max_seq_length=128,
    batch_size=16
)

# 4. Configure custom model
config = GPTConfig(
    model_type="MyTransformer",  # Your custom model
    vocab_size=tokenizer.vocab_size,
    n_layer=6,
    n_head=8,
    n_embd=512,
    block_size=128,
    # Custom parameters
    my_parameter=0.5,
    use_custom_feature=True
)

# 5. Initialize model
model = get_model(config.model_type, config=config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 6. Advanced training setup
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)

from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=1000)

trainer = get_trainer(
    trainer_type="advanced",  # Custom advanced trainer
    model=model,
    dataloader=dataloader,
    optimizer=optimizer,
    device=device,
    num_epochs=5,
    lr_scheduler=scheduler,
    gradient_accumulation_steps=4,
    mixed_precision=True
)

# 7. Train and save model
training_stats = trainer.train()

# 8. Evaluate performance
eval_dataloader, _ = load_and_prepare_data(
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    tokenizer=tokenizer,
    max_samples=1000,
    max_seq_length=128,
    batch_size=16,
    split="validation"
)
eval_stats = trainer.evaluate(eval_dataloader)
print(f"Perplexity: {eval_stats['perplexity']:.2f}")

# 9. Generate text with advanced sampling
from inference.generation import run_generation
generated_text = run_generation(
    model=model,
    tokenizer=tokenizer,
    prompt_text="In recent years, scientists have discovered",
    device=device,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.92  # Using nucleus sampling
)[1]
print(generated_text)
```

These examples demonstrate the flexibility and power of the cleanGPT framework for experimenting with different transformer architectures, tokenization strategies, and training methods.
