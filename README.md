# cleanGPT

A clean, modular implementation of transformer-based language models for research and experimentation.

## Project Overview

cleanGPT provides a flexible framework for training and experimenting with different transformer architectures, focusing on:

- **Modularity**: Easily swap tokenizers, model architectures, and training strategies
- **Clarity**: Clean, well-documented code that's easy to understand and modify
- **Extensibility**: Simple interfaces for adding new components

The project includes implementations of multiple transformer architectures, including a standard Vanilla Transformer and the Simplified Attention Sub-Block with Projections (SASP) variant.

## Project Structure

```
cleanGPT/
├── main.py                      # Main entry point
├── config.py                    # Configuration management
├── tokenizers/                  # Tokenizer implementations
│   ├── __init__.py              # Exports factory functions
│   ├── base_tokenizer.py        # Abstract base class 
│   ├── character_tokenizer.py   # Character-level tokenization
│   ├── gpt2_tokenizer.py        # GPT-2 tokenizer wrapper
│   └── factory.py               # Factory for creating tokenizers
├── utils/                       # Utility functions
│   ├── __init__.py              # Exports utilities
│   ├── data_utils.py            # Data loading and preparation
│   └── token_statistics.py      # Token usage analysis tools
├── model/                       # Model definitions
│   ├── __init__.py              # Factory for model selection
│   ├── model_SASPV.py           # SASP implementation
│   └── model_Vanilla.py         # Standard transformer
├── trainers/                    # Training implementations
│   ├── __init__.py              # Exports trainer factory
│   ├── base_trainer.py          # Abstract base trainer
│   └── simple_trainer.py        # Basic training loop
├── inference/                   # Generation utilities
│   ├── __init__.py              # Exports inference functions
│   ├── generation.py            # Text generation functions
│   └── sampling_strategies.py   # Different sampling approaches
└── examples/                    # Example scripts
    └── token_analysis.py        # Token usage analysis example
```

## Features

- **Multiple Tokenization Strategies**: Character-level, GPT-2, with consistent interface
- **Flexible Model Architectures**: 
  - Vanilla Transformer with Pre-LayerNorm
  - SASP Transformer (Simplified Attention Sub-Block with Projections)
- **Extensible Training**: Modular training loop with customizable strategies
- **Advanced Text Generation**: Multiple sampling methods (greedy, top-k, top-p, etc.)
- **Token Analysis Tools**: Analyze tokenizer performance and optimize vocabularies

## Getting Started

### Installation

```bash
git clone https://github.com/yourusername/cleanGPT.git
cd cleanGPT
pip install -r requirements.txt
```

### Training a Model

```bash
python main.py --model_type SASP --tokenizer_type gpt2 --n_layer 6 --n_head 6 --n_embd 384
```

### Analyzing Token Usage

```bash
python examples/token_analysis.py --dataset wikitext --tokenizer_type gpt2 --max_samples 1000 --plot
```

## Core Components

### Tokenizers

The `tokenizers` module provides a unified interface for different tokenization strategies:

```python
from tokenizers import create_tokenizer

# Create a GPT-2 tokenizer
tokenizer = create_tokenizer("gpt2")

# Tokenize text
encoded = tokenizer.encode("Hello, world!")
decoded = tokenizer.decode(encoded)
```

### Models

The `model` module contains different transformer architectures:

```python
from model import get_model
from config import GPTConfig

# Create model configuration
config = GPTConfig(
    model_type="SASP",
    n_layer=6,
    n_head=6,
    n_embd=384,
    vocab_size=50257
)

# Initialize model
model = get_model(config.model_type, config=config)
```

### Training

The `trainers` module handles model training:

```python
from trainers import get_trainer

trainer = get_trainer(
    trainer_type="simple",
    model=model,
    dataloader=dataloader,
    optimizer=optimizer,
    device=device,
    num_epochs=5
)

# Train the model
trainer.train()
```

### Inference

The `inference` module provides text generation capabilities:

```python
from inference.generation import run_generation

# Generate text from a prompt
generated_ids, generated_text = run_generation(
    model=model,
    tokenizer=tokenizer,
    prompt_text="Once upon a time",
    device=device,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50
)
```

## Extending the Framework

### Adding a New Tokenizer

1. Create a new class in `tokenizers/` that inherits from `BaseTokenizer`
2. Implement all required methods
3. Register it in `TokenizerFactory.TOKENIZER_TYPES`

### Adding a New Model Architecture

1. Create a new model class in `model/`
2. Register it in `MODEL_REGISTRY` in `model/__init__.py`

### Adding a New Training Strategy

1. Create a new trainer class in `trainers/` that inherits from `BaseTrainer` 
2. Register it in `TRAINER_REGISTRY` in `trainers/__init__.py`

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.20.0+
- Datasets 2.0.0+
- tqdm

## Citation

If you use this codebase in your research, please cite:

```
@software{cleangpt2023,
  author = {Clayton Kerce},
  title = {cleanGPT: Transformer Architecture Sandbox for Exploration of Language Models Features},
  year = {2023},
  url = {https://github.com/ckerce/cleanGPT}
}
```


