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
./
├── __init__.py
├── config.py
├── doc/
│   └── README_cleanGPT_Tutorial.md
├── examples/
│   ├── ignore/
│   │   └── ...
│   ├── train_sasp_char.py
│   └── train_{model}_{tokenizer}.py
├── ignore/
│   └── ...
├── inference/
│   ├── __init__.py
│   ├── generation.py
│   └── sampling_strategies.py
├── main.py
├── model/
│   ├── __init__.py
│   ├── model_SASPV.py
│   ├── model_token_factored.py
│   └── model_Vanilla.py
├── mytokenizers/
│   ├── __init__.py
│   ├── base_tokenizer.py
│   ├── character_tokenizer.py
│   ├── factory.py
│   └── gpt2_tokenizer.py
├── output/
│   └── info.md
├── requirements.txt
├── setup.py
├── tests/
│   ├── ...
├── trainers/
│   ├── __init__.py
│   ├── base_trainer.py
│   ├── simple_trainer.py
│   └── train_with_callbacks.py
└── utils/
    ├── __init__.py
    ├── data_utils.py
    ├── simple_quantitative_evaluation.py
    └── token_statistics.py
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


