# cleanGPT

A clean, modular implementation of transformer-based language models for research and experimentation. This supersedes https://github.com/ckerce/Transformer-Sandbox, but still reflects some of the spirit of https://github.com/karpathy/nanoGPT and similar repositories.

## Project Overview

cleanGPT provides a flexible framework for training, distilling, and experimenting with different transformer architectures, focusing on:

- **Modularity**: Easily swap tokenizers, model architectures, and training strategies
- **Clarity**: Clean, well-documented code that's easy to understand and modify
- **Extensibility**: Simple interfaces for adding new components
- **Distillation**: Comprehensive framework for knowledge distillation of transformer models

The project includes implementations of multiple transformer architectures, including a standard Vanilla Transformer, the Simplified Attention Sub-Block with Projections (SASP) variant, and Token-Factored Transformers.

## Project Structure

```
cleanGPT
├── __init__.py
├── config.py
├── config_distillation.py
├── distillation/
│   ├── __init__.py
│   ├── distillation_trainer.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate_distilled_model.py
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── hidden_state_loss.py
│   │   └── logit_loss.py
│   ├── trainers/
│   │   ├── __init__.py
│   │   ├── backbone_trainer.py
│   │   ├── base_trainer.py
│   │   └── head_trainer.py
│   └── utils/
│       ├── __init__.py
│       └── checkpoint.py
├── doc/
│   ├── README_cleanGPT_Distillation_Process.md
│   └── README_cleanGPT_Tutorial.md
├── examples/
│   ├── run_distillation_example.py
│   ├── train_factored_transformer.py
│   ├── train_sasp_char.py
│   └── train_{model}_{tokenizer}.py
├── inference/
│   ├── __init__.py
│   ├── generation.py
│   └── sampling_strategies.py
├── main.py
├── model/
│   ├── __init__.py
│   ├── model_SASPV.py
│   ├── model_SASPV_distillation.py
│   ├── model_Vanilla.py
│   ├── model_token_factored.py
│   ├── model_token_factored_distillation.py
│   └── model_vanilla_distillation.py
├── mytokenizers/
│   ├── __init__.py
│   ├── base_tokenizer.py
│   ├── character_tokenizer.py
│   ├── factory.py
│   └── gpt2_tokenizer.py
├── requirements.txt
├── setup.py
├── stitching_layers.py
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
  - Vanilla Transformer with Pre-LayerNorm as a baseline architecture
  - [SASP Transformer](https://github.com/ckerce/Transformer-Sandbox/blob/master/docs/simplified-transformers_README.md) (Simplified Attention Sub-Block with Projections) to examine alternative paths for information flow
  - Token-Factored Transformer -- in progress
- **Model Distillation**: Block-by-block distillation with stitching layers for efficient knowledge transfer
- **Extensible Training**: Modular training loop with customizable strategies
- **Advanced Text Generation**: Multiple sampling methods (greedy, top-k, top-p, etc.)
- **Token Analysis Tools**: Analyze tokenizer performance and optimize vocabularies
- **Comprehensive Evaluation**: Tools for evaluating model performance and comparing distilled models

## Getting Started

### Installation

```bash
git clone https://github.com/ckerce/cleanGPT.git
cd cleanGPT
pip install -r requirements.txt
```

### Training a Model

```bash
python main.py --model_type SASP --tokenizer_type gpt2 --n_layer 6 --n_head 6 --n_embd 384
```

### Distilling a Model

```bash
python examples/run_distillation_example.py \
    --teacher_model_name_or_path "gpt2" \
    --student_model_type "Factored" \
    --student_n_embd 384 \
    --student_n_head 6 \
    --dataset_name "roneneldan/TinyStories" \
    --dataset_text_column "story" \
    --epochs_per_block 3 \
    --batch_size 32 \
    --output_dir "./distilled_model_output"
```

### Analyzing Token Usage

```bash
python examples/token_analysis.py --dataset wikitext --tokenizer_type gpt2 --max_samples 1000 --plot
```

## Core Components

### Tokenizers

The `tokenizers` module provides a unified interface for different tokenization strategies:

```python
from mytokenizers import create_tokenizer

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

### Distillation

The `distillation` module provides comprehensive tools for model distillation:

```python
from distillation.distillation_trainer import DistillationTrainer

# Initialize distillation trainer
distill_trainer = DistillationTrainer(
    teacher_model=teacher_model,
    student_model=student_model,
    tokenizer=tokenizer,
    train_dataloader=train_dataloader,
    device=device,
    output_dir="./distilled_model_output",
    use_stitching_layers=True,
    logit_loss_type="kl_div"
)

# Run distillation
results = distill_trainer.train(
    epochs_per_block=3,
    lr_per_block=1e-3,
    train_lm_head=True,
    lm_head_epochs=3
)
```

### Evaluation

For evaluating distilled models:

```bash
python -m distillation.evaluation.evaluate_distilled_model \
    --model_path "./distilled_model_output/student_model_final_distilled.pt" \
    --model_type "Factored" \
    --teacher_model_name "gpt2" \
    --dataset "wikimedia/wikipedia" \
    --dataset_config "20231101.en" \
    --generate_comparisons
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

1. Create a new class in `mytokenizers/` that inherits from `BaseTokenizer`
2. Implement all required methods
3. Register it in `TokenizerFactory.TOKENIZER_TYPES`

### Adding a New Model Architecture

1. Create a new model class in `model/`
2. Register it in `MODEL_REGISTRY` in `model/__init__.py`
3. If intended for distillation, create a corresponding version with proper hidden state outputs

### Adding a New Training Strategy

1. Create a new trainer class in `trainers/` that inherits from `BaseTrainer` 
2. Register it in `TRAINER_REGISTRY` in `trainers/__init__.py`

### Creating a Custom Distillation Loss

1. Create a new loss class in `distillation/losses/`
2. Inherit from `nn.Module` and implement the forward method
3. Update the relevant trainer to use your custom loss

## Documentation

- [General Tutorial](doc/README_cleanGPT_Tutorial.md): Getting started with the framework
- [Distillation Tutorial](doc/README_cleanGPT_Distillation_Process.md): Comprehensive guide to the distillation framework

## Model Architectures

### Vanilla Transformer

Standard transformer implementation with pre-LayerNorm, similar to GPT-2.

### SASP Transformer

Simplified Attention Sub-Block with Projections - a more parameter-efficient variant with comparable performance.

### Token-Factored Transformer

A factorized implementation that reduces parameter count by sharing projections across token dimensions.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.20.0+
- Datasets 2.0.0+
- tqdm

## Citation

If you use this codebase in your research, please cite:

```
@software{cleangpt2025,
  author = {Clayton Kerce},
  title = {cleanGPT: Transformer Architecture Sandbox for Exploration of Language Models Features},
  year = {2025},
  url = {https://github.com/ckerce/cleanGPT}
}
```
