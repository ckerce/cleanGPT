# cleanGPT Distillation Tutorial

This tutorial provides a comprehensive guide to understanding and using the modular distillation framework in cleanGPT. Knowledge distillation is a technique where a smaller "student" model learns to mimic the behavior of a larger "teacher" model, resulting in a more efficient model with comparable performance.

## Table of Contents

1. [Overview](#overview)
2. [Framework Architecture](#framework-architecture)
3. [Student Model Requirements](#student-model-requirements)
4. [Setting Up Distillation](#setting-up-distillation)
5. [Distillation Options](#distillation-options)
6. [Advanced Features](#advanced-features)
   - [Stitching Layers](#stitching-layers)
   - [LM Head Training](#lm-head-training)
7. [Evaluating Distilled Models](#evaluating-distilled-models)
8. [Complete Example](#complete-example)
9. [Extending the Framework](#extending-the-framework)
10. [Troubleshooting](#troubleshooting)

## Overview

The distillation framework in cleanGPT implements block-by-block distillation for transformer models. This approach distills each transformer layer sequentially, allowing for more efficient and effective knowledge transfer. The framework supports:

- Different model architectures (Vanilla, Factored, SASP transformers)
- Flexible loss functions for both hidden states and logits
- Optional stitching layers for handling dimensional mismatches
- Separate language model head training
- Comprehensive evaluation tools

## Framework Architecture

The distillation module is organized in a modular, hierarchical structure:

```
distillation/
├── __init__.py                 # Top-level exports
├── distillation_trainer.py     # Main orchestrator
├── evaluation/                 # Evaluation tools
│   ├── __init__.py
│   └── evaluate_distilled_model.py
├── losses/                     # Loss functions
│   ├── __init__.py
│   ├── hidden_state_loss.py    # Hidden state distillation losses
│   └── logit_loss.py           # Logit distillation losses
├── trainers/                   # Training components
│   ├── __init__.py
│   ├── backbone_trainer.py     # Transformer layers trainer
│   ├── base_trainer.py         # Base class with common functionality
│   └── head_trainer.py         # LM head trainer
└── utils/                      # Utilities
    ├── __init__.py
    └── checkpoint.py           # Checkpoint handling
```

The `stitching_layers.py` file at the root level provides the stitching layer implementation for bridging dimensional differences between teacher and student hidden states.

### Key Components

1. **DistillationTrainer**: The main orchestrator that coordinates the entire distillation process
2. **BackboneDistillationTrainer**: Handles transformer block distillation
3. **HeadDistillationTrainer**: Handles language model head distillation
4. **StitchingLayer**: Projects student hidden states to teacher dimensions

## Student Model Requirements

To create a student model compatible with the distillation framework, it needs to:

1. Implement the same interface as the teacher model (typically GPT2LMHeadModel from HuggingFace)
2. Support block-by-block distillation with hidden state outputs
3. Have proper config attributes

### Model Classes

The repository provides several student model implementations:

- `model_vanilla_distillation.py`: Standard transformer implementation
- `model_token_factored_distillation.py`: Factored transformer implementation
- `model_SASPV_distillation.py`: SASP transformer implementation

Each model must:

1. Return all hidden states for distillation
2. Have a compatible configuration
3. Support token-by-token generation

### Configuration Requirements

The student model configuration should include:

- `n_layer`: Number of transformer layers (should match the teacher)
- `vocab_size`: Size of the vocabulary (should match the teacher)
- `output_hidden_states`: Set to `True` for distillation
- `teacher_n_embd`: The teacher's embedding dimension (for stitching layers)

Example model configuration:

```python
from config_distillation import GPTConfig

student_config = GPTConfig(
    block_size=128,                  # Context window size
    vocab_size=50257,                # Vocabulary size (match teacher)
    n_layer=12,                      # Number of layers (match teacher)
    padding_idx=0,                   # Padding token ID
    n_embd=384,                      # Student embedding dimension (can be smaller)
    n_head=6,                        # Student attention heads
    dropout=0.1,                     # Dropout rate
    bias=False,                      # Whether to use bias
    model_type="Factored",           # Student model type
    output_hidden_states=True,       # Required for distillation
    teacher_n_embd=768               # Teacher's embedding dimension
)
```

## Setting Up Distillation

### Prerequisites

1. A pre-trained teacher model (typically from HuggingFace)
2. A tokenizer compatible with both models
3. A dataset for distillation training

### Basic Distillation Setup

```python
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
from distillation.distillation_trainer import DistillationTrainer
from model.model_token_factored_distillation import FactoredTransformerModelDistillation
from config_distillation import GPTConfig

# 1. Load teacher model
teacher_model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 2. Create student model config
student_config = GPTConfig(
    model_type="Factored",
    vocab_size=teacher_model.config.vocab_size,
    n_layer=teacher_model.config.n_layer,
    n_embd=384,                      # Smaller than teacher (768)
    n_head=6,                        # Fewer heads than teacher (12)
    dropout=0.1,
    block_size=teacher_model.config.n_positions,
    output_hidden_states=True,
    padding_idx=tokenizer.pad_token_id,
    teacher_n_embd=teacher_model.config.hidden_size
)

# 3. Initialize student model
student_model = FactoredTransformerModelDistillation(config=student_config)

# 4. Prepare dataloader (simplified example)
train_dataset = TextDataset(texts, tokenizer, block_size=128)
data_collator = DistillationDataCollator(tokenizer, block_size=128)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=16,
    collate_fn=data_collator,
    shuffle=True
)

# 5. Initialize distillation trainer
distill_trainer = DistillationTrainer(
    teacher_model=teacher_model,
    student_model=student_model,
    tokenizer=tokenizer,
    train_dataloader=train_dataloader,
    device=torch.device("cuda"),
    output_dir="./distilled_model_output",
    # Backbone trainer parameters
    distill_loss_type="mse",
    use_stitching_layers=True,
    # Head trainer parameters
    logit_loss_type="kl_div",
    logit_loss_temperature=2.0
)

# 6. Run distillation
results = distill_trainer.train(
    epochs_per_block=3,
    lr_per_block=1e-3,
    train_lm_head=True,
    lm_head_epochs=3,
    lm_head_lr=1e-4,
    initialize_head_from_teacher=True
)
```

## Distillation Options

The framework provides many options to control the distillation process:

### Backbone Distillation Options

- `distill_loss_type`: Type of loss for hidden state distillation ("mse" or "kl_div")
- `distill_loss_temperature`: Temperature for KL divergence loss
- `use_stitching_layers`: Whether to use stitching layers for dimensional projection
- `stitching_layer_bias`: Whether to use bias in stitching layers
- `freeze_previous_blocks`: Whether to freeze previously trained blocks

### Training Hyperparameters

- `epochs_per_block`: Number of epochs to train each block
- `lr_per_block`: Learning rate for each block (can be a list for different rates per block)
- `wd_per_block`: Weight decay for each block
- `max_grad_norm_per_block`: Maximum gradient norm for clipping

### LM Head Training Options

- `train_lm_head`: Whether to train the language model head
- `lm_head_epochs`: Number of epochs for LM head training
- `lm_head_lr`: Learning rate for LM head training
- `lm_head_wd`: Weight decay for LM head training
- `logit_loss_type`: Type of loss for logit distillation ("mse", "kl_div", or "ce")
- `logit_loss_temperature`: Temperature for logit distillation
- `logit_loss_weight`: Weight of the logit loss relative to hidden state loss
- `initialize_head_from_teacher`: Whether to initialize LM head from teacher

## Advanced Features

### Stitching Layers

Stitching layers are trainable linear projections that map from student hidden dimensions to teacher hidden dimensions. They're crucial when student and teacher models have different hidden sizes.

```python
# Enable stitching layers in the DistillationTrainer
distill_trainer = DistillationTrainer(
    # ... other arguments
    use_stitching_layers=True,
    stitching_layer_bias=True  # Whether to use bias in the projections
)
```

Internally, each stitching layer consists of:

```python
# From stitching_layers.py
class StitchingLayer(nn.Module):
    def __init__(self, student_dim, teacher_dim, bias=True):
        super().__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        
        # Create a linear projection from student space to teacher space
        self.projection = nn.Linear(student_dim, teacher_dim, bias=bias)
        
        # Initialize appropriately
        if student_dim == teacher_dim:
            nn.init.eye_(self.projection.weight)  # Start with identity
            if bias:
                nn.init.zeros_(self.projection.bias)
        else:
            nn.init.xavier_uniform_(self.projection.weight)
            if bias:
                nn.init.zeros_(self.projection.bias)
```

### LM Head Training

The framework supports separate training for the language model head after distilling the transformer backbone:

```python
# Train the LM head
results = distill_trainer.train(
    # ... backbone parameters
    train_lm_head=True,
    lm_head_epochs=3,
    lm_head_lr=1e-4,
    lm_head_wd=0.01,
    lm_head_max_grad_norm=1.0,
    initialize_head_from_teacher=True  # Copy & project teacher weights
)
```

This separate phase can significantly improve the final model's performance.

## Evaluating Distilled Models

The framework provides an evaluation script (`evaluate_distilled_model.py`) to assess distilled model performance:

```bash
python -m distillation.evaluation.evaluate_distilled_model \
    --model_path "./distilled_model_output/student_model_final_distilled.pt" \
    --model_type "Factored" \
    --teacher_model_name "gpt2" \
    --dataset_name "wikitext" \
    --dataset_config_name "wikitext-2-raw-v1" \
    --dataset_split "test" \
    --max_samples 1000 \
    --batch_size 4 \
    --generate_comparisons
```

The evaluation includes:

1. **Perplexity**: Language modeling metric for both student and teacher
2. **Hidden State MSE**: Comparison of hidden state representations
3. **Generation Comparison**: Side-by-side comparison of text generation
4. **Visualization**: Plots of layer-wise MSE for analysis

The script produces comprehensive reports and visualizations to help understand how well the distillation worked.

## Complete Example

Here's a complete example using the provided `run_distillation_example.py`:

```bash
python -m examples.run_distillation_example \
    --teacher_model_name_or_path "gpt2" \
    --student_model_type "Factored" \
    --student_n_embd 384 \
    --student_n_head 6 \
    --dataset_name "roneneldan/TinyStories" \
    --dataset_text_column "story" \
    --block_size 128 \
    --batch_size 32 \
    --epochs_per_block 3 \
    --lr_per_block 1e-3 \
    --output_dir "./distilled_factored_gpt2" \
    --max_samples 1000 \
    --device "cuda" \
    --use_stitching_layers \
    --stitching_layer_bias \
    --log_interval 100 \
    --train_lm_head \
    --lm_head_epochs 3 \
    --lm_head_lr 1e-4 \
    --lm_head_weight_decay 0.01 \
    --logit_loss_type "kl_div" \
    --logit_loss_temperature 2.0 \
    --logit_loss_weight 1.0 \
    --initialize_head_from_teacher
```

After distillation, evaluate the model:

```bash
python -m distillation.evaluation.evaluate_distilled_model \
    --model_path "./distilled_factored_gpt2/student_model_final_distilled.pt" \
    --model_type "Factored" \
    --teacher_model_name "gpt2" \
    --dataset_name "roneneldan/TinyStories" \
    --dataset_split "validation" \
    --dataset_text_column "story" \
    --max_samples 500 \
    --generate_comparisons
```

## Extending the Framework

### Creating a Custom Student Model

1. Create a new model class inheriting from an appropriate base:

```python
# ./model/model_my_custom_distillation.py
import torch
import torch.nn as nn
from model.model_vanilla_distillation import VanillaTransformerModelDistillation

class MyCustomTransformerModelDistillation(VanillaTransformerModelDistillation):
    """Custom architecture for distillation."""
    
    def __init__(self, config):
        super().__init__(config)
        # Add custom layers or modifications
        
    def forward(self, input_ids, attention_mask=None, labels=None, output_hidden_states=None):
        """Forward pass with hidden state outputs for distillation."""
        # Ensure hidden_states are returned for distillation
        output_hidden_states = True if output_hidden_states is None else output_hidden_states
        return super().forward(input_ids, attention_mask, labels, output_hidden_states)
        
    def get_num_params(self):
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
```

2. Register your model in the factory function:

```python
# In examples/run_distillation_example.py or your custom script
def get_model_class(model_type_name: str):
    """Helper function to get the student model class based on its type name."""
    model_type_name_lower = model_type_name.lower()
    if model_type_name_lower == "factored":
        return FactoredTransformerModelDistillation
    elif model_type_name_lower == "sasp":
        return SASPTransformerModelDistillation
    elif model_type_name_lower == "vanilla":
        return VanillaTransformerModelDistillation
    elif model_type_name_lower == "mycustom":
        return MyCustomTransformerModelDistillation
    else:
        raise ValueError(f"Unsupported student model type: {model_type_name}")
```

### Custom Loss Functions

You can implement custom loss functions for distillation:

1. Create a new loss function in the losses directory:

```python
# ./distillation/losses/my_custom_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCustomDistillationLoss(nn.Module):
    """Custom distillation loss function."""
    
    def __init__(self, alpha=0.5, temperature=1.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.mse = nn.MSELoss()
        
    def forward(self, student_outputs, teacher_outputs):
        """
        Compute a weighted combination of MSE and cosine similarity.
        """
        # MSE component
        mse_loss = self.mse(student_outputs, teacher_outputs)
        
        # Cosine similarity component (maximize similarity)
        student_norm = F.normalize(student_outputs, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_outputs, p=2, dim=-1)
        cosine_loss = 1.0 - (student_norm * teacher_norm).sum(dim=-1).mean()
        
        # Combined loss
        return self.alpha * mse_loss + (1 - self.alpha) * cosine_loss
```

2. Update the trainer to use your custom loss:

```python
# In your distillation script
from distillation.losses.my_custom_loss import MyCustomDistillationLoss

# Create custom loss
custom_loss = MyCustomDistillationLoss(alpha=0.7, temperature=2.0)

# Use in trainer
class CustomBackboneTrainer(BackboneDistillationTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_state_loss_fn = custom_loss
```

## Troubleshooting

### Common Issues and Solutions

1. **Dimension Mismatch Errors**

```
ValueError: Dimension mismatch at block X without stitching: Student_dim=Y, Teacher_dim=Z.
```

Solution: Enable stitching layers with `use_stitching_layers=True` or ensure student and teacher dimensions match.

2. **Memory Issues**

```
CUDA out of memory
```

Solutions:
- Reduce batch size
- Use gradient accumulation
- Use a smaller student model
- Distill fewer layers at a time

3. **NaN Losses**

```
Error: Loss is nan. Skipping batch.
```

Solutions:
- Check for extremely large or small values in hidden states
- Reduce learning rate
- Use gradient clipping
- Add epsilon to prevent division by zero

4. **Slow Distillation**

Solutions:
- Use fewer epochs per block for initial experiments
- Start with a smaller dataset
- Use a small subset for hyperparameter tuning
- Consider freezing teacher embeddings

5. **Poor Performance**

Solutions:
- Increase epochs per block
- Try different loss functions
- Adjust temperature for KL divergence
- Ensure sufficient training data
- Balance hidden state loss and logit loss

### Debugging Tips

1. Use smaller models during development:
```python
student_config = GPTConfig(
    n_layer=2,  # Use fewer layers for debugging
    n_embd=128,
    n_head=2
)
```

2. Enable extensive logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

3. Monitor loss values across epochs:
```python
for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        # ... training step ...
        if step % 10 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")
```

4. Visualize hidden states during distillation:
```python
# After computing hidden states
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(student_hidden_state[0, 0].detach().cpu().numpy())
plt.title("Student Hidden State")
plt.subplot(1, 2, 2)
plt.imshow(teacher_hidden_state[0, 0].detach().cpu().numpy())
plt.title("Teacher Hidden State")
plt.savefig(f"hidden_states_block_{block_idx}_epoch_{epoch}.png")
```

## Conclusion

This tutorial covered the essential components and usage of the distillation framework in cleanGPT. The modular design allows for flexible experimentation with different model architectures, loss functions, and training strategies.

Hypotheses under investigation:

1. Block-by-block distillation enables effective knowledge transfer
2. Stitching layers provide effictive mechanisms to bridge dimension differences between teacher and student
3. Separate LM head training improves final model quality


Key properties of the implementation:
1. The framework supports various student architectures and loss functions
2. Evaluation tools are being developed to assess distillation effectiveness

By following this guide, you should be able to distill your own models and extend the framework for your specific research needs.
