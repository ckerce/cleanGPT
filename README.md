# cleanGPT

## Overview

cleanGPT is a PyTorch-based project designed to provide a clean, modular, and understandable environment for training and experimenting with sequence-to-sequence Transformer models. It prioritizes code clarity and separation of concerns, making it easier to modify, extend, and swap different model architectures.

This project was initially adapted from an implementation focusing on the **Simplified Attention Sub-Block with Projections and Value options (SAS-PV / SAS-P / SAS)** architectures.

## Project Structure

The codebase is organized into several distinct Python modules:

* **`config.py`**: Centralizes configuration settings, including dataset parameters, model hyperparameters (using `GPTConfig`), training options, and device setup.
* **`model.py`**: Defines the neural network architectures. It currently includes the `SASPTransformerModel` built from SAS-P components (`SimplifiedTransformerBlock`, `CausalShapedAttention`, `SASMLP`, `SASLayerNorm`) and can be easily extended with other model types.
* **`data_utils.py`**: Handles data loading (using Hugging Face `datasets`), tokenization (using Hugging Face `transformers`), preprocessing, and DataLoader creation.
* **`trainer.py`**: Contains the `Trainer` class, encapsulating the model training loop logic (forward pass, loss calculation, backpropagation, optimization).
* **`inference.py`**: Provides utility functions for generating text sequences using a trained model (e.g., `run_generation`).
* **`main.py`**: The main orchestration script that imports components, initializes the chosen model, data, trainer, and runs the training and inference workflow.

## Core Architecture: SAS-P Transformer

The primary model implemented in this repository is based on the **Simplified Transformer Block** architecture.

Details for the method come from the paper [Simplifying Transformer Blocks](https://arxiv.org/abs/2311.01906) by Bobby He and Thomas Hofmann @ ETH Zurich. Their approach uses signal propagation concepts and careful experimental analysis to trim down and re-organize the standard Transformer block.

Key features of the SAS-P architecture include:

1.  **Reduced Normalization:** Fewer LayerNorm components compared to standard Pre-LN or Post-LN blocks.
2.  **Parallelized Layers:** The multi-head attention (specifically, Causal Shaped Attention) and the feed-forward network (MLP) are processed in parallel branches.
3.  **No Skip Connections:** Traditional residual skip connections are removed.
4.  **Component Simplification:** The research suggests that the Value (V) matrix in attention and the post-attention Projection matrix (W_O) might not be strictly necessary for stable training in this simplified structure, potentially reducing parameter count.

<p align="center">
  <img src="path/to/your/assets/SAS-P.png" alt="SAS-P Block Diagram" style="width: 80%; height: auto;">
  <em>Figure 1: Simplified Transformer Block (SAS-P) Architecture (Source: He & Hofmann, 2023)</em>
</p>

The attention mechanism used is **Causal Shaped Attention**, which incorporates specific learnable parameters (`alpha`, `beta`, `gamma`) and buffer components (`MC`, `Id`) to modify the standard attention mechanism. This is detailed further in [Noci, Li, Li, He, Hoffman, Madison, and Roy; The Shaped Transformer: Attention Models in the Infinite Depth-and-Width Limit](https://arxiv.org/abs/2306.17759).

The core Python classes implementing this architecture in `model.py` are:

* `SASLayerNorm`: Custom Layer Normalization with optional bias.
* `SASMLP`: The feed-forward network component (with an optional LLaMA-style variant).
* `CausalShapedAttention`: The modified multi-head self-attention mechanism.
* `SimplifiedTransformerBlock`: Combines LayerNorm, Attention, and MLP according to the SAS-P structure.
* `SASPTransformerModel`: The complete GPT-like model composed of embedding layers and multiple `SimplifiedTransformerBlock` layers.

### SAS-P Algorithm Steps

The parallel processing within the `SimplifiedTransformerBlock` follows these general steps:

<p align="center">
  <img src="path/to/your/assets/SAS-P_Algorithm_Steps.png" alt="SAS-P Algorithm Steps" style="width: 80%; height: auto;">
  <em>Figure 2: SAS-P Algorithm Steps (Parallel Attention and MLP)</em>
</p>

## Usage

1.  **Setup:** Ensure you have PyTorch and the Hugging Face `datasets` and `transformers` libraries installed.
    ```bash
    pip install torch datasets transformers tqdm
    ```
2.  **Configure:** Adjust parameters in `config.py` as needed (dataset, model size, training settings).
3.  **Run:** Execute the main script.
    ```bash
    python main.py
    ```
    The script will handle data loading, model initialization, training, and a final inference step.

## Configuration

Most hyperparameters and settings can be modified directly in `config.py`:

* **Dataset:** `DATASET_NAME`, `DATASET_CONFIG`, `MAX_SAMPLES`.
* **Model:** Modify the `GPTConfig` dataclass for `block_size`, `vocab_size` (set dynamically), `n_layer`, `n_head`, `n_embd`, `dropout`, `bias`, and SAS-P specific flags (`use_proj`, `use_v`, `llama_mlp`).
* **Training:** `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`.
* **Inference:** `GENERATION_MAX_LEN`.

## Extending the Model

The modular structure makes it easy to add or modify model architectures:

1.  Define your new model class (e.g., `MyTransformerModel`) in `model.py`. Ensure its `forward` method returns a dictionary containing at least a `'loss'` key when labels are provided. Implement a `.generate()` method for inference.
2.  Import your new model in `main.py`.
3.  Change the model initialization line in `main.py` to use your new class:
    ```python
    # from model import SASPTransformerModel
    from model import MyTransformerModel # Import your model

    # ... inside main() ...

    # model = SASPTransformerModel(config=model_config)
    model = MyTransformerModel(config=model_config) # Instantiate your model
    ```
4.  Adjust the `GPTConfig` in `config.py` or create a new config class if your model requires different parameters.

The existing `Trainer` and `data_utils` should largely remain compatible as long as the model interface (input/output of `forward`, existence of `.parameters()`, `.generate()`) is consistent.
