#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_distilled_model.py - Evaluate a distilled transformer model.

This script loads a distilled model from a checkpoint and evaluates its performance on
test data, calculating perplexity, comparing hidden states with a teacher (if provided),
and generating sample texts.

Usage:
    python evaluate_distilled_model.py --model_path "./distilled_model_output/student_model_final_distilled.pt" \
                                      --model_type "Factored" \
                                      --dataset_name "wikitext" \
                                      --dataset_config_name "wikitext-2-raw-v1" \
                                      --dataset_split "test" \
                                      --max_samples 1000 \
                                      --batch_size 4

For teacher model comparison (including hidden state MSE if dimensions allow or stitching layers are handled):
    python evaluate_distilled_model.py --model_path "./distilled_model_output/student_model_final_distilled.pt" \
                                      --model_type "Factored" \
                                      --teacher_model_name "gpt2" \
                                      --dataset_name "wikitext" \
                                      --dataset_config_name "wikitext-2-raw-v1" \
                                      --dataset_split "test" \
                                      --max_samples 1000
"""

import argparse
import logging
import os
import sys
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List
from tqdm.auto import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Model, GPT2Config as HF_GPT2Config
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

# --- Path Setup ---
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_path, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_script_path not in sys.path:
     sys.path.insert(0, current_script_path)

# --- Custom Module Imports ---
try:
    from model.model_token_factored_distillation import FactoredTransformerModelDistillation
    from model.model_SASPV_distillation import SASPTransformerModelDistillation
    from model.model_vanilla_distillation import VanillaTransformerModelDistillation
    from config_distillation import GPTConfig, print_config, DEVICE
    from stitching_layers import StitchingLayer # Needed for hidden state comparison with stitching
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Please ensure 'config_distillation.py', 'stitching_layers.py', and the 'model' directory "
          "are correctly placed and in sys.path.")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def get_model_class(model_type: str):
    """Get the appropriate model class based on model type."""
    model_type_lower = model_type.lower()
    if model_type_lower == "factored":
        return FactoredTransformerModelDistillation
    elif model_type_lower == "sasp":
        return SASPTransformerModelDistillation
    elif model_type_lower == "vanilla":
        return VanillaTransformerModelDistillation
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported: Factored, SASP, Vanilla.")

def load_model_from_checkpoint(model_path: str, model_type: str, device: torch.device) -> Optional[Dict[str, Any]]:
    """
    Load a distilled model from checkpoint.
    Returns a dictionary containing the model, its configuration, and stitching layer info, or None on failure.
    """
    logger.info(f"Loading model from {model_path}")
    
    try:
        if 'config_distillation.GPTConfig' not in torch.serialization.get_default_safe_globals():
            torch.serialization.add_safe_globals([GPTConfig])
        if 'stitching_layers.StitchingLayer' not in torch.serialization.get_default_safe_globals(): # If StitchingLayer objects were ever pickled directly
            torch.serialization.add_safe_globals([StitchingLayer])


        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            logger.info("Checkpoint loaded with weights_only=False.")
        except Exception as e_false:
            logger.warning(f"Failed to load checkpoint with weights_only=False: {e_false}")
            logger.warning("Attempting to load with weights_only=True.")
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                logger.info("Checkpoint loaded with weights_only=True.")
            except Exception as e_true:
                logger.error(f"Failed to load checkpoint with both weights_only settings: {e_true}", exc_info=True)
                return None
        
        config_data = checkpoint.get('student_config')
        if config_data is None:
            logger.error("No 'student_config' found in checkpoint.")
            return None
        
        if isinstance(config_data, dict):
            logger.info("Instantiating GPTConfig from dictionary.")
            try:
                config = GPTConfig(**config_data)
            except TypeError as te:
                logger.error(f"Error creating GPTConfig from dict: {te}. Config dict: {config_data}")
                return None
        elif isinstance(config_data, GPTConfig):
            logger.info("Using GPTConfig object from checkpoint.")
            config = config_data
        else:
            logger.error(f"Unsupported type for 'student_config': {type(config_data)}")
            return None
            
        if not hasattr(config, 'model_type') or config.model_type is None:
            logger.warning(f"model_type not in config. Using CLI provided: '{model_type}'")
            config.model_type = model_type
        elif config.model_type.lower() != model_type.lower():
            logger.warning(f"Config model_type ('{config.model_type}') != CLI ('{model_type}'). Using config's.")
        
        ModelClass = get_model_class(config.model_type)
        model = ModelClass(config)
        
        if 'model_state_dict' not in checkpoint:
            logger.error("No 'model_state_dict' in checkpoint.")
            return None
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(device)
        model.eval()
        
        # Load stitching layer information if present in the checkpoint
        stitching_layers_state_dict = checkpoint.get('stitching_layers_state_dict')
        use_stitching_layers_from_checkpoint = checkpoint.get('use_stitching_layers', False)
        
        if use_stitching_layers_from_checkpoint and stitching_layers_state_dict:
            logger.info("Stitching layer state dictionary found in checkpoint.")
        elif use_stitching_layers_from_checkpoint and not stitching_layers_state_dict:
            logger.warning("'use_stitching_layers' is true in checkpoint, but 'stitching_layers_state_dict' is missing.")
        
        logger.info(f"Successfully loaded '{config.model_type}' model ({model.get_num_params()/1e6:.2f}M params).")
        if 'print_config' in globals() and callable(print_config): print_config(config)
        
        return {
            "model": model, 
            "config": config,
            "stitching_layers_state_dict": stitching_layers_state_dict,
            "use_stitching_layers": use_stitching_layers_from_checkpoint
        }
    
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found: {model_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        return None

def load_teacher_model(model_name: str, device: torch.device, lm_head: bool = False):
    """Load a teacher model from Hugging Face."""
    logger.info(f"Loading teacher model: {model_name} {'with' if lm_head else 'without'} LM head.")
    try:
        teacher_config = HF_GPT2Config.from_pretrained(model_name, output_hidden_states=True) # Ensure hidden states output
        ModelClass = GPT2LMHeadModel if lm_head else GPT2Model
        teacher_model = ModelClass.from_pretrained(model_name, config=teacher_config)
        
        teacher_model.to(device)
        teacher_model.eval()
        logger.info(f"Teacher model '{model_name}' loaded successfully.")
        return {"model": teacher_model, "config": teacher_config}
    except Exception as e:
        logger.error(f"Error loading teacher model '{model_name}': {e}", exc_info=True)
        return None

class EvaluationDataset(Dataset):
    """Dataset for language model evaluation (perplexity)."""
    def __init__(self, texts: List[str], tokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size # Max sequence length for input_ids and labels
        self.examples = []
        
        logger.info(f"Tokenizing {len(texts)} texts for evaluation dataset...")
        for text in tqdm(texts, desc="Encoding texts for evaluation"):
            if not text or not isinstance(text, str) or not text.strip():
                continue # Skip empty or invalid texts
            
            # Tokenize, ensuring space for shifted labels.
            # Example: "Hello world" -> input_ids: [H, e, l, l, o], labels: [e, l, l, o, w]
            # So, we tokenize up to block_size + 1, then slice.
            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=self.block_size + 1, 
                padding=False, # Collator will handle padding
                return_tensors=None, 
                add_special_tokens=True # Usually True for LM tasks
            )
            
            input_ids_full = tokenized['input_ids']
            
            # We need at least 2 tokens to form a pair (input_id, label)
            if len(input_ids_full) <= 1:
                continue
                
            # Input is all but last, label is all but first.
            self.examples.append({
                "input_ids": input_ids_full[:-1],
                "labels": input_ids_full[1:] 
            })
            
        logger.info(f"Created {len(self.examples)} examples for evaluation dataset.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # The collator will truncate/pad these to self.block_size
        return self.examples[i]

class EvaluationCollator:
    """Collate function for evaluation, handling padding and label creation."""
    def __init__(self, tokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size # This is the target length after padding/truncation
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad_token_id or eos_token_id for padding.")

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = []
        batch_labels = []
        batch_attention_masks = []
        
        for ex in examples:
            # Truncate to block_size if longer
            input_ids = ex["input_ids"][:self.block_size]
            labels = ex["labels"][:self.block_size]
            
            # Pad to block_size
            padding_length = self.block_size - len(input_ids)
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            
            input_ids_padded = input_ids + [self.pad_token_id] * padding_length
            # Use -100 for labels where padding occurs, as this is ignored by CrossEntropyLoss
            labels_padded = labels + [-100] * padding_length 
            
            batch_input_ids.append(input_ids_padded)
            batch_labels.append(labels_padded)
            batch_attention_masks.append(attention_mask)
        
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_masks, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long) # Labels for loss calculation
        }

def calculate_perplexity(model, dataloader, device):
    """Calculate perplexity."""
    model.eval()
    total_loss = 0.0
    total_tokens_for_loss = 0 # Count only non-padding tokens that contributed to loss
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating perplexity", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device) # Labels are already shifted and padded with -100
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            # The loss is already averaged over the batch by the model's forward if labels are provided
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs["loss"]
            
            # Number of tokens that contributed to this batch's loss (non -100 labels)
            num_active_tokens = (labels != -100).sum().item()
            
            if num_active_tokens > 0:
                total_loss += loss.item() * num_active_tokens # Re-scale loss before averaging over all tokens
                total_tokens_for_loss += num_active_tokens
            elif loss.item() != 0.0 : # If loss is non-zero with no active tokens, it's unusual
                 logger.warning(f"Batch resulted in loss {loss.item()} but had 0 active tokens for loss calculation.")

    if total_tokens_for_loss == 0:
        logger.warning("No tokens were processed for perplexity calculation. Returning inf.")
        return {"avg_loss": float('inf'), "perplexity": float('inf'), "total_tokens": 0}

    avg_loss = total_loss / total_tokens_for_loss
    perplexity = math.exp(avg_loss) if avg_loss != float('inf') else float('inf')
    
    return {"avg_loss": avg_loss, "perplexity": perplexity, "total_tokens": total_tokens_for_loss}

def teacher_hidden_state_evaluation(
    student_model, student_config, 
    teacher_model, teacher_config, 
    dataloader, device, 
    stitching_info: Optional[Dict[str, Any]] = None
):
    """Calculate MSE between student and (optionally projected) teacher hidden states."""
    student_model.eval()
    teacher_model.eval()
    
    num_student_layers = student_config.n_layer
    # Initialize MSE storage for each layer (embedding layer + transformer layers)
    # Hidden states usually include embedding output as [0], then transformer layers [1]...[N]
    layer_mse_sums = [0.0] * (num_student_layers + 1) # +1 for embedding layer
    layer_counts = [0] * (num_student_layers + 1)
    
    # Determine if stitching layers should be dynamically loaded and used
    use_stitching = stitching_info and stitching_info.get('use_stitching_layers', False)
    stitching_layers_state_dict = stitching_info.get('stitching_layers_state_dict') if use_stitching else None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Comparing hidden states", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device) # For masking MSE calculation
            
            student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            student_hidden_states = student_outputs.get('hidden_states') # List of tensors
            teacher_hidden_states = teacher_outputs.get('hidden_states') # List or tuple of tensors
            
            if not student_hidden_states or not teacher_hidden_states:
                logger.warning("Hidden states not found in model outputs. Skipping batch for MSE.")
                continue

            # Number of layers to compare, min of (student layers+emb, teacher layers+emb)
            # Student hidden_states: [emb_output, layer1_out, ..., layerN_out] -> N+1 states
            # Teacher (GPT2) hidden_states: [emb_output, layer1_out, ..., layerM_out] -> M+1 states
            num_layers_to_compare = min(len(student_hidden_states), len(teacher_hidden_states), num_student_layers + 1)

            for i in range(num_layers_to_compare):
                student_h = student_hidden_states[i] # Shape: [batch, seq_len, student_dim]
                teacher_h = teacher_hidden_states[i] # Shape: [batch, seq_len, teacher_dim]
                
                projected_student_h = student_h
                
                # If using stitching and dimensions differ for this layer
                if use_stitching and stitching_layers_state_dict and student_h.shape[-1] != teacher_h.shape[-1]:
                    # Stitching layers in trainer are indexed 0 to N-1 for transformer blocks.
                    # Hidden state index i=0 is embeddings, i=1 is block 0 output, etc.
                    # So, stitching layer for hidden_state[i] (if i > 0) is stitching_layers[str(i-1)]
                    stitching_layer_trainer_idx = i - 1 
                    
                    if stitching_layer_trainer_idx >= 0:
                        layer_key_prefix = str(stitching_layer_trainer_idx)
                        # Check if this specific stitching layer's weights are in the state dict
                        # A simple check for one of its parameters:
                        if any(key.startswith(f"{layer_key_prefix}.projection.weight") for key in stitching_layers_state_dict):
                            try:
                                s_dim_actual = student_h.shape[-1]
                                t_dim_actual = teacher_h.shape[-1]
                                bias_present = any(key.startswith(f"{layer_key_prefix}.projection.bias") for key in stitching_layers_state_dict)

                                stitch_module = StitchingLayer(s_dim_actual, t_dim_actual, bias=bias_present).to(device)
                                
                                # Filter state_dict for this specific layer module
                                current_layer_sd = {
                                    k.split('.', 1)[1]: v 
                                    for k, v in stitching_layers_state_dict.items() 
                                    if k.startswith(layer_key_prefix + ".")
                                }
                                stitch_module.load_state_dict(current_layer_sd)
                                stitch_module.eval()
                                projected_student_h = stitch_module(student_h)
                                logger.debug(f"Applied loaded stitching layer {stitching_layer_trainer_idx} for hidden state comparison (idx {i}).")
                            except Exception as e:
                                logger.warning(f"Failed to load/apply stitching layer {stitching_layer_trainer_idx} for HS idx {i}: {e}. Comparing raw states if dims match.")
                        else:
                            logger.debug(f"No stitching layer weights for trainer_idx {stitching_layer_trainer_idx} (HS idx {i}) in state_dict.")
                    else: # i == 0 (embedding layer), no per-block stitching layer typically applies here.
                        logger.debug(f"No stitching layer applied for embedding output (HS idx {i}).")


                # Calculate MSE only if dimensions now match
                if projected_student_h.shape[-1] == teacher_h.shape[-1]:
                    # Masked MSE: only consider non-padding tokens
                    # attention_mask is [B, T]. Expand for hidden_dim.
                    expanded_mask = attention_mask.unsqueeze(-1).expand_as(teacher_h)
                    
                    # Calculate squared error per element, apply mask, sum, and normalize by active elements.
                    # This computes MSE for each item in the batch, then averages.
                    # Sum over hidden_dim, then sum over seq_len (masked), then mean over batch.
                    mse_val = F.mse_loss(projected_student_h * expanded_mask, teacher_h * expanded_mask, reduction='sum')
                    num_active_elements = expanded_mask.sum().item()
                    
                    if num_active_elements > 0:
                        batch_avg_mse = mse_val / num_active_elements
                        layer_mse_sums[i] += batch_avg_mse.item() * input_ids.size(0) # Accumulate sum of MSEs weighted by batch size
                        layer_counts[i] += input_ids.size(0) # Count number of batches processed for this layer
                    
                else:
                    logger.warning(
                        f"Skipping MSE for HS idx {i}: Student dim {projected_student_h.shape[-1]} != Teacher dim {teacher_h.shape[-1]} "
                        f"(after potential stitching attempt)."
                    )
    
    # Calculate average MSE for each layer
    avg_layer_mse = []
    for idx in range(num_student_layers + 1): # Iterate up to the number of student layers + embedding
        if layer_counts[idx] > 0:
            avg_mse = layer_mse_sums[idx] / layer_counts[idx]
            avg_layer_mse.append(avg_mse)
        else:
            # If comparison was skipped for all batches (e.g. consistent dim mismatch and no stitching)
            # or if this layer index was beyond num_layers_to_compare for all batches.
            if idx < num_layers_to_compare : # Only append NaN if it was supposed to be compared
                 avg_layer_mse.append(float('nan'))
            # else: this layer was not compared, so don't add to list.
            # This logic needs to be careful to align with plot expectations.
            # For simplicity, if it wasn't compared, we can omit or use NaN.
            # Let's ensure avg_layer_mse has one entry per layer up to num_student_layers + 1
            # or up to num_layers_to_compare.
            # The plot function expects a list of MSEs.
    
    # Ensure avg_layer_mse has entries for all layers up to num_layers_to_compare
    # If a layer was consistently skipped, its entry will be NaN.
    # Pad with NaNs if avg_layer_mse is shorter than num_layers_to_compare (e.g. if some layers were never reached)
    while len(avg_layer_mse) < num_layers_to_compare:
        avg_layer_mse.append(float('nan'))

    # Overall average MSE (ignoring NaNs)
    valid_mses = [m for m in avg_layer_mse if not math.isnan(m)]
    overall_avg_mse = sum(valid_mses) / len(valid_mses) if valid_mses else float('nan')
    
    # Return list of layer-wise MSEs (0=embeddings, 1=block0_out, etc.)
    return {"avg_mse": overall_avg_mse, "layer_mse": avg_layer_mse[:num_layers_to_compare]}


def plot_layer_mse(layer_mse: List[float], output_path: str, model_type: str):
    """Plot MSE by layer."""
    if not layer_mse:
        logger.warning("Layer MSE list is empty. Skipping plot.")
        return

    plt.figure(figsize=(12, 7))
    # Layer indices: 0 for embeddings, 1 for layer 0 output, etc.
    layer_labels = [f"Emb."] + [f"L{i}" for i in range(len(layer_mse) -1)]
    
    # Filter out NaN values for plotting, but keep track of original indices for labels
    valid_indices = [i for i, mse in enumerate(layer_mse) if not math.isnan(mse)]
    valid_mse_values = [layer_mse[i] for i in valid_indices]
    valid_layer_labels = [layer_labels[i] for i in valid_indices]

    if not valid_mse_values:
        logger.warning("All layer MSE values are NaN. Skipping plot.")
        plt.close()
        return

    plt.bar(range(len(valid_mse_values)), valid_mse_values, color='skyblue')
    plt.xlabel('Model Layer (0=Embeddings, L_i=Transformer Block i Output)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title(f'Teacher-Student Hidden State MSE by Layer ({model_type})')
    plt.xticks(range(len(valid_mse_values)), valid_layer_labels, rotation=45, ha="right")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    for i, v in enumerate(valid_mse_values):
        plt.text(i, v + (max(valid_mse_values) * 0.01), f"{v:.2e}", ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Layer MSE plot saved to {output_path}")
    plt.close()

def generate_comparison_samples(student_model, teacher_model, tokenizer, prompts: List[str], 
                                max_new_tokens=50, temperature=0.8, top_k=40, device=None):
    """Generate text from student and teacher for comparison."""
    if not hasattr(teacher_model, 'generate'): # Ensure teacher can generate
        logger.warning("Teacher model does not have a 'generate' method. Skipping generation comparison.")
        return None
    if not hasattr(student_model, 'generate'): # Ensure student can generate
        logger.warning("Student model does not have a 'generate' method. Skipping generation comparison.")
        return None
        
    if device is None: device = next(student_model.parameters()).device
    results = []
    
    for prompt in tqdm(prompts, desc="Generating comparison samples", leave=False):
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            student_out_ids = student_model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k if top_k > 0 else None)
            teacher_out_ids = teacher_model.generate(input_ids, max_length=input_ids.size(1) + max_new_tokens, temperature=temperature, top_k=top_k if top_k > 0 else None) # HF generate uses max_length
            
        student_text = tokenizer.decode(student_out_ids[0], skip_special_tokens=True)
        teacher_text = tokenizer.decode(teacher_out_ids[0], skip_special_tokens=True)
        results.append({"prompt": prompt, "student_text": student_text, "teacher_text": teacher_text})
    return results

def save_generation_comparison(comparison_results: List[Dict[str,str]], output_path: str):
    """Save generated text comparisons to a file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Text Generation Comparison: Student vs. Teacher\n\n")
        for i, res in enumerate(comparison_results):
            f.write(f"## Sample {i+1}\n\n**Prompt:**\n{res['prompt']}\n\n")
            f.write(f"**Student Output:**\n{res['student_text']}\n\n")
            f.write(f"**Teacher Output:**\n{res['teacher_text']}\n\n")
            f.write("---\n\n")
    logger.info(f"Generation comparison saved to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a distilled transformer model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to student model checkpoint')
    parser.add_argument('--model_type', type=str, choices=['Factored', 'SASP', 'Vanilla'], required=True, help='Student model architecture type')
    parser.add_argument('--teacher_model_name', type=str, default=None, help='Teacher HF model name for comparison (e.g., "gpt2")')
    parser.add_argument('--dataset_name', type=str, default='wikitext', help='Dataset name from HF Datasets')
    parser.add_argument('--dataset_config_name', type=str, default='wikitext-2-raw-v1', help='Dataset configuration name')
    parser.add_argument('--dataset_split', type=str, default='test', help='Dataset split (e.g., "test", "validation")')
    parser.add_argument('--dataset_text_column', type=str, default='text', help='Name of the text column in the dataset')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples from dataset for evaluation (None for all)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation dataloader')
    parser.add_argument('--block_size', type=int, default=128, help='Context window / block size for tokenization')
    parser.add_argument('--generate_comparisons', action='store_true', help='Generate and save text samples from student and teacher')
    parser.add_argument('--num_comparison_prompts', type=int, default=5, help='Number of prompts for generation comparison')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='Directory to save results (plots, text files)')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default=None, help='Device (cpu, cuda, mps). Auto-detects if None.')
    parser.add_argument('--num_workers', type=int, default=0, help='Dataloader num_workers.')
    return parser.parse_args()

def main():
    args = parse_args()
    
    current_device = torch.device(args.device) if args.device else DEVICE
    logger.info(f"Using device: {current_device}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load student model
    student_data = load_model_from_checkpoint(args.model_path, args.model_type, current_device)
    if not student_data: return
    student_model, student_config = student_data["model"], student_data["config"]
    stitching_info = {
        "use_stitching_layers": student_data["use_stitching_layers"],
        "stitching_layers_state_dict": student_data["stitching_layers_state_dict"]
    }

    # Load tokenizer (consistent with training, typically "gpt2" based)
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load and prepare dataset
    logger.info(f"Loading dataset: {args.dataset_name} ({args.dataset_config_name or 'default'}) - Split: {args.dataset_split}")
    try:
        ds_args = [args.dataset_name]
        if args.dataset_config_name: ds_args.append(args.dataset_config_name)
        split_spec = f"{args.dataset_split}" + (f"[:{args.max_samples}]" if args.max_samples is not None else "")
        raw_dataset = load_dataset(*ds_args, split=split_spec, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}", exc_info=True); return

    if args.dataset_text_column not in raw_dataset.column_names:
        logger.error(f"Text column '{args.dataset_text_column}' not found. Available: {raw_dataset.column_names}. Exiting."); return
    
    texts = [t for t in raw_dataset[args.dataset_text_column] if isinstance(t, str) and t.strip()]
    logger.info(f"Using {len(texts)} non-empty text samples from dataset.")

    eval_block_size = min(args.block_size, getattr(student_config, 'block_size', args.block_size))
    eval_dataset = EvaluationDataset(texts, tokenizer, eval_block_size)
    eval_collator = EvaluationCollator(tokenizer, eval_block_size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=eval_collator, 
                                 shuffle=False, num_workers=args.num_workers, pin_memory=(current_device.type == 'cuda'))

    if len(eval_dataset) == 0:
        logger.error("Evaluation dataset is empty. Cannot proceed."); return

    # --- Student Model Evaluation ---
    logger.info(f"Evaluating student model ({args.model_type}) perplexity...")
    student_ppl_results = calculate_perplexity(student_model, eval_dataloader, current_device)
    logger.info(f"Student PPL: {student_ppl_results['perplexity']:.4f}, Avg Loss: {student_ppl_results['avg_loss']:.4f} (Tokens: {student_ppl_results['total_tokens']})")
    
    eval_summary = {
        "student_model": {"type": args.model_type, "path": args.model_path, **student_ppl_results},
        "dataset": {"name": args.dataset_name, "config": args.dataset_config_name, "split": args.dataset_split, "samples_used": len(eval_dataset)}
    }

    # --- Teacher Model Comparison (Optional) ---
    if args.teacher_model_name:
        eval_summary["teacher_model"] = {"name": args.teacher_model_name}
        
        # Teacher for Perplexity (needs LM head)
        teacher_lm_data = load_teacher_model(args.teacher_model_name, current_device, lm_head=True)
        if teacher_lm_data:
            teacher_lm_model = teacher_lm_data["model"]
            logger.info(f"Evaluating teacher model ({args.teacher_model_name}) perplexity...")
            teacher_ppl_results = calculate_perplexity(teacher_lm_model, eval_dataloader, current_device)
            logger.info(f"Teacher PPL: {teacher_ppl_results['perplexity']:.4f}, Avg Loss: {teacher_ppl_results['avg_loss']:.4f}")
            eval_summary["teacher_model"].update(teacher_ppl_results)
            if student_ppl_results['perplexity'] != float('inf') and teacher_ppl_results['perplexity'] != float('inf'):
                relative_ppl = student_ppl_results['perplexity'] / teacher_ppl_results['perplexity']
                logger.info(f"Relative PPL (Student/Teacher): {relative_ppl:.4f}")
                eval_summary["relative_perplexity"] = relative_ppl

            # Text Generation Comparison
            if args.generate_comparisons and args.num_comparison_prompts > 0:
                sample_prompts = [tokenizer.decode(eval_dataset[i]["input_ids"][:20]) for i in range(min(args.num_comparison_prompts, len(eval_dataset)))] # Take first 20 tokens as prompt
                if sample_prompts:
                    comparison_results = generate_comparison_samples(student_model, teacher_lm_model, tokenizer, sample_prompts, device=current_device)
                    if comparison_results:
                        save_generation_comparison(comparison_results, os.path.join(args.output_dir, f"generation_comparison_{args.model_type}.md"))
        
        # Teacher for Hidden States (base model is fine)
        teacher_base_data = load_teacher_model(args.teacher_model_name, current_device, lm_head=False)
        if teacher_base_data:
            teacher_base_model, teacher_base_config = teacher_base_data["model"], teacher_base_data["config"]
            logger.info(f"Comparing student hidden states with teacher ({args.teacher_model_name})...")
            
            # Create a dataloader for hidden state comparison (might use different tokenization/batching if needed, here using same)
            # Note: EvaluationDataset for perplexity uses labels. For hidden states, only input_ids are strictly needed.
            # The current dataloader is fine as it provides input_ids and attention_mask.
            hs_results = teacher_hidden_state_evaluation(
                student_model, student_config, 
                teacher_base_model, teacher_base_config,
                eval_dataloader, current_device, stitching_info
            )
            logger.info(f"Hidden State Avg MSE (Student vs Teacher): {hs_results['avg_mse']:.6f}")
            eval_summary["hidden_state_comparison"] = hs_results
            if hs_results["layer_mse"]:
                logger.info("Layer-wise MSEs (0=Embeddings):")
                for i, mse_val in enumerate(hs_results["layer_mse"]):
                    logger.info(f"  HS Output {i}: {mse_val:.6f}")
                plot_layer_mse(hs_results["layer_mse"], os.path.join(args.output_dir, f"layer_mse_{args.model_type}.png"), args.model_type)

    # Save overall summary
    summary_path = os.path.join(args.output_dir, f"evaluation_summary_{args.model_type}.txt")
    with open(summary_path, 'w') as f:
        import json
        f.write(json.dumps(eval_summary, indent=4, ensure_ascii=False))
    logger.info(f"Evaluation summary saved to {summary_path}")
    logger.info("Evaluation script finished.")

if __name__ == "__main__":
    main()

