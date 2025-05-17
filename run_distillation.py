# ./run_distillation.py
"""
Main script to run block-by-block distillation of transformer models.
"""
import argparse
import logging
import torch
import os
import sys
from tqdm.auto import tqdm 
from typing import Optional, Dict, Any, List, Union

# --- Path Setup ---
# Add parent directory to sys.path to attempt to access custom modules
# This assumes run_distillation.py might be in a subdirectory like 'scripts/'
# and custom modules (config, models) are in the parent directory or a 'model' subdirectory.
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_path, '..')) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    # Also add the current script's directory in case modules are alongside it
if current_script_path not in sys.path:
     sys.path.insert(0, current_script_path)


# --- Imports ---
# Attempt to import custom modules first
try:
    from config_distillation import GPTConfig, print_config, DEVICE
    # Assuming model definitions are in a 'model' subdirectory relative to project_root
    from model.model_token_factored_distillation import FactoredTransformerModelDistillation
    from model.model_SASPV_distillation import SASPTransformerModelDistillation
    from model.model_vanilla_distillation import VanillaTransformerModelDistillation
    from distillation_module import BlockDistillationTrainer
except ImportError as e:
    print(f"Initial error importing custom modules: {e}")
    print("Please ensure 'config_distillation.py' and model definitions (e.g., in a 'model' folder) "
          "are in the Python path or structured correctly relative to this script's parent directory.")
    # Exit if critical modules are missing, as the script cannot function.
    sys.exit(1)


from transformers import GPT2Model, GPT2Config as HF_GPT2Config, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Data Loading Utilities ---
class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []
        logger.info(f"Tokenizing {len(texts)} texts with block_size {block_size}...")
        for text in tqdm(texts, desc="Encoding texts"):
            if not text or not isinstance(text, str): 
                # logger.warning(f"Skipping invalid text entry: {text}")
                continue
            # Tokenize the text. Padding is handled by the collator.
            tokenized_output = self.tokenizer(
                text,
                truncation=True,
                max_length=self.block_size, 
                padding=False, 
                return_tensors=None, # Return list of IDs
                add_special_tokens=True # Add BOS/EOS if configured in tokenizer
            )
            self.examples.append(tokenized_output.input_ids)
        logger.info(f"Created {len(self.examples)} examples from texts.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, List[int]]:
        return {"input_ids": self.examples[i]}


class DistillationDataCollator:
    def __init__(self, tokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        # Ensure pad token is set
        if self.tokenizer.pad_token_id is None:
            logger.warning("Tokenizer does not have a pad_token_id. Using eos_token_id for padding.")
            self.tokenizer.pad_token = self.tokenizer.eos_token # Set pad_token string
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id # Set pad_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = []
        for ex_dict in examples:
            ids = ex_dict['input_ids']
            # Pad or truncate each example to block_size
            padding_length = self.block_size - len(ids)
            if padding_length > 0:
                padded_ids = ids + [self.pad_token_id] * padding_length
            elif padding_length < 0: # Should be handled by TextDataset's truncation
                padded_ids = ids[:self.block_size]
            else:
                padded_ids = ids
            batch_input_ids.append(padded_ids)

        input_ids_tensor = torch.tensor(batch_input_ids, dtype=torch.long)
        # Create attention_mask: 1 for real tokens, 0 for padding tokens
        attention_mask = (input_ids_tensor != self.pad_token_id).long()
        return {"input_ids": input_ids_tensor, "attention_mask": attention_mask}

def get_model_class(model_type_name: str):
    """Helper function to get the student model class based on its type name."""
    model_type_name_lower = model_type_name.lower()
    if model_type_name_lower == "factored":
        return FactoredTransformerModelDistillation
    elif model_type_name_lower == "sasp":
        return SASPTransformerModelDistillation
    elif model_type_name_lower == "vanilla":
        return VanillaTransformerModelDistillation
    else:
        raise ValueError(f"Unsupported student model type: {model_type_name}. Supported: Factored, SASP, Vanilla.")

def parse_args():
    parser = argparse.ArgumentParser(description="Block-by-block Transformer Distillation Script")

    # --- Teacher Model Arguments ---
    parser.add_argument("--teacher_model_name_or_path", type=str, default="gpt2",
                        help="Name or path of the Hugging Face teacher model (e.g., 'gpt2', 'gpt2-medium').")

    # --- Student Model Arguments ---
    parser.add_argument("--student_model_type", type=str, required=True, choices=["Factored", "SASP", "Vanilla"],
                        help="Type of student model architecture.")
    parser.add_argument("--student_n_embd", type=int, default=384, help="Student model embedding dimension.")
    parser.add_argument("--student_n_head", type=int, default=6, help="Student model number of attention heads.")
    parser.add_argument("--student_dropout", type=float, default=0.1, help="Student model dropout rate.")
    parser.add_argument("--student_bias", action="store_true", default=False, 
                        help="Use bias in student model linear layers (activates if flag is present).")

    # --- Factored Model Specific Arguments (conditionally used) ---
    parser.add_argument("--student_use_proj", action="store_true", default=False, help="FactoredModel: Use projection in attention.")
    parser.add_argument("--student_use_v", action="store_true", default=False, help="FactoredModel: Use V vector in attention.")
    parser.add_argument("--student_use_channel_factor_v", action="store_true", default=False, help="FactoredModel: Use channel factor for V.")
    parser.add_argument("--student_use_channel_factor_proj", action="store_true", default=False, help="FactoredModel: Use channel factor for projection.")

    # --- Dataset Arguments ---
    parser.add_argument("--dataset_name", type=str, default="wikitext",
                        help="Name of the dataset from Hugging Face datasets library (e.g., 'wikitext', 'roneneldan/TinyStories').")
    parser.add_argument("--dataset_config_name", type=str, default=None,
                        help="Configuration name for the dataset (e.g., 'wikitext-2-raw-v1'). If None, uses dataset default.")
    parser.add_argument("--dataset_text_column", type=str, default="text",
                        help="The name of the column in the dataset that contains the text.")
    parser.add_argument("--max_samples", type=int, default=10000,
                        help="Maximum number of samples to use from the dataset for training. 0 for all.")
    parser.add_argument("--block_size", type=int, default=128,
                        help="Block size for tokenization and model context window.")

    # --- Distillation Training Arguments ---
    parser.add_argument("--output_dir", type=str, default="./distilled_models_output",
                        help="Directory to save distilled models and logs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for distillation training.")
    parser.add_argument("--epochs_per_block", type=int, default=3,
                        help="Number of epochs to train each block.")
    parser.add_argument("--lr_per_block", type=float, default=5e-5,
                        help="Learning rate for distilling each block. Can be a single float or list (not supported via CLI yet).")
    parser.add_argument("--weight_decay_per_block", type=float, default=0.01, help="Weight decay per block.")
    parser.add_argument("--max_grad_norm_per_block", type=float, default=1.0, help="Max gradient norm per block.")
    parser.add_argument("--distill_loss_type", type=str, default="mse", choices=["mse", "kl_div"],
                        help="Type of loss function for distillation.")
    parser.add_argument("--distill_loss_temperature", type=float, default=2.0,
                        help="Temperature for KL divergence loss (if used).")
    parser.add_argument("--freeze_previous_blocks", action="store_true", default=True,
                        help="Freeze parameters of previously distilled blocks during training of current block.")
    parser.add_argument("--no_freeze_previous_blocks", action="store_false", dest="freeze_previous_blocks",
                        help="Do not freeze previously distilled blocks.")

    # --- Stitching Layer Arguments ---
    parser.add_argument("--use_stitching_layers", action="store_true", default=True,
                        help="Use trainable linear stitching layers. Default is True if not specified otherwise.")
    parser.add_argument("--no_stitching_layers", action="store_false", dest="use_stitching_layers",
                        help="Disable stitching layers and use direct comparison of hidden states.")
    parser.add_argument("--stitching_layer_bias", action="store_true", default=True,
                        help="Include bias term in stitching layer projections. Default is True.")
    parser.add_argument("--no_stitching_layer_bias", action="store_false", dest="stitching_layer_bias",
                        help="Disable bias term in stitching layer projections.")

    # --- Environment Arguments ---
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"],
                        help="Device to use (cpu, cuda, mps). Auto-detects if None, using DEVICE from config_distillation.")
    parser.add_argument("--log_interval", type=int, default=50, help="Logging interval during training (batches).")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers.")

    return parser.parse_args()

def main():
    args = parse_args()

    # --- Setup Device ---
    if args.device: # User specified device via CLI
        current_device = torch.device(args.device)
        logger.info(f"Using device specified via CLI: {current_device}")
    else: # Use DEVICE from config_distillation.py (which auto-detects)
        current_device = DEVICE 
        logger.info(f"Using device from config_distillation.py: {current_device} ({DEVICE_NAME})")


    # --- Load Teacher Model and Tokenizer ---
    logger.info(f"Loading teacher model: {args.teacher_model_name_or_path}")
    try:
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_name_or_path, use_fast=True)
        teacher_hf_config = HF_GPT2Config.from_pretrained(args.teacher_model_name_or_path)
        teacher_model = GPT2Model.from_pretrained(args.teacher_model_name_or_path, config=teacher_hf_config)
        teacher_model.eval().to(current_device) # Ensure teacher is in eval mode and on correct device
    except Exception as e:
        logger.error(f"Failed to load teacher model or tokenizer ('{args.teacher_model_name_or_path}'): {e}", exc_info=True)
        return # Critical error, cannot proceed

    # Ensure teacher tokenizer has a pad token (use EOS if not present)
    if teacher_tokenizer.pad_token_id is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
        teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id
        logger.info(f"Set teacher tokenizer pad_token to eos_token (ID: {teacher_tokenizer.pad_token_id})")

    # --- Create Student Model ---
    logger.info(f"Creating student model of type: {args.student_model_type}")

    # Student's n_layer should match teacher's for block-by-block distillation
    student_n_layer = teacher_hf_config.n_layer
    logger.info(f"Student n_layer will be set to teacher's n_layer: {student_n_layer}")

    # Prepare student configuration parameters
    student_config_params = {
        'block_size': min(args.block_size, teacher_hf_config.n_positions), # Ensure block_size <= teacher's max
        'vocab_size': teacher_hf_config.vocab_size, # Match teacher's vocab
        'n_layer': student_n_layer, 
        'padding_idx': teacher_tokenizer.pad_token_id, # Use teacher's padding token ID
        'n_embd': args.student_n_embd,
        'n_head': args.student_n_head,
        'dropout': args.student_dropout,
        'bias': args.student_bias,
        'model_type': args.student_model_type,
        'output_hidden_states': True, # Crucial for distillation to get hidden states
        'teacher_n_embd': teacher_hf_config.n_embd # Store teacher's embedding dim for reference/projection
    }

    # Add Factored-specific parameters only if the model type is Factored
    if args.student_model_type.lower() == "factored":
        student_config_params.update({
            'use_proj': args.student_use_proj,
            'use_v': args.student_use_v,
            'use_channel_factor_v': args.student_use_channel_factor_v,
            'use_channel_factor_proj': args.student_use_channel_factor_proj,
        })
    
    try:
        student_config = GPTConfig(**student_config_params)
    except Exception as e: # Catch potential errors during GPTConfig initialization
        logger.error(f"Error initializing student GPTConfig: {e}", exc_info=True)
        logger.error(f"Parameters passed to GPTConfig: {student_config_params}")
        return
        
    # Instantiate the student model
    StudentModelClass = get_model_class(args.student_model_type)
    student_model = StudentModelClass(config=student_config)
    student_model.to(current_device) # Move student model to the correct device
    logger.info(f"Student model '{args.student_model_type}' created with {student_model.get_num_params()/1e6:.2f}M params.")
    
    # Print configuration details (if print_config is available)
    if 'print_config' in globals() and callable(print_config):
        print_config(cfg=student_config, dataset_name=args.dataset_name, dataset_config=args.dataset_config_name)


    # --- Load Dataset and Create DataLoader ---
    logger.info(f"Loading dataset: {args.dataset_name} (Config: {args.dataset_config_name or 'default'})")
    try:
        dataset_load_args = [args.dataset_name]
        if args.dataset_config_name and args.dataset_config_name.lower() != "default":
            dataset_load_args.append(args.dataset_config_name)
        
        split_str = f"train[:{args.max_samples}]" if args.max_samples > 0 else "train"
        raw_dataset = load_dataset(*dataset_load_args, split=split_str, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load dataset '{args.dataset_name}': {e}", exc_info=True)
        return

    # Verify and select the text column
    if args.dataset_text_column not in raw_dataset.column_names:
        logger.warning(f"Text column '{args.dataset_text_column}' not found in dataset. Available columns: {raw_dataset.column_names}")
        # Attempt to find a suitable text column (e.g., the first string column)
        potential_text_cols = [col for col in raw_dataset.column_names if raw_dataset.features[col].dtype == 'string']
        if potential_text_cols:
            args.dataset_text_column = potential_text_cols[0]
            logger.info(f"Using automatically selected text column: '{args.dataset_text_column}'")
        else:
            logger.error("No suitable string text column found in the dataset. Exiting.")
            return
            
    texts = raw_dataset[args.dataset_text_column]
    # Filter out any non-string or empty entries before passing to TextDataset
    processed_texts = [text for text in texts if isinstance(text, str) and text.strip()]
    if len(processed_texts) != len(texts):
        logger.warning(f"Filtered out {len(texts) - len(processed_texts)} invalid/empty text entries from the dataset.")

    if not processed_texts:
        logger.error("No valid text samples found after processing. Exiting.")
        return
        
    train_dataset = TextDataset(processed_texts, teacher_tokenizer, student_config.block_size)
    if len(train_dataset) == 0: # Should be redundant if processed_texts is checked
        logger.error("TextDataset created 0 examples. Exiting.")
        return

    data_collator = DistillationDataCollator(teacher_tokenizer, student_config.block_size)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        collate_fn=data_collator,
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=(current_device.type == 'cuda') # pin_memory for CUDA acceleration
    )
    logger.info(f"DataLoader created with {len(train_dataloader)} batches.")

    # --- Initialize Distillation Trainer ---
    logger.info("Initializing BlockDistillationTrainer...")
    try:
        distill_trainer = BlockDistillationTrainer(
            teacher_model=teacher_model,
            student_model=student_model,
            tokenizer=teacher_tokenizer, # Pass tokenizer for potential internal use
            train_dataloader=train_dataloader,
            distill_loss_type=args.distill_loss_type,
            distill_loss_temperature=args.distill_loss_temperature,
            use_stitching_layers=args.use_stitching_layers,      
            stitching_layer_bias=args.stitching_layer_bias,    
            optimizer_cls=torch.optim.AdamW, # Optimizer class
            device=current_device,
            output_dir=args.output_dir,
            log_interval=args.log_interval,
            freeze_previous_blocks=args.freeze_previous_blocks
        )
    except Exception as e:
        logger.error(f"Error initializing BlockDistillationTrainer: {e}", exc_info=True)
        return

    # --- Run Distillation ---
    logger.info("Starting distillation process...")
    try:
        distill_trainer.train(
            epochs_per_block=args.epochs_per_block,
            lr_per_block=args.lr_per_block, # This is a float from argparse
            wd_per_block=args.weight_decay_per_block, # This is a float
            max_grad_norm_per_block=args.max_grad_norm_per_block # This is a float
        )
    except Exception as e:
        logger.error(f"An error occurred during distillation training: {e}", exc_info=True)
        # Optionally save a checkpoint on error
        error_checkpoint_path = os.path.join(args.output_dir, "student_model_error_checkpoint.pt")
        if hasattr(distill_trainer, 'save_checkpoint'): # Check if trainer and method exist
            distill_trainer.save_checkpoint(error_checkpoint_path, error=str(e), status="error_during_training")
            logger.info(f"Saved error checkpoint to {error_checkpoint_path}")

    logger.info("Distillation script finished.")

if __name__ == "__main__":
    main()

