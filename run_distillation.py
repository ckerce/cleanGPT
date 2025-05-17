# ./run_distillation.py
"""
Main script to run block-by-block distillation of transformer models.

python run_distillation.py \
    --teacher_model_name_or_path "gpt2" \
    --student_model_type "Factored" \
    --student_n_embd 384 \
    --student_n_head 6 \
    --dataset_name "roneneldan/TinyStories" \
    --dataset_text_column "story" \
    --block_size 128 \
    --batch_size 8 \
    --epochs_per_block 1 \
    --lr_per_block 5e-5 \
    --output_dir "./distilled_factored_gpt2_tinystories" \
    --max_samples 1000 \
    --device "cuda" 
    # Add --student_bias, --student_use_proj etc. for Factored model if needed

# Example for SASP (assuming it's a vanilla target)
python run_distillation.py \
    --teacher_model_name_or_path "gpt2" \
    --student_model_type "SASP" \
    --student_n_embd 768 \
    --student_n_head 12 \
    --dataset_name "wikitext" \
    --dataset_config_name "wikitext-2-raw-v1" \
    --block_size 256 \
    --batch_size 4 \
    --epochs_per_block 2 \
    --lr_per_block 3e-5 \
    --output_dir "./distilled_sasp_gpt2_wikitext" \
    --max_samples 5000 \
    --distill_loss_type "mse"

# Example for Vanilla model
python run_distillation.py \
    --teacher_model_name_or_path "gpt2" \
    --student_model_type "Vanilla" \
    --student_n_embd 384 \
    --student_n_head 6 \
    --dataset_name "wikitext" \
    --dataset_config_name "wikitext-2-raw-v1" \
    --block_size 128 \
    --batch_size 8 \
    --epochs_per_block 1 \
    --lr_per_block 5e-5 \
    --output_dir "./distilled_vanilla_gpt2_wikitext" \
    --max_samples 10000
"""
import argparse
import logging
import torch
import os
import sys
from tqdm.auto import tqdm  # Added tqdm import

# Add parent directory to sys.path to access custom modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import GPT2Model, GPT2Config as HF_GPT2Config, AutoTokenizer # Renamed GPT2Config to HF_GPT2Config to avoid clash
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset # For loading training data

# Import your custom modules
# Assuming your config file is now named config_distillation.py
from config_distillation import GPTConfig, print_config, DEVICE
# Removed create_config_from_args as we are directly instantiating GPTConfig

# Import the distillation versions of the models
from model.model_token_factored_distillation import FactoredTransformerModelDistillation
from model.model_SASPV_distillation import SASPTransformerModelDistillation
from model.model_vanilla_distillation import VanillaTransformerModelDistillation

from distillation_module import BlockDistillationTrainer

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Data Loading Utilities (Simplified Example) ---
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = []
        logger.info(f"Tokenizing {len(texts)} texts with block_size {block_size}...")
        for text in tqdm(texts, desc="Encoding texts"):
            if not text or not isinstance(text, str): # Ensure text is a non-empty string
                # logger.warning(f"Skipping invalid text entry: {text}")
                continue
            tokenized_output = self.tokenizer(
                text,
                truncation=True,
                max_length=self.block_size,
                padding=False,
                return_tensors=None,
                add_special_tokens=True
            )
            self.examples.append(tokenized_output.input_ids)
        logger.info(f"Created {len(self.examples)} examples from texts.")


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Return a dictionary to be consistent with collators expecting dicts
        return {"input_ids": self.examples[i]}


class DistillationDataCollator:
    def __init__(self, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        if self.tokenizer.pad_token_id is None:
            logger.warning("Tokenizer does not have a pad_token_id. Using eos_token_id for padding.")
            # Ensure pad_token is also set if only pad_token_id was missing
            if self.tokenizer.pad_token is None:
                 self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, examples: list[dict]):
        batch_input_ids = []

        for ex_dict in examples:
            ids = ex_dict['input_ids'] # ex_dict is now {'input_ids': [list_of_ids]}
            padding_length = self.block_size - len(ids)
            if padding_length < 0:
                ids = ids[:self.block_size]
                padding_length = 0

            padded_ids = ids + [self.pad_token_id] * padding_length
            batch_input_ids.append(padded_ids)

        input_ids_tensor = torch.tensor(batch_input_ids, dtype=torch.long)
        attention_mask = (input_ids_tensor != self.pad_token_id).long()

        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask,
            # "labels": input_ids_tensor.clone() # Only if student/teacher models internally require labels for forward pass
        }

def get_model_class(model_type_name: str):
    if model_type_name.lower() == "factored":
        return FactoredTransformerModelDistillation
    elif model_type_name.lower() == "sasp":
        return SASPTransformerModelDistillation
    elif model_type_name.lower() == "vanilla":
        return VanillaTransformerModelDistillation
    else:
        raise ValueError(f"Unsupported student model type: {model_type_name}")

def parse_args():
    parser = argparse.ArgumentParser(description="Block-by-block Transformer Distillation Script")

    # Teacher Model
    parser.add_argument("--teacher_model_name_or_path", type=str, default="gpt2",
                        help="Name or path of the Hugging Face teacher model (e.g., 'gpt2', 'gpt2-medium').")

    # Student Model
    parser.add_argument("--student_model_type", type=str, required=True, choices=["Factored", "SASP", "Vanilla"],
                        help="Type of student model architecture.")
    parser.add_argument("--student_n_embd", type=int, default=384, help="Student model embedding dimension.")
    parser.add_argument("--student_n_head", type=int, default=6, help="Student model number of attention heads.")
    parser.add_argument("--student_dropout", type=float, default=0.1, help="Student model dropout rate.")
    parser.add_argument("--student_bias", action="store_true", default=False, help="Use bias in student model linear layers.")

    # Factored specific args (if student_model_type is Factored)
    parser.add_argument("--student_use_proj", action="store_true", default=False, help="FactoredModel: Use projection in attention.")
    parser.add_argument("--student_use_v", action="store_true", default=False, help="FactoredModel: Use V vector in attention.")
    parser.add_argument("--student_use_channel_factor_v", action="store_true", default=False, help="FactoredModel: Use channel factor for V.")
    parser.add_argument("--student_use_channel_factor_proj", action="store_true", default=False, help="FactoredModel: Use channel factor for projection.")


    # Dataset
    parser.add_argument("--dataset_name", type=str, default="wikitext",
                        help="Name of the dataset from Hugging Face datasets library (e.g., 'wikitext', 'roneneldan/TinyStories').")
    parser.add_argument("--dataset_config_name", type=str, default="default",
                        help="Configuration name for the dataset (e.g., 'wikitext-2-raw-v1').")
    parser.add_argument("--dataset_text_column", type=str, default="text",
                        help="The name of the column in the dataset that contains the text.")
    parser.add_argument("--max_samples", type=int, default=10000,
                        help="Maximum number of samples to use from the dataset for training.")
    parser.add_argument("--block_size", type=int, default=128,
                        help="Block size for tokenization and model context window.")

    # Distillation Training
    parser.add_argument("--output_dir", type=str, default="./distilled_models_output",
                        help="Directory to save distilled models and logs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for distillation training.")
    parser.add_argument("--epochs_per_block", type=int, default=3,
                        help="Number of epochs to train each block.")
    parser.add_argument("--lr_per_block", type=float, default=5e-5,
                        help="Learning rate for distilling each block.")
    parser.add_argument("--weight_decay_per_block", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--max_grad_norm_per_block", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument("--distill_loss_type", type=str, default="mse", choices=["mse", "kl_div"],
                        help="Type of loss function for distillation.")
    parser.add_argument("--distill_loss_temperature", type=float, default=2.0,
                        help="Temperature for KL divergence loss (if used).")
    parser.add_argument("--freeze_previous_blocks", action="store_true", default=True,
                        help="Freeze parameters of previously distilled blocks during training of current block.")
    parser.add_argument("--no_freeze_previous_blocks", action="store_false", dest="freeze_previous_blocks")


    # Environment
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"],
                        help="Device to use (cpu, cuda, mps). Auto-detects if None.")
    parser.add_argument("--log_interval", type=int, default=50, help="Logging interval during training.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers.")

    return parser.parse_args()

def main():
    args = parse_args()

    # --- Setup Device ---
    if args.device:
        current_device = torch.device(args.device)
    else:
        current_device = DEVICE # From config_distillation.py, auto-detected
    logger.info(f"Using device: {current_device}")

    # --- Load Teacher Model and Tokenizer ---
    logger.info(f"Loading teacher model: {args.teacher_model_name_or_path}")
    try:
        # It's good practice to use AutoTokenizer for flexibility
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_name_or_path, use_fast=True)
        teacher_hf_config = HF_GPT2Config.from_pretrained(args.teacher_model_name_or_path) # Use renamed HF_GPT2Config
        teacher_model = GPT2Model.from_pretrained(args.teacher_model_name_or_path, config=teacher_hf_config)
        teacher_model.eval()
    except Exception as e:
        logger.error(f"Failed to load teacher model or tokenizer: {e}", exc_info=True)
        return

    if teacher_tokenizer.pad_token_id is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
        teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id
        logger.info(f"Set teacher tokenizer pad_token to eos_token (ID: {teacher_tokenizer.pad_token_id})")

    # --- Create Student Model ---
    logger.info(f"Creating student model of type: {args.student_model_type}")

    # Ensure n_layer matches for block-by-block distillation
    # Student's n_layer should be derived from teacher's config for compatibility
    student_n_layer = teacher_hf_config.n_layer
    logger.info(f"Student n_layer will be set to teacher's n_layer: {student_n_layer}")

    student_config_params = {
        'block_size': min(args.block_size, teacher_hf_config.n_positions),
        'vocab_size': teacher_hf_config.vocab_size,
        'n_layer': student_n_layer, # Match teacher for block-by-block
        'padding_idx': teacher_tokenizer.pad_token_id,
        'n_embd': args.student_n_embd,
        'n_head': args.student_n_head,
        'dropout': args.student_dropout,
        'bias': args.student_bias,
        'model_type': args.student_model_type,
        'output_hidden_states': True, # Crucial for distillation
        'teacher_n_embd': teacher_hf_config.n_embd # Store teacher's embedding dim for potential projection
    }

    # Add Factored-specific parameters only if the model type is Factored
    if args.student_model_type.lower() == "factored":
        student_config_params['use_proj'] = args.student_use_proj
        student_config_params['use_v'] = args.student_use_v
        student_config_params['use_channel_factor_v'] = args.student_use_channel_factor_v
        student_config_params['use_channel_factor_proj'] = args.student_use_channel_factor_proj

    try:
        student_config = GPTConfig(**student_config_params)
    except TypeError as e:
        logger.error(f"TypeError during GPTConfig initialization for student model: {e}")
        logger.error(f"Parameters passed: {student_config_params}")
        logger.error("Please ensure your `config_distillation.py`'s GPTConfig class defines all expected fields, "
                     "especially 'use_proj', 'use_v', 'use_channel_factor_v', 'use_channel_factor_proj' if student is Factored.")
        return


    if args.distill_loss_type == "mse" and student_config.n_embd != teacher_hf_config.n_embd:
        logger.warning(
            f"Student n_embd ({student_config.n_embd}) and Teacher n_embd ({teacher_hf_config.n_embd}) "
            f"differ. MSE loss on hidden states will likely require a projection layer. "
            f"A hidden state projection will be added automatically."
        )

    StudentModelClass = get_model_class(args.student_model_type)
    student_model = StudentModelClass(config=student_config)
    student_model.to(current_device)
    logger.info(f"Student model '{args.student_model_type}' created with {student_model.get_num_params()/1e6:.2f}M params.")
    # Assuming print_config is defined in config_distillation.py
    print_config(student_config, dataset_name=args.dataset_name, dataset_config=args.dataset_config_name)


    # --- Load Dataset and Create DataLoader ---
    logger.info(f"Loading dataset: {args.dataset_name} ({args.dataset_config_name or 'default config'})")
    try:
        dataset_args = [args.dataset_name]
        if args.dataset_config_name:
            dataset_args.append(args.dataset_config_name)

        raw_dataset = load_dataset(*dataset_args, split=f"train[:{args.max_samples}]", trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}", exc_info=True)
        return

    if args.dataset_text_column not in raw_dataset.column_names:
        logger.warning(f"Text column '{args.dataset_text_column}' not found in dataset. Available columns: {raw_dataset.column_names}")
        potential_text_cols = [col for col in raw_dataset.column_names if raw_dataset.features[col].dtype == 'string']
        if potential_text_cols:
            args.dataset_text_column = potential_text_cols[0]
            logger.info(f"Using automatically selected text column: '{args.dataset_text_column}'")
        else:
            logger.error("No suitable text column found. Exiting.")
            return

    texts = raw_dataset[args.dataset_text_column]
    # Filter out any non-string or empty entries before passing to TextDataset
    processed_texts = [text for text in texts if isinstance(text, str) and text.strip()]
    if len(processed_texts) != len(texts):
        logger.warning(f"Filtered out {len(texts) - len(processed_texts)} invalid text entries from the dataset.")

    train_dataset = TextDataset(processed_texts, teacher_tokenizer, student_config.block_size)
    if len(train_dataset) == 0:
        logger.error("No valid samples found in the dataset after processing. Exiting.")
        return

    data_collator = DistillationDataCollator(teacher_tokenizer, student_config.block_size)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if current_device.type == 'cuda' else False # pin_memory for CUDA
    )
    logger.info(f"DataLoader created with {len(train_dataloader)} batches.")

    # --- Initialize Distillation Trainer ---
    logger.info("Initializing BlockDistillationTrainer...")
    distill_trainer = BlockDistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        tokenizer=teacher_tokenizer,
        train_dataloader=train_dataloader,
        distill_loss_type=args.distill_loss_type,
        distill_loss_temperature=args.distill_loss_temperature,
        optimizer_cls=torch.optim.AdamW,
        device=current_device,
        output_dir=args.output_dir,
        log_interval=args.log_interval,
        freeze_previous_blocks=args.freeze_previous_blocks
    )

    # --- Run Distillation ---
    logger.info("Starting distillation process...")
    try:
        distill_trainer.train(
            epochs_per_block=args.epochs_per_block,
            lr_per_block=args.lr_per_block,
            wd_per_block=args.weight_decay_per_block,
            max_grad_norm_per_block=args.max_grad_norm_per_block
        )
    except Exception as e:
        logger.error(f"An error occurred during distillation training: {e}", exc_info=True)
        # Optionally save a checkpoint on error
        error_checkpoint_path = os.path.join(args.output_dir, "student_model_error_checkpoint.pt")
        if hasattr(distill_trainer, 'save_checkpoint'):
            distill_trainer.save_checkpoint(error_checkpoint_path, error=str(e))
            logger.info(f"Saved error checkpoint to {error_checkpoint_path}")


    logger.info("Distillation script finished.")

if __name__ == "__main__":
    main()
