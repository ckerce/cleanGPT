# simple_qualitative_evaluation.py
"""
A utility to load a model from a checkpoint saved by the original
SimpleTrainer and run qualitative inference on several prompts.

This script requires manual specification of model and tokenizer
configurations via command-line arguments if they differ from the
defaults, as this information is not present in the older, simpler
checkpoint files.
"""
import argparse
import logging
import torch
import os
import sys

# Ensure the script can find modules from the repository root
# If this script is in a subdirectory, adjust the path accordingly.
# If run from the repo root, this might not be strictly necessary
# but doesn't hurt.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) # Adjust '..' if script is nested deeper

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from config import GPTConfig
    from model import get_model
    from mytokenizers import create_tokenizer
    from inference.generation_exp import run_generation, get_generation_args
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print(f"Ensure that the script is placed correctly in your project structure,")
    print(f"or that the Python path is set up to find 'config', 'model', 'mytokenizers', and 'inference'.")
    print(f"Attempted to add REPO_ROOT: {REPO_ROOT} to sys.path.")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load a model from an original SimpleTrainer checkpoint and run inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Checkpoint ---
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the .pt checkpoint file saved by the original SimpleTrainer.")

    # --- Model Configuration (defaults from the user's last sasp_gpt2 run) ---
    parser.add_argument("--model_type", type=str, default="SASP", choices=["SASP", "Vanilla", "Factored"],
                        help="Type of the model architecture.")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of transformer layers.")
    parser.add_argument("--n_head", type=int, default=6, help="Number of attention heads.")
    parser.add_argument("--n_embd", type=int, default=288, help="Embedding dimension.")
    parser.add_argument("--block_size", type=int, default=512, help="Context window size (max sequence length).")
    parser.add_argument("--bias", action='store_true', help="Use bias in Linear layers and LayerNorm. Set if used during training.")
    parser.add_argument("--no_bias", action='store_false', dest='bias', help="Do not use bias. Set if not used during training.")
    parser.set_defaults(bias=False) # Default to False as in original GPTConfig for SASP
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate (set to 0.0 for evaluation).")

    # SASP Specific Configs (defaults from the user's last sasp_gpt2 run)
    sasp_group = parser.add_argument_group('SASP Model Configuration (if model_type=SASP)')
    sasp_group.add_argument("--use_proj", action='store_true', help="SASP: Use projection in CausalShapedAttention.")
    sasp_group.add_argument("--no_use_proj", action='store_false', dest='use_proj', help="SASP: Do not use projection.")
    sasp_group.set_defaults(use_proj=True) # Defaulting to True as per typical SASP training
    sasp_group.add_argument("--use_v", action='store_true', help="SASP: Use Value vector in CausalShapedAttention.")
    sasp_group.add_argument("--no_use_v", action='store_false', dest='use_v', help="SASP: Do not use Value vector.")
    sasp_group.set_defaults(use_v=True) # Defaulting to True as per typical SASP training
    sasp_group.add_argument("--llama_mlp", action='store_true', help="SASP: Use LLaMA-style MLP.")
    sasp_group.add_argument("--no_llama_mlp", action='store_false', dest='llama_mlp', help="SASP: Do not use LLaMA-style MLP.")
    sasp_group.set_defaults(llama_mlp=False) # Assuming False if not specified in train command
    sasp_group.add_argument("--transformer_block_type", type=str, default='SASP', choices=['SASP', 'PreLN'],
                        help="SASP: Transformer block type (e.g., 'SASP').")


    # --- Tokenizer Configuration ---
    parser.add_argument("--tokenizer_type", type=str, default="gpt2", choices=["character", "gpt2"],
                        help="Type of tokenizer to use.")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to a pretrained tokenizer directory (especially for 'character' type if vocab was saved). "
                             "For 'gpt2', this is usually not needed unless custom vocab or it was saved alongside the model.")
    # For character tokenizer, if vocab was not saved, it might be hard to reconstruct perfectly.
    # This script assumes if tokenizer_path is given for 'character', it's a saved vocab.

    # --- Device ---
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"],
                        help="Device to run inference on.")

    # --- Inference ---
    parser.add_argument("--prompts", type=str, nargs='+',
                        default=["Once upon a time in a land far away", "The meaning of life is"],
                        help="List of prompts to generate text from.")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum new tokens to generate per prompt.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter (e.g., 50). Default is None (no top-k).")

    return parser.parse_args()

def main():
    """Main function to load model and run inference."""
    args = parse_arguments()

    logger.info(f"Starting qualitative evaluation with arguments: {args}")

    # --- 1. Setup Device ---
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA specified but not available. Falling back to CPU.")
        device = torch.device("cpu")
    elif args.device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS specified but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # --- 2. Instantiate Tokenizer ---
    try:
        # For GPT-2, tokenizer_path is typically not needed if using the standard 'gpt2' vocab.
        # However, if the training script saved the tokenizer (e.g., if it added special tokens or used a specific version),
        # providing the path would be more robust.
        # The default for output_dir in train_sasp_gpt2.py was "./outputs/sasp_gpt2_large"
        # A tokenizer might be saved there if --save_tokenizer was an option (it's not in train_sasp_gpt2.py but is in char version)
        # For now, we assume standard 'gpt2' if no path is given.
        if args.tokenizer_path:
            logger.info(f"Loading tokenizer '{args.tokenizer_type}' from path: {args.tokenizer_path}")
            tokenizer = create_tokenizer(args.tokenizer_type, from_pretrained=args.tokenizer_path)
        else:
            logger.info(f"Creating tokenizer '{args.tokenizer_type}' (no path specified, using defaults).")
            tokenizer_params = {}
            if args.tokenizer_type == "gpt2":
                 tokenizer_params['use_fast'] = True # Default for GPT2Tokenizer
            tokenizer = create_tokenizer(args.tokenizer_type, **tokenizer_params)

        if args.tokenizer_type == "character" and not args.tokenizer_path:
            logger.warning("Character tokenizer created without a specific vocabulary path. "
                           "Ensure this matches the training setup for meaningful results, "
                           "or provide a --tokenizer_path to a saved character vocab.")
    except Exception as e:
        logger.error(f"Failed to create/load tokenizer: {e}", exc_info=True)
        sys.exit(1)
    logger.info(f"Tokenizer '{args.tokenizer_type}' initialized. Vocab size: {tokenizer.vocab_size}")


    # --- 3. Instantiate GPTConfig ---
    logger.info("Constructing GPTConfig...")
    config_params = {
        "block_size": args.block_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "dropout": args.dropout,
        "bias": args.bias,
        "model_type": args.model_type,
    }
    if args.model_type == "SASP":
        config_params.update({
            "use_proj": args.use_proj,
            "use_v": args.use_v,
            "llama_mlp": args.llama_mlp,
            "transformer_block_type": args.transformer_block_type,
        })

    try:
        model_config = GPTConfig(**config_params)
        model_config.update_from_tokenizer(tokenizer)
    except Exception as e:
        logger.error(f"Failed to create GPTConfig: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"GPTConfig created: {model_config}")

    # --- 4. Instantiate Model ---
    logger.info(f"Instantiating model of type: {model_config.model_type}...")
    try:
        model = get_model(model_config.model_type, config=model_config)
    except Exception as e:
        logger.error(f"Failed to instantiate model: {e}", exc_info=True)
        sys.exit(1)
    logger.info(f"Model instantiated. Number of parameters: {model.get_num_params()/1e6:.2f}M")


    # --- 5. Load Checkpoint and State ---
    logger.info(f"Loading checkpoint from: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        logger.error(f"Checkpoint file not found: {args.checkpoint_path}")
        sys.exit(1)

    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)

        if 'model_state_dict' not in checkpoint:
            logger.error("Checkpoint does not contain 'model_state_dict'. "
                         "This script is for original SimpleTrainer checkpoints.")
            sys.exit(1)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        logger.info("Model state_dict loaded successfully and model set to evaluation mode.")

    except Exception as e:
        logger.error(f"Failed to load model state_dict from checkpoint: {e}", exc_info=True)
        sys.exit(1)

    # --- 6. Run Inference ---
    logger.info(f"\n--- Starting Inference for {len(args.prompts)} Prompts ---")

    generation_params = get_generation_args()
    generation_params['max_new_tokens'] = args.max_new_tokens
    generation_params['temperature'] = args.temperature
    generation_params['top_k'] = args.top_k
    generation_params['show_progress'] = False

    for i, prompt_text in enumerate(args.prompts):
        logger.info(f"\nPrompt {i+1}/{len(args.prompts)}: \"{prompt_text}\"")
        try:
            _, generated_text = run_generation(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                device=device,
                **generation_params
            )
            print(f"--- Generated Text for Prompt {i+1} ---")
            print(generated_text)
            print("------------------------------------")

        except Exception as e:
            logger.error(f"Error during generation for prompt '{prompt_text}': {e}", exc_info=True)

    logger.info("\n--- Qualitative Evaluation Finished ---")

if __name__ == "__main__":
    main()

