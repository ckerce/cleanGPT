# ./tests/training_tests/test_train_factored_transformer.py
"""
Test script for trainers.train_factored_transformer.py
Runs a minimal training loop.
"""
import sys
import os
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def run_test_factored_transformer():
    """
    Configures and runs the train_factored_transformer.py script with test parameters.
    """
    print("Running test for train_factored_transformer.py...")
    # Note: The training script is in trainers/ not examples/
    script_path = os.path.join(PROJECT_ROOT, "trainers", "train_factored_transformer.py")
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "test_outputs", "factored_transformer_test")

    args = [
        "python", script_path,
        "--dataset", "roneneldan/TinyStories",
        "--max_samples", "128",
        "--num_epochs", "1",
        "--batch_size", "16",
        "--n_layer", "1",
        "--n_head", "1",
        "--n_embd", "32",
        "--block_size", "64",
        "--device", "cpu",
        "--output_dir", output_dir,
        "--skip_generation" # This script supports skipping generation
    ]

    print(f"Executing command: {' '.join(args)}")
    try:
        os.makedirs(output_dir, exist_ok=True)
        process = subprocess.run(args, capture_output=True, text=True, check=True)
        print("train_factored_transformer.py ran successfully.")
        print("STDOUT:")
        print(process.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running train_factored_transformer.py: {e}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        raise
    except FileNotFoundError:
        print(f"Error: The script {script_path} was not found.")
        raise

if __name__ == "__main__":
    run_test_factored_transformer()
    print("Test for train_factored_transformer.py completed.")


