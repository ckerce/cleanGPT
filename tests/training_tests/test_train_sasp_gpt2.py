# ./tests/training_tests/test_train_sasp_gpt2.py
"""
Test script for examples.train_sasp_gpt2.py
Runs a minimal training loop.
"""
import sys
import os
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def run_test_sasp_gpt2():
    """
    Configures and runs the train_sasp_gpt2.py script with test parameters.
    """
    print("Running test for train_sasp_gpt2.py...")
    script_path = os.path.join(PROJECT_ROOT, "examples", "train_sasp_gpt2.py")
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "test_outputs", "sasp_gpt2_test")

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
        "--output_dir", output_dir,
        # train_sasp_gpt2.py does not have --skip_generation
        # Its generation is embedded and uses fixed new tokens (100)
    ]

    print(f"Executing command: {' '.join(args)}")
    try:
        os.makedirs(output_dir, exist_ok=True)
        process = subprocess.run(args, capture_output=True, text=True, check=True)
        print("train_sasp_gpt2.py ran successfully.")
        print("STDOUT:")
        print(process.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running train_sasp_gpt2.py: {e}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        raise
    except FileNotFoundError:
        print(f"Error: The script {script_path} was not found.")
        raise

if __name__ == "__main__":
    run_test_sasp_gpt2()
    print("Test for train_sasp_gpt2.py completed.")

