# ./tests/training_tests/test_train_sasp_char.py
"""
Test script for examples.train_sasp_char.py
Runs a minimal training loop to ensure the script executes without errors.
"""
import sys
import os
import subprocess

# Add project root to sys.path to allow importing modules from the project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def run_test_sasp_char():
    """
    Configures and runs the train_sasp_char.py script with test parameters.
    """
    print("Running test for train_sasp_char.py...")
    
    # Define the command and arguments for the training script
    # Ensure the path to the script is correct relative to the project root
    script_path = os.path.join(PROJECT_ROOT, "examples", "train_sasp_char.py")
    output_dir = os.path.join(PROJECT_ROOT, "outputs", "test_outputs", "sasp_char_test")
    
    # Base arguments for a quick test run
    args = [
        "python", script_path,
        "--dataset", "roneneldan/TinyStories", # Using a small, accessible dataset
        "--max_samples", "128",
        "--num_epochs", "1",
        "--batch_size", "16",
        "--n_layer", "2", # Minimal model for speed
        "--n_head", "2",  # Minimal model for speed
        "--n_embd", "32", # Minimal model for speed
        "--block_size", "64",
        "--output_dir", output_dir,
        "--save_tokenizer" # Test this functionality as well
        # train_sasp_char.py does not have --skip_generation
        # Its generation is embedded and uses fixed new tokens (100)
    ]
    
    print(f"Executing command: {' '.join(args)}")
    
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Run the training script as a subprocess
        process = subprocess.run(args, capture_output=True, text=True, check=True)
        print("train_sasp_char.py ran successfully.")
        print("STDOUT:")
        print(process.stdout)
        if process.stderr:
            print("STDERR:") # Should be empty on success if check=True
            print(process.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"Error running train_sasp_char.py: {e}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        raise # Re-raise the exception to mark the test as failed
    except FileNotFoundError:
        print(f"Error: The script {script_path} was not found. Make sure the path is correct.")
        raise

if __name__ == "__main__":
    run_test_sasp_char()
    print("Test for train_sasp_char.py completed.")


