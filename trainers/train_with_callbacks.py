# ./trainers/callbacks.py
"""
Custom callback implementations for the training loop.
"""
import os
import logging
import torch # For InferenceCallback
from typing import List, Dict, Optional, Any # For InferenceCallback

# Assuming Callback is in base_trainer or accessible via from .base_trainer import Callback
from .base_trainer import Callback
# For InferenceCallback, assuming your generation functions are structured like this:
from inference.generation import run_generation
from mytokenizers import BaseTokenizer


logger = logging.getLogger(__name__)

class TestCallback(Callback):
    """
    A simple callback for testing the callback system.
    It logs messages and counts calls to its methods.
    """
    def __init__(self):
        super().__init__()
        self.call_counts = {
            'on_train_begin': 0,
            'on_train_end': 0,
            'on_epoch_begin': 0,
            'on_epoch_end': 0,
            'on_batch_begin': 0,
            'on_batch_end': 0,
            'on_evaluate_begin': 0,
            'on_evaluate_end': 0,
        }
        self.epochs_ended = []
        self.batch_losses = []
        logger.info("TestCallback initialized.")

    def on_train_begin(self, logs=None):
        self.call_counts['on_train_begin'] += 1
        logger.info(f"TestCallback: on_train_begin called. Logs: {logs}")
        assert self.model is not None, "Model not set in TestCallback on_train_begin"
        assert self.optimizer is not None, "Optimizer not set in TestCallback on_train_begin"

    def on_train_end(self, logs=None):
        self.call_counts['on_train_end'] += 1
        logger.info(f"TestCallback: on_train_end called. Logs: {logs}")

    def on_epoch_begin(self, epoch, logs=None):
        self.call_counts['on_epoch_begin'] += 1
        logger.info(f"TestCallback: on_epoch_begin called for epoch {epoch}. Logs: {logs}")
        assert self.current_epoch == epoch, "Epoch number mismatch in TestCallback"

    def on_epoch_end(self, epoch, logs=None):
        self.call_counts['on_epoch_end'] += 1
        self.epochs_ended.append(epoch)
        logger.info(f"TestCallback: on_epoch_end called for epoch {epoch}. Logs: {logs}")
        if logs:
            assert 'loss' in logs, "Loss not in logs for on_epoch_end"

    def on_batch_begin(self, batch_idx, logs=None):
        self.call_counts['on_batch_begin'] += 1
        # logger.debug(f"TestCallback: on_batch_begin called for batch {batch_idx}. Logs: {logs}") # Too verbose for info

    def on_batch_end(self, batch_idx, logs=None):
        self.call_counts['on_batch_end'] += 1
        if logs and 'loss' in logs and logs['loss'] is not None:
            self.batch_losses.append(logs['loss'])
        # logger.debug(f"TestCallback: on_batch_end called for batch {batch_idx}. Logs: {logs}") # Too verbose for info

    def on_evaluate_begin(self, logs=None):
        self.call_counts['on_evaluate_begin'] += 1
        logger.info(f"TestCallback: on_evaluate_begin called. Logs: {logs}")

    def on_evaluate_end(self, logs=None):
        self.call_counts['on_evaluate_end'] += 1
        logger.info(f"TestCallback: on_evaluate_end called. Logs: {logs}")
        if logs:
            assert 'loss' in logs, "Loss not in logs for on_evaluate_end"

    def get_report(self):
        report = "TestCallback Report:\n"
        for event, count in self.call_counts.items():
            report += f"  {event}: called {count} times\n"
        report += f"  Epochs ended: {self.epochs_ended}\n"
        report += f"  Number of batch losses recorded: {len(self.batch_losses)}\n"
        if self.batch_losses:
            report += f"  Avg batch loss: {sum(self.batch_losses)/len(self.batch_losses):.4f}\n"
        return report


class InferenceCallback(Callback):
    """
    Callback to run inference on a set of test prompts after each epoch.
    """
    def __init__(self,
                 tokenizer: BaseTokenizer,
                 test_prompts: List[str],
                 generation_kwargs: Optional[Dict[str, Any]] = None,
                 output_subdir: str = "epoch_generations",
                 log_generations: bool = True):
        super().__init__()
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts
        self.generation_kwargs = generation_kwargs if generation_kwargs is not None else {}
        self.output_subdir = output_subdir
        self.log_generations = log_generations # Whether to log generated text to console
        self.generation_count = 0
        logger.info(f"InferenceCallback initialized for {len(test_prompts)} prompts.")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        if not self.model or not self.device:
            logger.warning("InferenceCallback: Model or device not set. Skipping inference.")
            return
        if not self.output_dir:
            logger.warning("InferenceCallback: Output directory not set. Generations will not be saved to files.")
            # Decide if you want to proceed with console logging only or skip
            # return # Uncomment to skip if output_dir is mandatory

        logger.info(f"InferenceCallback: Epoch {epoch} ended. Running inference...")
        original_mode_is_training = self.model.training
        self.model.eval() # Ensure model is in eval mode

        epoch_gen_dir = None
        if self.output_dir:
            epoch_gen_dir = os.path.join(self.output_dir, self.output_subdir, f"epoch_{epoch}")
            os.makedirs(epoch_gen_dir, exist_ok=True)

        for i, prompt_text in enumerate(self.test_prompts):
            if self.log_generations:
                logger.info(f"  Generating for prompt: '{prompt_text}'")
            try:
                _, generated_text = run_generation(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt_text=prompt_text,
                    device=self.device,
                    show_progress=False, # Typically false for automated callbacks
                    **self.generation_kwargs
                )
                if self.log_generations:
                    logger.info(f"    Generated (first 150 chars): {generated_text[:150].replace(os.linesep, ' ')}...")

                if epoch_gen_dir:
                    output_file = os.path.join(epoch_gen_dir, f"prompt_{i+1}_epoch_{epoch}.txt")
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(f"Epoch: {epoch}\n")
                        f.write(f"Prompt: {prompt_text}\n\n")
                        f.write(f"Generated Text:\n{generated_text}\n")
                self.generation_count += 1

            except Exception as e:
                logger.error(f"    Error generating text for prompt '{prompt_text}': {e}", exc_info=True)

        if original_mode_is_training:
            self.model.train() # Switch back to original mode

        if epoch_gen_dir:
            logger.info(f"InferenceCallback: Inference complete for epoch {epoch}. Saved to {epoch_gen_dir}")
        else:
            logger.info(f"InferenceCallback: Inference complete for epoch {epoch}. (Not saved to files as output_dir was not set)")


