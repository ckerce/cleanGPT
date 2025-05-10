# -*- coding: utf-8 -*-
############################################
#                                          #
#  Training Loop Class                     #
#                                          #
############################################

import torch
import time
from tqdm.auto import tqdm # Progress bar

class Trainer:
    """
    Encapsulates the training loop logic.
    """
    def __init__(self, model, dataloader, optimizer, device, num_epochs):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        print("\nTrainer initialized.")
        print(f"  Model: {type(self.model).__name__}")
        print(f"  Optimizer: {type(self.optimizer).__name__}")
        print(f"  Device: {self.device}")
        print(f"  Num Epochs: {self.num_epochs}")

    def train(self):
        """
        Executes the training loop.
        """
        print("\nStarting training...")
        self.model.to(self.device) # Ensure model is on the correct device
        self.model.train()         # Set model to training mode

        total_training_start_time = time.time()

        for epoch in range(self.num_epochs):
            print(f"\n--- Epoch {epoch+1}/{self.num_epochs} ---")
            epoch_loss = 0
            epoch_start_time = time.time()
            
            # Use tqdm for a progress bar over the dataloader
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}", leave=False)

            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch) # Assumes model returns dict with 'loss'
                loss = outputs['loss']

                # Check if loss is valid (e.g., not None if labels were missing and model handles it)
                if loss is None:
                    # This might happen if a batch contains only padding or if the model logic skips loss calculation
                    print("Warning: Loss is None for a batch, skipping optimization step.")
                    continue
                if torch.isnan(loss):
                    print("Warning: Loss is NaN, stopping training.")
                    return # Stop training if loss becomes NaN

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                # Optional: Gradient Clipping (prevents exploding gradients)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss
                
                # Update progress bar description
                progress_bar.set_postfix({"loss": f"{batch_loss:.4f}"})


            # --- Epoch End ---
            avg_epoch_loss = epoch_loss / len(self.dataloader)
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            print(f"Epoch {epoch+1} finished.")
            print(f"  Average Loss: {avg_epoch_loss:.6f}")
            # Print model-specific info if available (like OOBC param)
            if hasattr(self.model, 'learnable_param'):
                 print(f"  OOBC Parameter Value: {self.model.learnable_param.item():.6f}")
            print(f"  Epoch Duration: {epoch_duration:.2f} seconds")

        total_training_end_time = time.time()
        total_duration = total_training_end_time - total_training_start_time
        print("\n--- Training Complete ---")
        print(f"Total Training Duration: {total_duration:.2f} seconds")
