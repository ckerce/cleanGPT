# dataloaders/utils.py
"""
Utility functions for curriculum learning.
"""

import logging
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def visualize_curriculum_schedule(strategy, num_epochs: int, dataset_names: List[str] = None):
    """
    Visualize how dataset weights change over epochs.
    
    Args:
        strategy: CurriculumStrategy instance
        num_epochs: Number of epochs to visualize
        dataset_names: Names of datasets for legend
    """
    epochs = list(range(num_epochs))
    weights_over_time = []
    
    for epoch in epochs:
        weights = strategy.get_weights(epoch)
        weights_over_time.append(weights)
    
    # Convert to numpy array for easier plotting
    weights_array = np.array(weights_over_time)
    
    plt.figure(figsize=(10, 6))
    
    for i in range(weights_array.shape[1]):
        label = dataset_names[i] if dataset_names and i < len(dataset_names) else f'Dataset {i}'
        plt.plot(epochs, weights_array[:, i], label=label, marker='o', markersize=4)
    
    plt.xlabel('Epoch')
    plt.ylabel('Dataset Weight')
    plt.title('Curriculum Learning Schedule')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    return plt.gcf()


def log_dataset_statistics(dataloader, num_batches_to_sample: int = 100):
    """Log statistics about the curriculum dataset sampling."""
    dataset_counts = {}
    total_samples = 0
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches_to_sample:
            break
            
        sources = batch.get('dataset_sources', [])
        for source in sources:
            dataset_counts[source] = dataset_counts.get(source, 0) + 1
            total_samples += 1
    
    logger.info("Dataset sampling statistics:")
    for dataset, count in dataset_counts.items():
        percentage = (count / total_samples) * 100
        logger.info(f"  {dataset}: {count} samples ({percentage:.1f}%)")
    
    return dataset_counts
