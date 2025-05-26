# dataloaders/strategies.py
"""
Concrete implementations of curriculum learning strategies.
"""

from .base import CurriculumStrategy
from typing import List
import math


class LinearTransitionStrategy(CurriculumStrategy):
    """Linear transition between start and end weights over a specified number of epochs."""
    
    def __init__(self, start_weights: List[float], end_weights: List[float], 
                 transition_epochs: int, **kwargs):
        if len(start_weights) != len(end_weights):
            raise ValueError("Start and end weights must have the same length")
        
        super().__init__(len(start_weights), **kwargs)
        self.start_weights = start_weights
        self.end_weights = end_weights
        self.transition_epochs = transition_epochs
        
        # Normalize weights
        self.start_weights = self._normalize_weights(start_weights)
        self.end_weights = self._normalize_weights(end_weights)
    
    def _normalize_weights(self, weights: List[float]) -> List[float]:
        """Normalize weights to sum to 1.0."""
        total = sum(weights)
        return [w / total for w in weights] if total > 0 else weights
    
    def get_weights(self, epoch: int) -> List[float]:
        if epoch >= self.transition_epochs:
            return self.end_weights[:]
        
        # Linear interpolation
        progress = epoch / self.transition_epochs
        weights = []
        for start, end in zip(self.start_weights, self.end_weights):
            weight = start + (end - start) * progress
            weights.append(weight)
        
        return self._normalize_weights(weights)


class StepScheduleStrategy(CurriculumStrategy):
    """Step-wise changes in dataset weights at specified epochs."""
    
    def __init__(self, schedule: List[Dict], **kwargs):
        """
        Args:
            schedule: List of dicts like [{'epoch': 0, 'weights': [0.7, 0.3]}, 
                                        {'epoch': 5, 'weights': [0.9, 0.1]}]
        """
        if not schedule:
            raise ValueError("Schedule cannot be empty")
        
        # Sort by epoch
        self.schedule = sorted(schedule, key=lambda x: x['epoch'])
        num_datasets = len(self.schedule[0]['weights'])
        super().__init__(num_datasets, **kwargs)
        
        # Normalize all weights
        for item in self.schedule:
            item['weights'] = self._normalize_weights(item['weights'])
    
    def _normalize_weights(self, weights: List[float]) -> List[float]:
        total = sum(weights)
        return [w / total for w in weights] if total > 0 else weights
    
    def get_weights(self, epoch: int) -> List[float]:
        # Find the appropriate schedule entry
        current_weights = self.schedule[0]['weights']
        for item in self.schedule:
            if epoch >= item['epoch']:
                current_weights = item['weights']
            else:
                break
        return current_weights[:]


class ExponentialDecayStrategy(CurriculumStrategy):
    """Exponential decay for transitioning between datasets."""
    
    def __init__(self, start_weights: List[float], decay_rates: List[float], **kwargs):
        if len(start_weights) != len(decay_rates):
            raise ValueError("Start weights and decay rates must have same length")
        
        super().__init__(len(start_weights), **kwargs)
        self.start_weights = self._normalize_weights(start_weights)
        self.decay_rates = decay_rates
    
    def _normalize_weights(self, weights: List[float]) -> List[float]:
        total = sum(weights)
        return [w / total for w in weights] if total > 0 else weights
    
    def get_weights(self, epoch: int) -> List[float]:
        weights = []
        for start_weight, decay_rate in zip(self.start_weights, self.decay_rates):
            weight = start_weight * math.exp(-decay_rate * epoch)
            weights.append(weight)
        
        return self._normalize_weights(weights)

