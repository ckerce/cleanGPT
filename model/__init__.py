# -*- coding: utf-8 -*-
"""
Model package initialization.
Imports model architectures for easy access.
"""

# Import model implementations for easier access
from .model_SASPV import SASPTransformerModel
from .model_Vanilla import VanillaTransformerModel

# Dictionary mapping architecture names to model classes
MODEL_REGISTRY = {
    "SASP": SASPTransformerModel,
    "Vanilla": VanillaTransformerModel,
}

def get_model(model_type, config):
    """
    Factory function to get the appropriate model class.
    
    Args:
        model_type (str): The type of model to use (e.g., 'SASP', 'Vanilla')
        config: Configuration object for the model
        
    Returns:
        An instance of the requested model
        
    Raises:
        ValueError: If the model_type is not recognized
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_type]
    return model_class(config=config)
