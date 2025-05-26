# -*- coding: utf-8 -*-
"""
Model package initialization.
Imports model architectures for easy access.
"""

# Import model implementations for easier access
from .model_SASPV import SASPTransformerModel
from .model_Vanilla import VanillaTransformerModel
from .model_token_factored import FactoredTransformerModel
from .model_token_factored_alibi import FactoredTransformerModelALiBi  

# Dictionary mapping architecture names to model classes
MODEL_REGISTRY = {
    "SASP": SASPTransformerModel,
    "Vanilla": VanillaTransformerModel,
    "Factored": FactoredTransformerModel,
    "FactoredALiBi": FactoredTransformerModelALiBi,  
}

def get_model(model_type, config):
    """
    Factory function to get the appropriate model class.
    
    Args:
        model_type (str): The type of model to use (e.g., 'SASP', 'Vanilla', 'Factored', 'FactoredALiBi')
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


def register_model(model_name, model_class):
    """
    Register a new model class in the registry.
    
    Args:
        model_name (str): Name to register the model under
        model_class: The model class to register
        
    Raises:
        ValueError: If the model_name is already registered
    """
    if model_name in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' is already registered. Use a different name or unregister first.")
    
    MODEL_REGISTRY[model_name] = model_class
    print(f"Registered model '{model_name}' -> {model_class.__name__}")


def unregister_model(model_name):
    """
    Remove a model from the registry.
    
    Args:
        model_name (str): Name of the model to unregister
        
    Raises:
        ValueError: If the model_name is not found in the registry
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' is not registered. Available models: {list(MODEL_REGISTRY.keys())}")
    
    removed_class = MODEL_REGISTRY.pop(model_name)
    print(f"Unregistered model '{model_name}' ({removed_class.__name__})")


def list_available_models():
    """
    List all available model types in the registry.
    
    Returns:
        List[str]: List of registered model names
    """
    return list(MODEL_REGISTRY.keys())


def get_model_info():
    """
    Get detailed information about all registered models.
    
    Returns:
        Dict[str, str]: Dictionary mapping model names to their class names
    """
    return {name: cls.__name__ for name, cls in MODEL_REGISTRY.items()}


# Compatibility aliases for backward compatibility
def get_factored_model(config):
    """
    Convenience function to get the original factored model.
    
    Args:
        config: Configuration object for the model
        
    Returns:
        FactoredTransformerModel instance
    """
    return get_model("Factored", config)


def get_factored_alibi_model(config):
    """
    Convenience function to get the ALiBi factored model.
    
    Args:
        config: Configuration object for the model (should be GPTConfigALiBi)
        
    Returns:
        FactoredTransformerModelALiBi instance
    """
    return get_model("FactoredALiBi", config)


# Print available models on import (optional, can be commented out for production)
if __name__ != "__main__":
    print(f"cleanGPT models loaded. Available models: {list_available_models()}")
