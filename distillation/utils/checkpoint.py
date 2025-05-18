# ./distillation/utils/checkpoint.py
"""
Checkpoint utilities for distillation.
"""
import os
import logging
import torch

logger = logging.getLogger(__name__)

def save_checkpoint(model, path, **kwargs):
    """
    Save a model checkpoint.
    
    Args:
        model: Model to save
        path: Path to save the checkpoint
        **kwargs: Additional information to include in the checkpoint
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': model.config if hasattr(model, 'config') else None,
            **kwargs
        }
        
        torch.save(checkpoint, path, pickle_protocol=4)
        logger.info(f"Checkpoint saved to {path}")
        return True
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")
        return False

def load_checkpoint(path, model=None, device=None):
    """
    Load a checkpoint.
    
    Args:
        path: Path to the checkpoint
        model: Optional model to load weights into
        device: Device to load the checkpoint to
        
    Returns:
        Loaded checkpoint dictionary if successful, None otherwise
    """
    try:
        if not os.path.exists(path):
            logger.error(f"Checkpoint path does not exist: {path}")
            return None
            
        map_location = device if device is not None else torch.device('cpu')
        checkpoint = torch.load(path, map_location=map_location)
        
        # Load weights into model if provided
        if model is not None and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model weights from {path}")
            
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return None
