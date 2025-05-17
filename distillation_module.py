# ./distillation_module.py
"""
Main module for block-by-block distillation of transformer models.
This module imports and exposes the necessary trainer and loss classes.
"""
import logging

# Configure basic logging for the module if not already configured by the application
# This is a simple setup; a more sophisticated application might configure logging globally.
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Initialize to None in case of import errors, to make explicit what happens.
DistillationLoss = None
LogitDistillationLoss = None
BlockDistillationTrainer = None

# Import classes from the separated files using direct imports
try:
    # Changed from .distillation_loss to distillation_loss
    from distillation_loss import DistillationLoss, LogitDistillationLoss
    # Changed from .distillation_trainers to distillation_trainers
    from distillation_trainers import BlockDistillationTrainer
    # Assuming stitching_layers.py is co-located or in PYTHONPATH and handles its own exports
    # from stitching_layers import StitchingLayer, StitchingDistillationLoss # If needed to be re-exported
except ImportError as e:
    logger.error(f"Error importing submodules in distillation_module: {e}. "
                 "Ensure distillation_loss.py and distillation_trainers.py are in the same directory "
                 "as distillation_module.py or correctly placed in your Python path.")
    # Variables remain None if imports fail


# Define what gets imported when 'from distillation_module import *' is used
__all__ = [
    'DistillationLoss',
    'LogitDistillationLoss',
    'BlockDistillationTrainer',
    # Add StitchingLayer, StitchingDistillationLoss here if you want to re-export them
    # from this top-level module and they are imported above.
]

# Log loaded classes, checking if they were successfully imported
loaded_classes = [name for name in __all__ if globals().get(name) is not None]
if len(loaded_classes) < len(__all__):
    missing_classes = [name for name in __all__ if globals().get(name) is None]
    logger.warning(f"distillation_module loaded, but some classes might be missing due to import errors: {missing_classes}")
logger.info("distillation_module loaded. Available classes (successfully imported): %s", loaded_classes)


