# ./inference/__init__.py
"""
Inference module for cleanGPT
Provides utilities for text generation and model inference
"""

from .generation import (
    run_generation,
    batch_generate,
    get_generation_args
)

# Export main functions
__all__ = [
    'run_generation',
    'batch_generate',
    'get_generation_args'
]
