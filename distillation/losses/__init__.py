# ./distillation/losses/__init__.py
"""
Loss functions for distillation.
"""
from distillation.losses.hidden_state_loss import DistillationLoss
from distillation.losses.logit_loss import LogitDistillationLoss

__all__ = ['DistillationLoss', 'LogitDistillationLoss']
