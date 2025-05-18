# ./distillation/trainers/__init__.py
"""
Trainers for distillation.
"""
from distillation.trainers.backbone_trainer import BackboneDistillationTrainer
from distillation.trainers.head_trainer import HeadDistillationTrainer
from distillation.trainers.base_trainer import BaseDistillationTrainer

__all__ = ['BackboneDistillationTrainer', 'HeadDistillationTrainer', 'BaseDistillationTrainer']
