"""
cognitum: A library for classifying free response poll data using language models.

This library provides tools for managing datasets of text responses and using
language models to classify them according to a predefined codebook.

Main components:
- DatasetManager: Handles data loading, sampling, and hashing
- Model: Manages model configuration, prediction, and evaluation
"""

from .dataset import DatasetManager
from .model import Model
from .exceptions import PollClassifierError, DatasetError, ModelError, ValidationError
from .types import (
    DataItem, Dataset, Label, Labels, 
    Confidence, Confidences, PredictionItem, 
    Predictions, EvaluationMetrics
)

__all__ = [
    "DatasetManager",
    "Model",
    "PollClassifierError",
    "DatasetError",
    "ModelError",
    "ValidationError",
    "DataItem",
    "Dataset",
    "Label",
    "Labels",
    "Confidence",
    "Confidences",
    "PredictionItem",
    "Predictions",
    "EvaluationMetrics"
]
