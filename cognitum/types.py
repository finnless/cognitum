from typing import TypeVar, Union, List, Tuple, Dict, Optional
from dataclasses import dataclass

# Type aliases
DataItem = Tuple[str, List[str]]  # (id, labels)
Dataset = List[DataItem]
Label = str
Labels = Union[Label, List[Label]]
Confidence = float
Confidences = Union[Confidence, List[Confidence]]

# Prediction types
PredictionItem = Union[
    Tuple[str, Labels],  # (id, labels)
    Tuple[str, Labels, Confidences]  # (id, labels, confidences)
]
Predictions = List[PredictionItem]

# Evaluation metrics
@dataclass
class EvaluationMetrics:
    exact: float
    partial: float
    false_positives: float
    false_negatives: float  # Add this if calculating false negatives
