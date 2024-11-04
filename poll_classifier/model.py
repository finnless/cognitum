from typing import List, Optional, Union, Any
import lmql

from .types import (
    DataItem, Dataset, Label, Labels, 
    Confidence, Confidences, PredictionItem, 
    Predictions, EvaluationMetrics
)
from .exceptions import ModelError, DatasetError

class Model:
    """
    A model for classifying text data.
    
    Attributes:
        prompt (str): Template string for formatting input text
        valid_labels (List[str]): List of valid classification labels
        model: The language model to use for classification
    """
    
    def __init__(
        self,
        model: Any,
        prompt: str,
        valid_labels: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize a new model.
        
        Args:
            model: Language model instance
            prompt: Template string for formatting input text. There must be
                a placeholder for the text to classify marked with `{text}`.
            valid_labels: List of valid classification labels, optional.
            **kwargs: Additional model configuration
        """
        self.model = model
        self.prompt = prompt
        self.valid_labels = valid_labels
        self.config = kwargs
        
        # Check that prompt contains {text}
        if "{text}" not in self.prompt:
            raise ValueError("Prompt must contain {text} placeholder")
    
    def predict(
        self,
        dataset: Dataset,
        return_confidences: bool = False,
        batch_size: int = 10
    ) -> Predictions:
        """
        Generate predictions for a dataset in batches.
        
        Args:
            dataset: List of (id, text) tuples to classify
            return_confidences: Whether to include confidence scores
            batch_size: Number of rows to process in each batch
            
        Returns:
            List of predictions, each either (id, labels) or 
            (id, labels, confidences) if return_confidences is True
        """
        predictions = []
        remaining_items = list(dataset)
        # TODO: Implement caching
        # TODO: Experiment with RAG
        
        while remaining_items:
            # Process a batch of items
            batch = remaining_items[:batch_size]
            remaining_items = remaining_items[batch_size:]
            
            # Concatenate text for the batch into a single prompt
            concatenated_text = "\n".join(text for _, text in batch)
            formatted_prompt = self.prompt.format(text=concatenated_text)
            
            # Get model prediction for the concatenated prompt
            # Note: Actual implementation would depend on the specific model
            # This is a placeholder
            batch_result = self._get_model_prediction(formatted_prompt, return_confidences)
            
            # Match predictions with inputs
            # Assuming batch_result is a list of (text, labels, confidences) tuples
            matched_ids = set()
            for (item_id, text) in batch:
                if return_confidences:
                    for result_text, labels, confidences in batch_result:
                        if text == result_text:
                            predictions.append((item_id, labels, confidences))
                            matched_ids.add(item_id)
                            break
                else:
                    for result_text, labels in batch_result:
                        if text == result_text:
                            predictions.append((item_id, labels))
                            matched_ids.add(item_id)
                            break
            
            # Add unmatched items back to the queue
            unmatched_items = [(item_id, text) for item_id, text in batch if item_id not in matched_ids]
            remaining_items.extend(unmatched_items)
        
        return predictions
    
    def evaluate(
        self, 
        predictions: Predictions,  # Use Predictions type
        ground_truth: Predictions  # Use Predictions type
    ) -> EvaluationMetrics:
        """
        Evaluate model predictions against ground truth.
        
        Args:
            predictions: List of predictions, each either (id, labels) or 
                         (id, labels, confidences)
            ground_truth: Ground truth labels, each either (id, labels) or 
                          (id, labels, confidences)
        
        Returns:
            EvaluationMetrics containing exact match, partial match,
            and false positive rates
        
        Raises:
            DatasetError: If there is a mismatch in item_ids between predictions and ground_truth
        """
        # Convert ground_truth to a dictionary for quick lookup
        ground_truth_dict = {item_id: labels for item_id, labels, *_ in ground_truth}
        
        # Validate that both predictions and ground_truth have the same item_ids
        prediction_ids = {item_id for item_id, _ in predictions}
        ground_truth_ids = set(ground_truth_dict.keys())
        
        if prediction_ids != ground_truth_ids:
            missing_in_predictions = ground_truth_ids - prediction_ids
            missing_in_ground_truth = prediction_ids - ground_truth_ids
            raise DatasetError(
                f"Mismatch in item_ids. "
                f"Missing in predictions: {missing_in_predictions}. "
                f"Missing in ground_truth: {missing_in_ground_truth}."
            )
        
        exact_matches = 0
        partial_matches = 0
        extra_in_predictions = 0
        extra_in_ground_truth = 0
        
        for item in predictions:
            item_id, predicted_labels = item[:2]  # Extract id and labels
            true_labels = ground_truth_dict[item_id]
            
            # Calculate exact matches
            if set(predicted_labels) == set(true_labels):
                exact_matches += 1
            else:
                # Calculate partial matches
                if set(predicted_labels) & set(true_labels):
                    partial_matches += 1
                
                # Calculate extra in predictions
                extra_in_predictions += len(set(predicted_labels) - set(true_labels))
                
                # Calculate extra in ground truth
                extra_in_ground_truth += len(set(true_labels) - set(predicted_labels))
        
        total_items = len(predictions)
        
        return EvaluationMetrics(
            exact=exact_matches / total_items,
            partial=partial_matches / total_items,
            false_positives=extra_in_predictions / total_items,
            false_negatives=extra_in_ground_truth / total_items
        )
    
    def _get_model_prediction(self, prompt: str, return_confidences: bool = False) -> List[Union[tuple[str, Labels], tuple[str, Labels, Confidences]]]:
        """
        Get prediction from the model.
        
        Args:
            prompt: Formatted prompt to send to model
            return_confidences: Whether to include confidence scores
            
        Returns:
            List of tuples, each containing result_text, labels, and optionally confidences
        """
        # TODO: Constrain output to valid labels in self.valid_labels
        @lmql.query(model=self.model)
        def llm_query():
            '''lmql
            argmax
                "{prompt}"
                "[RESPONSE]"
            '''
        result = llm_query()
        
        # Remove the input prompt (prompt) from the output prompt to get the response
        result_output = result.prompt.split(prompt)[1]
        
        # Parse the result into a list of (result_text, labels) or (result_text, labels, confidences) tuples
        lines = result_output.strip().split("\n")
        parsed_results = []
        
        for line in lines:
            parts = line.split("|")
            if len(parts) == 2:
                result_text, labels_str = parts
                labels = [label.lower() for label in labels_str.split(", ")]  # Lowercase each label
                if return_confidences:
                    # Placeholder for confidence extraction logic
                    # TODO: Implement confidence extraction logic
                    confidences = [1.0] * len(labels)  # Example: default confidence
                    parsed_results.append((result_text, labels, confidences))
                else:
                    parsed_results.append((result_text, labels))
            else:
                raise ModelError(f"Invalid format from llm: {line}")
        
        return parsed_results
