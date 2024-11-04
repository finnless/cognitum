import hashlib
import random
from typing import List, Optional
import json

from .types import DataItem, Dataset
from .exceptions import DatasetError

class DatasetManager:
    """
    Manages a dataset of items to be classified.
    
    Each item in the dataset is a tuple of (id, text) where:
    - id: A unique identifier for the item
    - text: The text to be classified
    """
    
    def __init__(self, data: List[DataItem]):
        """
        Initialize a new dataset.
        
        Args:
            data: List of tuples, each containing (id, text)
        
        Raises:
            DatasetError: If data format is invalid
        """
        # TODO handle input if list of lists or other formats
        self._validate_data(data)
        self.data = data
    
    def _validate_data(self, data: List[DataItem]) -> None:
        """
        Validate the input data format.
        
        Args:
            data: List of tuples to validate
            
        Raises:
            DatasetError: If validation fails
        """
        if not isinstance(data, list):
            raise DatasetError("Data must be a list")
        
        seen_ids = set()
        for item in data:
            if not isinstance(item, tuple) or len(item) != 2:
                raise DatasetError("Each item must be a tuple of (id, text)")
            if not isinstance(item[0], str) or not isinstance(item[1], str):
                raise DatasetError("Both id and text must be strings")
            if item[0] in seen_ids:
                raise DatasetError(f"Duplicate id found: {item[0]}")
            seen_ids.add(item[0])
    
    def hash(self) -> str:
        """
        Generate a unique hash for the dataset.
        
        Returns:
            str: A hex string representing the hash of the dataset
        """
        # Sort data by id to ensure consistent hashing
        sorted_data = sorted(self.data, key=lambda x: x[0])
        data_str = json.dumps(sorted_data)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def sample(self, n: int, seed: Optional[int] = None) -> List[DataItem]:
        """
        Return a random sample of the dataset.
        
        Args:
            n: Number of samples to return
            seed: Random seed for reproducibility
            
        Returns:
            List[DataItem]: A random sample of n items
            
        Raises:
            DatasetError: If n is larger than dataset size
        """
        if n > len(self.data):
            raise DatasetError(f"Sample size {n} larger than dataset size {len(self.data)}")
        
        if seed is not None:
            random.seed(seed)
        
        return random.sample(self.data, n)
    
    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> DataItem:
        """Get an item from the dataset by index."""
        return self.data[idx]
