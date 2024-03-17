# Libraries
from abc import ABC, abstractmethod
import numpy as np

# Abstract class
class HarmfulnessPredictor(ABC):
    @abstractmethod
    def predict_proba(self, image:np.ndarray) -> float:
        """Takes in a numpy array image and returns the prediction probability"""
        pass	
