from abc import ABC, abstractmethod
import numpy as np
from .trajectories import Trajectory
from typing import List

class DiffusionAnalysis(ABC):
    @abstractmethod
    def analyze(self, trajectory: Trajectory) -> bool:
        pass
    
    @abstractmethod
    def parameters(self, trajectory: Trajectory) -> dict:
        pass