from abc import ABC, abstractmethod
from logic.trajectories import Trajectory

class DiffusionAnalysis(ABC):
    @abstractmethod
    def analyze(self, trajectory: Trajectory) -> bool:
        pass
    
    @abstractmethod
    def parameters(self, trajectory: Trajectory) -> dict:
        pass