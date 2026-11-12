from abc import ABC, abstractmethod

from microorganism_computer_vision.tracking.trajectories import Trajectory

Measurement = dict[str, float | list[float]]


class DiffusionTest(ABC):
    """One hypothesis about how a cell moves, decided from its trajectory."""

    @abstractmethod
    def analyze(self, trajectory: Trajectory) -> bool:
        """Whether the trajectory follows this hypothesis."""

    @abstractmethod
    def parameters(self, trajectory: Trajectory) -> Measurement:
        """The numbers the decision was taken on, to explain it."""
