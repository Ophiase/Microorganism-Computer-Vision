import numpy as np

from logic.diffusion.base import DiffusionTest
from logic.trajectories import Trajectory

WINDOW = 5


class PersistentMotionTest(DiffusionTest):
    """Active transport: the direction of travel survives from one step to the next.

    Decided on the autocorrelation of the velocity, seen as complex numbers so
    that a rotation counts as a loss of correlation.
    """

    def __init__(
        self, persistence_threshold: float = 0.7, window_size: int = WINDOW
    ) -> None:
        super().__init__()
        self.persistence_threshold = persistence_threshold
        self.window_size = window_size

    def analyze(self, trajectory: Trajectory) -> bool:
        velocities = self._velocities(trajectory)
        if len(velocities) < self.window_size * 2:
            return False
        return float(np.mean(self._autocorrelation(velocities)[: self.window_size])) > (
            self.persistence_threshold
        )

    def parameters(self, trajectory: Trajectory) -> dict[str, float | list[float]]:
        autocorrelation = self._autocorrelation(self._velocities(trajectory))
        return {
            "autocorrelation": [float(value) for value in autocorrelation],
            "mean_persistence": float(np.mean(autocorrelation[: self.window_size])),
        }

    @staticmethod
    def _velocities(trajectory: Trajectory) -> list[complex]:
        return [
            complex(dx / delta, dy / delta)
            for (dx, dy), delta in zip(
                trajectory.displacements, trajectory.time_deltas, strict=True
            )
            if delta != 0
        ]

    @staticmethod
    def _autocorrelation(velocities: list[complex]) -> np.ndarray:
        if not velocities:
            return np.zeros(1)
        correlated = np.correlate(velocities, velocities, mode="full")
        return np.abs(correlated[len(correlated) // 2 :] / len(velocities))
