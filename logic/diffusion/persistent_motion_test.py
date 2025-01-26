from typing import List, Tuple

import numpy as np
from logic.diffusion.base import DiffusionTest
from logic.trajectories import Trajectory


class PersistentMotionTest(DiffusionTest):
    """
    Detects directional persistence in motion using velocity autocorrelation.
    Characteristic of active transport or guided migration.
    """
    def __init__(self, persistence_threshold: float = 0.7, window_size: int = 5):
        super().__init__()
        self.persistence_threshold = persistence_threshold
        self.window_size = window_size

    def analyze(self, trajectory: Trajectory) -> bool:
        velocities = self._calculate_velocities(trajectory)
        if len(velocities) < self.window_size * 2:
            return False
            
        # Calculate directional autocorrelation
        corr = self._autocorrelation(velocities)
        return np.mean(corr[:self.window_size]) > self.persistence_threshold

    def _calculate_velocities(self, trajectory: Trajectory) -> List[Tuple[float, float]]:
        displacements = trajectory.displacements()
        time_deltas = trajectory.time_deltas()
        return [(dx/dt, dy/dt) for (dx, dy), dt in zip(displacements, time_deltas)]

    def _autocorrelation(self, velocities: List[Tuple[float, float]]):
        complex_vectors = [dx + 1j*dy for dx, dy in velocities]
        corr = np.correlate(complex_vectors, complex_vectors, mode='full')
        return np.abs(corr[len(corr)//2:]/len(velocities))

    def parameters(self, trajectory: Trajectory) -> dict:
        velocities = self._calculate_velocities(trajectory)
        return {
            'autocorrelation': self._autocorrelation(velocities),
            'mean_persistence': np.mean(self._autocorrelation(velocities)[:self.window_size])
        }