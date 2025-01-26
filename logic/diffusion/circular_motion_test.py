import numpy as np
from logic.diffusion.base import DiffusionTest
from logic.trajectories import Trajectory


class CircularMotionTest(DiffusionTest):
    """
    Identifies circular/helical motion patterns using curvature analysis.
    Typical of magnetotactic bacteria or spiral-seeking organisms.
    """
    def __init__(self, min_rotations: float = 0.5, curvature_threshold: float = 0.1):
        super().__init__()
        self.min_rotations = min_rotations
        self.curvature_threshold = curvature_threshold

    def analyze(self, trajectory: Trajectory) -> bool:
        positions = np.array(trajectory.positions())
        if len(positions) < 10:
            return False
            
        # Calculate total rotation angle
        total_angle = self._calculate_total_rotation(positions)
        mean_curvature = self._calculate_mean_curvature(positions)
        
        return (abs(total_angle) > 2*np.pi*self.min_rotations and 
                mean_curvature > self.curvature_threshold)

    def _calculate_total_rotation(self, positions: np.ndarray) -> float:
        vectors = positions[1:] - positions[:-1]
        angles = np.arctan2(vectors[:,1], vectors[:,0])
        return np.sum(np.diff(angles))

    def _calculate_mean_curvature(self, positions: np.ndarray) -> float:
        dx = np.gradient(positions[:, 0])
        dy = np.gradient(positions[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        denominator = (dx**2 + dy**2)**1.5
        denominator[denominator == 0] = np.nan  # Avoid division by zero
        curvature = np.abs(ddx * dy - dx * ddy) / denominator
        return np.nanmean(curvature)

    def parameters(self, trajectory: Trajectory) -> dict:
        positions = np.array(trajectory.positions())
        return {
            'total_rotation': self._calculate_total_rotation(positions),
            'mean_curvature': self._calculate_mean_curvature(positions)
        }