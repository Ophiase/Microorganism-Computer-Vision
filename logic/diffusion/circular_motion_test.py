import numpy as np

from logic.diffusion.base import DiffusionTest
from logic.trajectories import Trajectory


class CircularMotionTest(DiffusionTest):
    """Circular or helical swimming, as magnetotactic bacteria do.

    Decided on the total rotation over the trajectory and on its mean
    curvature: a straight run turns by nothing, a random walk turns back.
    """

    def __init__(
        self, min_rotations: float = 0.5, curvature_threshold: float = 0.1
    ) -> None:
        super().__init__()
        self.min_rotations = min_rotations
        self.curvature_threshold = curvature_threshold

    def analyze(self, trajectory: Trajectory) -> bool:
        positions = np.array(trajectory.positions)
        if len(positions) < 10:
            return False

        total_angle = self._total_rotation(positions)
        return (
            abs(total_angle) > 2 * np.pi * self.min_rotations
            and self._mean_curvature(positions) > self.curvature_threshold
        )

    def parameters(self, trajectory: Trajectory) -> dict[str, float | list[float]]:
        positions = np.array(trajectory.positions)
        return {
            "total_rotation": self._total_rotation(positions),
            "mean_curvature": self._mean_curvature(positions),
        }

    @staticmethod
    def _total_rotation(positions: np.ndarray) -> float:
        vectors = positions[1:] - positions[:-1]
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        return float(np.sum(np.diff(angles)))

    @staticmethod
    def _mean_curvature(positions: np.ndarray) -> float:
        dx = np.gradient(positions[:, 0])
        dy = np.gradient(positions[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        denominator = (dx**2 + dy**2) ** 1.5
        denominator[denominator == 0] = np.nan
        curvature = np.abs(ddx * dy - dx * ddy) / denominator
        return float(np.nanmean(curvature))
