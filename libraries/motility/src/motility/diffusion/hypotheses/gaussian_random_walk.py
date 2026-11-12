import numpy as np
from scipy.stats import kstest

from motility.diffusion.base import DiffusionTest
from motility.tracking.trajectories import Trajectory

MIN_DISPLACEMENTS = 3


class GaussianRandomWalkTest(DiffusionTest):
    """Passive diffusion: each coordinate of the displacement is normal.

    Decided with a Kolmogorov-Smirnov test against the normal law of the same
    mean and deviation, on both coordinates.
    """

    def __init__(self, p_threshold: float = 0.05) -> None:
        super().__init__()
        self.p_threshold = p_threshold

    def analyze(self, trajectory: Trajectory) -> bool:
        if len(trajectory) < MIN_DISPLACEMENTS:
            return False

        displacements = np.array(trajectory.displacements)
        return all(
            self._p_value(displacements[:, axis]) > self.p_threshold for axis in (0, 1)
        )

    def parameters(self, trajectory: Trajectory) -> dict[str, float | list[float]]:
        displacements = np.array(trajectory.displacements)
        return {
            "dx_mean": float(np.mean(displacements[:, 0])),
            "dy_mean": float(np.mean(displacements[:, 1])),
            "dx_std": float(np.std(displacements[:, 0])),
            "dy_std": float(np.std(displacements[:, 1])),
        }

    @staticmethod
    def _p_value(data: np.ndarray) -> float:
        if len(data) < 2 or np.std(data) <= np.finfo(float).eps:
            return 0.0
        standardized = (data - np.mean(data)) / np.std(data)
        _, p_value = kstest(standardized, "norm")
        return float(p_value)
