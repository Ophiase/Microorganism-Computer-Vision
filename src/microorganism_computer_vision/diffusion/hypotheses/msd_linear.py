import numpy as np

from microorganism_computer_vision.diffusion.base import DiffusionTest
from microorganism_computer_vision.diffusion.msd import (
    DisplacementCurve,
    mean_squared_displacement,
)
from microorganism_computer_vision.tracking.trajectories import Trajectory

LAG_DIVISOR = 4


class MSDLinearTest(DiffusionTest):
    """Normal diffusion: the mean squared displacement grows linearly with time.

    Decided on the R squared of a straight line fitted to the first quarter of
    the curve, where the statistics are the richest.
    """

    def __init__(self, r_squared_threshold: float = 0.9, min_points: int = 5) -> None:
        super().__init__()
        self.r_squared_threshold = r_squared_threshold
        self.min_points = max(min_points, 2)

    def analyze(self, trajectory: Trajectory) -> bool:
        curve = self.curve(trajectory)
        if len(curve.squared) < self.min_points or len(set(curve.squared)) < 2:
            return False

        squared = np.array(curve.squared)
        predicted = np.polyval(np.polyfit(curve.lags, squared, 1), curve.lags)
        residual = float(np.sum((squared - predicted) ** 2))
        total = float(np.sum((squared - np.mean(squared)) ** 2))
        if np.isclose(total, 0.0, atol=1e-12):
            return False

        return 1 - residual / total >= self.r_squared_threshold

    def parameters(self, trajectory: Trajectory) -> dict[str, float | list[float]]:
        curve = self.curve(trajectory)
        return {"msd": list(curve.squared), "tau": [float(lag) for lag in curve.lags]}

    @staticmethod
    def curve(trajectory: Trajectory) -> DisplacementCurve:
        return mean_squared_displacement(
            trajectory, max_lag=len(trajectory.positions) // LAG_DIVISOR
        )
