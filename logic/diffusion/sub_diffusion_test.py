import numpy as np

from logic.diffusion.base import DiffusionTest
from logic.diffusion.msd import DisplacementCurve, mean_squared_displacement
from logic.trajectories import Trajectory

LAG_DIVISOR = 2
NULL_DISPLACEMENT = 1e-9


class SubDiffusionTest(DiffusionTest):
    """Subdiffusion in a crowded medium: MSD grows as a power law below one.

    Decided on the exponent of the log-log fit of the first half of the curve,
    where a straight line still holds on a trajectory that leaves the field.
    """

    def __init__(self, alpha_threshold: float = 0.8, min_points: int = 10) -> None:
        super().__init__()
        self.alpha_threshold = alpha_threshold
        self.min_points = min_points

    def analyze(self, trajectory: Trajectory) -> bool:
        curve = self.curve(trajectory)
        if len(curve.squared) < self.min_points:
            return False
        alpha = self._alpha(curve)
        return alpha is not None and alpha < self.alpha_threshold

    def parameters(self, trajectory: Trajectory) -> dict[str, float | list[float]]:
        curve = self.curve(trajectory)
        alpha = self._alpha(curve)
        return {
            "msd": list(curve.squared),
            "tau": [float(lag) for lag in curve.lags],
            "alpha": alpha if alpha is not None else 0.0,
        }

    @staticmethod
    def curve(trajectory: Trajectory) -> DisplacementCurve:
        return mean_squared_displacement(
            trajectory, max_lag=len(trajectory.positions) // LAG_DIVISOR
        )

    @staticmethod
    def _alpha(curve: DisplacementCurve) -> float | None:
        squared = np.array(curve.squared)
        lags = np.array(curve.lags, dtype=float)
        usable = (squared > NULL_DISPLACEMENT) & (lags > 0)
        if usable.sum() < 2:
            return None
        return float(np.polyfit(np.log(lags[usable]), np.log(squared[usable]), 1)[0])
