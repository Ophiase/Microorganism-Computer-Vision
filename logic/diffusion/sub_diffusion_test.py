import numpy as np
from logic.diffusion.base import DiffusionTest
from logic.trajectories import Trajectory


class SubDiffusionTest(DiffusionTest):
    """
    Tests for subdiffusive behavior using power law MSD: <x²(t)> ~ t^α with α < 1
    """
    def __init__(self, alpha_threshold: float = 0.8, min_points: int = 10):
        super().__init__()
        self.alpha_threshold = alpha_threshold
        self.min_points = min_points

    def analyze(self, trajectory: Trajectory) -> bool:
        msd, tau = self.calculate_msd(trajectory)
        if len(msd) < self.min_points:
            return False
            
        try:
            # Fit power law in log space
            log_t = np.log(tau)
            log_msd = np.log(msd)
            coeffs = np.polyfit(log_t, log_msd, 1)
            alpha = coeffs[0]
        except:
            return False
            
        return alpha < self.alpha_threshold

    def parameters(self, trajectory: Trajectory) -> dict:
        msd, tau = self.calculate_msd(trajectory)
        return {'msd': msd, 'tau': tau, 'alpha': self._calculate_alpha(msd, tau)}