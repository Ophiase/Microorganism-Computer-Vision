import numpy as np
from logic.diffusion.base import DiffusionTest
from logic.trajectories import Trajectory


class SubDiffusionTest(DiffusionTest):
    """
    Tests for subdiffusive behavior using power law MSD: 
    <x²(t)> ~ t^alpha with alpha < 1
    """

    def __init__(self, alpha_threshold: float = 0.8, min_points: int = 10):
        super().__init__()
        self.alpha_threshold = alpha_threshold
        self.min_points = min_points

    def calculate_msd(self, trajectory: Trajectory) -> tuple[list[float], list[int]]:
        """Calculate mean squared displacement for different time intervals"""
        positions = trajectory.positions()
        frames = trajectory.frames()
        msd = []
        tau = []

        # Use first half of trajectory for analysis
        max_dt = len(positions) // 2
        for dt in range(1, max_dt + 1):
            displacements = []
            for i in range(len(positions) - dt):
                dx = positions[i+dt][0] - positions[i][0]
                dy = positions[i+dt][1] - positions[i][1]
                displacements.append(dx**2 + dy**2)
            if displacements:
                msd.append(np.mean(displacements))
                tau.append(frames[dt] - frames[0])
        return msd, tau

    def analyze(self, trajectory: Trajectory) -> bool:
        msd, tau = self.calculate_msd(trajectory)
        if len(msd) < self.min_points:
            return False

        try:
            # Filter out zero values for log transform
            valid_idx = (np.array(msd) > 1e-9) & (np.array(tau) > 0)
            log_t = np.log(np.array(tau)[valid_idx])
            log_msd = np.log(np.array(msd)[valid_idx])

            if len(log_t) < 2:
                return False

            coeffs = np.polyfit(log_t, log_msd, 1)
            alpha = coeffs[0]
        except Exception as e:
            return False

        return alpha < self.alpha_threshold

    def parameters(self, trajectory: Trajectory) -> dict:
        msd, tau = self.calculate_msd(trajectory)
        try:
            valid_idx = (np.array(msd) > 1e-9) & (np.array(tau) > 0)
            log_t = np.log(np.array(tau)[valid_idx])
            log_msd = np.log(np.array(msd)[valid_idx])
            coeffs = np.polyfit(log_t, log_msd, 1)
            alpha = coeffs[0]
        except:
            alpha = None
        return {'msd': msd, 'tau': tau, 'alpha': alpha}
