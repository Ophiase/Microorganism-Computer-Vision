from typing import List
import numpy as np
from logic.diffusion import DiffusionTest
from logic.trajectories import Trajectory


class MSDLinearTest(DiffusionTest):
    """
    Analyzes diffusion characteristics by testing if mean squared displacement (MSD)
    shows linear behavior over time, using R-squared of linear fit as the criterion.
    """

    def __init__(self, r_squared_threshold: float = 0.9, min_points: int = 5):
        super().__init__()
        self.r_squared_threshold = r_squared_threshold
        self.min_points = max(min_points, 2)  # Minimum 2 points for linear fit

    def analyze(self, trajectory: Trajectory) -> bool:
        msd, tau = self.calculate_msd(trajectory)

        # Early exit conditions
        if len(msd) < self.min_points or len(set(msd)) < 2:
            return False

        # Fit linear regression with validation
        try:
            coeffs = np.polyfit(tau, msd, 1)
        except (np.linalg.LinAlgError, TypeError):
            return False

        predicted = np.polyval(coeffs, tau)
        ss_res = np.sum((msd - predicted)**2)
        ss_tot = np.sum((msd - np.mean(msd))**2)

        # Handle zero variance case
        if np.isclose(ss_tot, 0.0, atol=1e-12):
            return False

        r_squared = 1 - (ss_res / ss_tot)
        return r_squared >= self.r_squared_threshold

    def calculate_msd(self, trajectory: Trajectory) -> tuple[List[float], List[int]]:
        positions = trajectory.positions()
        frames = trajectory.frames()
        msd = []
        tau = []

        # Calculate MSD for 25% of trajectory length maximum
        max_dt = max(1, len(positions)//4)
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

    def parameters(self, trajectory: Trajectory) -> dict:
        msd, tau = self.calculate_msd(trajectory)
        return {'msd': msd, 'tau': tau}
