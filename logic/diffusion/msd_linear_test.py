from typing import List
import numpy as np
from logic.diffusion import DiffusionTest
from logic.trajectories import Trajectory


class MSDLinearTest(DiffusionTest):
    """
    Analyzes diffusion characteristics by testing if mean squared displacement (MSD)
    shows linear behavior over time, using R-squared of linear fit as the criterion.
    """
    
    def __init__(self, r_squared_threshold: float = 0.9):
        super().__init__()
        self.r_squared_threshold = r_squared_threshold

    def analyze(self, trajectory: Trajectory) -> bool:
        msd, tau = self.calculate_msd(trajectory)
        if len(msd) < 2:
            return False

        # Fit linear regression
        coeffs = np.polyfit(tau, msd, 1)
        predicted = np.polyval(coeffs, tau)
        ss_res = np.sum((msd - predicted)**2)
        ss_tot = np.sum((msd - np.mean(msd))**2)
        r_squared = 1 - (ss_res / ss_tot)
        return r_squared > self.r_squared_threshold

    def calculate_msd(self, trajectory: Trajectory) -> tuple[List[float], List[int]]:
        positions = trajectory.positions()
        frames = trajectory.frames()
        msd = []
        tau = []

        for dt in range(1, len(positions)):
            displacements = []
            for i in range(len(positions)-dt):
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