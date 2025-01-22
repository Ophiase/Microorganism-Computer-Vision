from abc import ABC, abstractmethod
import numpy as np
from .trajectories import Trajectory
from typing import List

class DiffusionAnalysis(ABC):
    @abstractmethod
    def analyze(self, trajectory: Trajectory) -> bool:
        pass
    
    @abstractmethod
    def parameters(self, trajectory: Trajectory) -> dict:
        pass

class GaussianRandomWalkAnalysis(DiffusionAnalysis):
    def analyze(self, trajectory: Trajectory) -> bool:
        if len(trajectory.valid_entries) < 3:
            return False
        
        displacements = np.array(trajectory.displacements())
        dx = displacements[:, 0]
        dy = displacements[:, 1]
        
        # Kolmogorov-Smirnov test for normality
        from scipy.stats import kstest, norm
        _, p_x = kstest(dx, 'norm', args=(np.mean(dx), np.std(dx)))
        _, p_y = kstest(dy, 'norm', args=(np.mean(dy), np.std(dy)))
        return p_x > 0.05 and p_y > 0.05
    
    def parameters(self, trajectory: Trajectory) -> dict:
        displacements = np.array(trajectory.displacements())
        return {
            'dx_mean': np.mean(displacements[:, 0]),
            'dy_mean': np.mean(displacements[:, 1]),
            'dx_std': np.std(displacements[:, 0]),
            'dy_std': np.std(displacements[:, 1])
        }

class MSDLinearAnalysis(DiffusionAnalysis):
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
        return r_squared > 0.9
    
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