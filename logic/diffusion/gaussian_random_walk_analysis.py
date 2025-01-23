import numpy as np
from logic.diffusion import DiffusionAnalysis
from logic.trajectories import Trajectory


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
