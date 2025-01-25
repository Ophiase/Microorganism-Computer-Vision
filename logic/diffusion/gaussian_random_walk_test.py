import numpy as np
from logic.diffusion import DiffusionTest
from logic.trajectories import Trajectory


class GaussianRandomWalkTest(DiffusionTest):
    """
    Tests if particle displacements follow a Gaussian random walk by checking
    normality of coordinate displacements using Kolmogorov-Smirnov test.
    """
    
    def __init__(self, p_threshold: float = 0.05):
        super().__init__()
        self.p_threshold = p_threshold

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
        return p_x > self.p_threshold and p_y > self.p_threshold

    def parameters(self, trajectory: Trajectory) -> dict:
        displacements = np.array(trajectory.displacements())
        return {
            'dx_mean': np.mean(displacements[:, 0]),
            'dy_mean': np.mean(displacements[:, 1]),
            'dx_std': np.std(displacements[:, 0]),
            'dy_std': np.std(displacements[:, 1])
        }