import numpy as np
from logic.diffusion import DiffusionTest
from logic.trajectories import Trajectory
from scipy.stats import kstest, norm

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
        p_x = self._get_p_value(dx) 
        p_y = self._get_p_value(dy) 
        return p_x > self.p_threshold and p_y > self.p_threshold
    

    def _get_p_value(self, data: np.ndarray) -> float:
        if np.std(data) == 0.0 or len(data) < 2:
            return 0.0
        try:
            _, p = kstest(data, 'norm', args=(np.mean(data), np.std(data)))
            return p
        except:
            return 0.0


    def parameters(self, trajectory: Trajectory) -> dict:
        displacements = np.array(trajectory.displacements())
        return {
            'dx_mean': np.mean(displacements[:, 0]),
            'dy_mean': np.mean(displacements[:, 1]),
            'dx_std': np.std(displacements[:, 0]),
            'dy_std': np.std(displacements[:, 1])
        }