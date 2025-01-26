from typing import List
import numpy as np
from logic.diffusion.base import DiffusionTest
from logic.trajectories import Trajectory


class DirectionalSwitchTest(DiffusionTest):
    """
    Detects periodic direction reversal patterns.
    Common in some bacterial run-and-tumble behaviors.
    """
    def __init__(self, min_switches: int = 3, periodicity_threshold: float = 0.8):
        super().__init__()
        self.min_switches = min_switches
        self.periodicity_threshold = periodicity_threshold

    def analyze(self, trajectory: Trajectory) -> bool:
        angles = self._get_direction_angles(trajectory)
        if len(angles) < 10:
            return False
            
        # Find dominant frequency
        fft = np.fft.fft(angles)
        frequencies = np.fft.fftfreq(len(angles))
        dominant = np.argmax(np.abs(fft[1:])) + 1  # Ignore DC component
        
        # Check periodicity strength
        period_strength = np.abs(fft[dominant])/len(angles)
        return (self._count_switches(angles) >= self.min_switches and 
                period_strength > self.periodicity_threshold)

    def _get_direction_angles(self, trajectory: Trajectory) -> List[float]:
        displacements = trajectory.displacements()
        return [np.arctan2(dy, dx) for dx, dy in displacements]

    def _count_switches(self, angles: List[float]) -> int:
        angle_changes = np.diff(angles)
        return np.sum(np.abs(angle_changes) > np.pi/2)  # >90° changes

    def parameters(self, trajectory: Trajectory) -> dict:
        angles = self._get_direction_angles(trajectory)
        fft = np.fft.fft(angles)
        return {
            'switch_count': self._count_switches(angles),
            'periodicity_strength': np.max(np.abs(fft[1:]))/len(angles)
        }