import numpy as np

from microorganism_computer_vision.diffusion.base import DiffusionTest
from microorganism_computer_vision.tracking.trajectories import Trajectory

MIN_ANGLES = 10
REVERSAL = np.pi / 2


class DirectionalSwitchTest(DiffusionTest):
    """Run-and-tumble behaviour: periodic reversals of the direction of travel.

    A reversal is a turn of more than ninety degrees between two displacements;
    the periodicity of the angles says whether the reversals are a rhythm or
    an accident.
    """

    def __init__(
        self, min_switches: int = 3, periodicity_threshold: float = 0.8
    ) -> None:
        super().__init__()
        self.min_switches = min_switches
        self.periodicity_threshold = periodicity_threshold

    def analyze(self, trajectory: Trajectory) -> bool:
        angles = self._angles(trajectory)
        if len(angles) < MIN_ANGLES:
            return False

        switch_count = self._switch_count(angles)
        return (
            switch_count >= self.min_switches
            and self._periodicity(angles) > self.periodicity_threshold
        )

    def parameters(self, trajectory: Trajectory) -> dict[str, float | list[float]]:
        angles = self._angles(trajectory)
        return {
            "switch_count": float(self._switch_count(angles)),
            "periodicity_strength": self._periodicity(angles),
        }

    @staticmethod
    def _angles(trajectory: Trajectory) -> list[float]:
        return [float(np.arctan2(dy, dx)) for dx, dy in trajectory.displacements]

    @staticmethod
    def _switch_count(angles: list[float]) -> int:
        return int(np.sum(np.abs(np.diff(angles)) > REVERSAL))

    @staticmethod
    def _periodicity(angles: list[float]) -> float:
        if not angles:
            return 0.0
        spectrum = np.abs(np.fft.fft(angles)[1:])
        return float(np.max(spectrum) / len(angles)) if len(spectrum) else 0.0
