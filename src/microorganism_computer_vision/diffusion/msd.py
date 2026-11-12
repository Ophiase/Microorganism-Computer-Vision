from dataclasses import dataclass

import numpy as np

from microorganism_computer_vision.tracking.trajectories import Trajectory


@dataclass(frozen=True, slots=True)
class DisplacementCurve:
    """The mean squared displacement at each lag, in pixels squared and frames."""

    squared: tuple[float, ...]
    lags: tuple[int, ...]


def mean_squared_displacement(
    trajectory: Trajectory, max_lag: int
) -> DisplacementCurve:
    """Average the squared displacement of every pair of points `lag` apart."""
    positions = trajectory.positions
    frames = trajectory.frames
    squared = []
    lags = []

    for lag in range(1, max(1, max_lag) + 1):
        if lag >= len(positions):
            break
        displacements = [
            (positions[index + lag][0] - positions[index][0]) ** 2
            + (positions[index + lag][1] - positions[index][1]) ** 2
            for index in range(len(positions) - lag)
        ]
        squared.append(float(np.mean(displacements)))
        lags.append(frames[lag] - frames[0])

    return DisplacementCurve(tuple(squared), tuple(lags))
