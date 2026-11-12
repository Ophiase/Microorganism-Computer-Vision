from dataclasses import dataclass

import numpy as np

from motility.tracking.trajectories import Trajectory


@dataclass(frozen=True, slots=True)
class DisplacementCurve:
    """The mean squared displacement at each lag, in pixels squared and frames."""

    squared: tuple[float, ...]
    lags: tuple[int, ...]


def mean_squared_displacement(
    trajectory: Trajectory, max_lag: int
) -> DisplacementCurve:
    """Average the squared displacement of every pair of observations `lag` apart.

    A frame the track missed leaves a gap: the pairs across it are measured on
    their true lag, never on their rank in the track.
    """
    if not trajectory.frames:
        return DisplacementCurve((), ())
    first = trajectory.frames[0]
    dense = np.full((trajectory.frames[-1] - first + 1, 2), np.nan)
    dense[np.array(trajectory.frames) - first] = trajectory.positions

    squared = []
    lags = []
    for lag in range(1, min(max(1, max_lag), len(dense) - 1) + 1):
        steps = dense[lag:] - dense[:-lag]
        distances = np.sum(steps**2, axis=1)
        if np.all(np.isnan(distances)):
            continue
        squared.append(float(np.nanmean(distances)))
        lags.append(lag)
    return DisplacementCurve(tuple(squared), tuple(lags))
