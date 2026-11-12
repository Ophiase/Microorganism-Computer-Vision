from dataclasses import dataclass

import numpy as np
from scipy import stats

from motility.tracking.trajectories import Trajectory


@dataclass(frozen=True, slots=True)
class Turning:
    """How a track turns: its net signed turns, and how far from chance that is.

    `significance` is one minus the two-sided p-value of a mean turning angle of
    zero: a circling cell turns the same way at every step and is near one, a
    random walk or a straight swimmer turns both ways and is spread over [0, 1].
    """

    turns: float
    significance: float


def turning(trajectory: Trajectory, min_step: float, stride: int) -> Turning:
    """The turning between steps of `stride` observations, shorter steps left out.

    A step over several frames is long against the localisation noise, so its
    heading is the cell's and not the jitter's.
    """
    positions = np.array(trajectory.positions).reshape(-1, 2)[::stride]
    steps = np.diff(positions, axis=0)
    moving = steps[np.hypot(steps[:, 0], steps[:, 1]) > min_step]
    if len(moving) < 4:
        return Turning(0.0, 0.0)
    headings = np.arctan2(moving[:, 1], moving[:, 0])
    angles = np.angle(np.exp(1j * np.diff(headings)))
    spread = float(np.std(angles, ddof=1))
    if spread <= np.finfo(float).eps:
        return Turning(abs(float(np.sum(angles))) / (2 * np.pi), 1.0)
    statistic = float(np.mean(angles)) / (spread / np.sqrt(len(angles)))
    return Turning(
        abs(float(np.sum(angles))) / (2 * np.pi),
        1 - 2 * float(stats.norm.sf(abs(statistic))),
    )
