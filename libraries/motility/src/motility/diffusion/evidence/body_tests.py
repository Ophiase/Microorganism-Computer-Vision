from typing import Final

import numpy as np
from scipy import stats

from motility.diffusion.evidence.result import Evidence, Hypothesis
from motility.tracking.trajectories import Trajectory

MIN_ELONGATION: Final = 1.4
MIN_STEPS: Final = 12
FLOOR: Final = 1e-12


def body_frame(trajectory: Trajectory) -> Evidence | None:
    """Kicks larger along the body than across it: an F-test on the steps' moments.

    Each one-frame step is projected on the body's axis measured at its start,
    so the axis comes from the shape and the step from the centres: the test
    does not measure itself. Under isotropic kicks the mean squares along and
    across are equal, their ratio F-distributed. A swimmer, moving along its
    axis, rejects it too: the model selection says which alternative fits.
    """
    if (
        len(trajectory.angles) != len(trajectory)
        or trajectory.elongation < MIN_ELONGATION
    ):
        return None
    frames = np.array(trajectory.frames)
    consecutive = np.flatnonzero(np.diff(frames) == 1)
    if len(consecutive) < MIN_STEPS:
        return None
    positions = np.array(trajectory.positions)
    steps = positions[consecutive + 1] - positions[consecutive]
    axes = np.array(trajectory.angles)[consecutive]
    along = steps[:, 0] * np.cos(axes) + steps[:, 1] * np.sin(axes)
    across = -steps[:, 0] * np.sin(axes) + steps[:, 1] * np.cos(axes)
    along_square, across_square = float(np.mean(along**2)), float(np.mean(across**2))
    if along_square < FLOOR or across_square < FLOOR:
        return None
    ratio = along_square / across_square
    count = len(steps)
    tail = (
        stats.f.sf(ratio, count, count)
        if ratio > 1
        else stats.f.cdf(ratio, count, count)
    )
    return Evidence.of_test(
        Hypothesis.BODY_FRAME,
        str(trajectory.index),
        ratio,
        min(2 * tail, 1.0),
        float(np.log(ratio)),
        count,
    )
