from typing import Final

import numpy as np
from scipy import stats

from motility.diffusion.behaviour import strided_turns
from motility.diffusion.evidence.result import Evidence, Hypothesis
from motility.tracking.trajectories import Trajectory

MIN_TURNS: Final = 8
SHARP: Final = np.pi / 3
MAD_TO_SIGMA: Final = 1.4826
SURROGATES: Final = 199
MIN_INTERVALS: Final = 4
MIN_STEP: Final = 0.5
MERGE: Final = 2
FLOOR: Final = 1e-9


def persistent(trajectory: Trajectory) -> Evidence | None:
    """Does the heading survive a step? Mean cos of the turns above 0, one-sided."""
    turns = _turns(trajectory)
    if turns is None:
        return None
    cosines = np.cos(turns)
    result = stats.ttest_1samp(cosines, 0.0, alternative="greater")
    return Evidence.of_test(
        Hypothesis.PERSISTENT,
        str(trajectory.index),
        result.statistic,
        result.pvalue,
        float(cosines.mean()),
        len(turns),
    )


def chiral(trajectory: Trajectory) -> Evidence | None:
    """Does it turn one way? Mean signed turn against 0, two-sided."""
    turns = _turns(trajectory)
    if turns is None or np.std(turns) < FLOOR:
        return None
    result = stats.ttest_1samp(turns, 0.0)
    return Evidence.of_test(
        Hypothesis.CHIRAL,
        str(trajectory.index),
        result.statistic,
        result.pvalue,
        float(turns.mean()),
        len(turns),
    )


def tumbling(trajectory: Trajectory) -> Evidence | None:
    """More sharp turns than rotational diffusion allows.

    The spread of ordinary turns is the median absolute deviation, which a few
    tumbles do not inflate; under a Gaussian of that spread, a turn beyond
    sixty degrees has a known probability, and the count of them is binomial.
    """
    turns = _turns(trajectory)
    if turns is None:
        return None
    spread = max(
        MAD_TO_SIGMA * float(np.median(np.abs(turns - np.median(turns)))), FLOOR
    )
    expected = float(min(2 * stats.norm.sf(SHARP / spread), 1.0))
    sharp = int(np.sum(np.abs(turns) > SHARP))
    result = stats.binomtest(sharp, len(turns), expected, alternative="greater")
    return Evidence.of_test(
        Hypothesis.TUMBLING,
        str(trajectory.index),
        sharp,
        result.pvalue,
        sharp / len(turns) - expected,
        len(turns),
    )


def periodic(trajectory: Trajectory, seed: int = 0) -> Evidence | None:
    """Sharp turns in a rhythm: their intervals more regular than chance allows.

    Under the null the sharp turns fall at random frames, a Bernoulli sequence:
    shuffling which frames hold them keeps their number and destroys any
    rhythm. The statistic is the coefficient of variation of the intervals
    between them, small when they keep time. A signed spectrum, the 2025 test,
    misses reversals, whose sign is random.
    """
    sharp = _sharp_frames(trajectory)
    if sharp is None or sharp.sum() < MIN_INTERVALS + 1:
        return None
    observed = _variation(sharp)
    generator = np.random.default_rng(seed + trajectory.index)
    shuffled = np.array([
        _variation(generator.permutation(sharp)) for _ in range(SURROGATES)
    ])
    p_value = (1 + np.sum(shuffled <= observed)) / (1 + SURROGATES)
    return Evidence.of_test(
        Hypothesis.PERIODIC,
        str(trajectory.index),
        observed,
        p_value,
        float(np.median(shuffled)) - observed,
        int(sharp.sum()),
    )


def _sharp_frames(trajectory: Trajectory) -> np.ndarray | None:
    """For each moving step, whether the turn into it is sharper than ninety degrees."""
    steps = np.diff(np.array(trajectory.positions).reshape(-1, 2), axis=0)
    moving = steps[np.hypot(steps[:, 0], steps[:, 1]) > MIN_STEP]
    if len(moving) < MIN_TURNS * 3:
        return None
    headings = np.arctan2(moving[:, 1], moving[:, 0])
    return np.abs(np.angle(np.exp(1j * np.diff(headings)))) > np.pi / 2


def _variation(sharp: np.ndarray) -> float:
    """The spread of the intervals between turns over their mean, after merging.

    A reversal blurred by noise turns sharply on two neighbouring frames: turns
    closer than `MERGE` frames count as one event.
    """
    frames = np.flatnonzero(sharp)
    events = (
        frames[np.concatenate(([True], np.diff(frames) > MERGE))]
        if len(frames)
        else frames
    )
    intervals = np.diff(events)
    if len(intervals) < 2 or intervals.mean() == 0:
        return np.inf
    return float(intervals.std() / intervals.mean())


def _turns(trajectory: Trajectory) -> np.ndarray | None:
    turns = strided_turns(np.array(trajectory.positions).reshape(-1, 2))
    return turns if len(turns) >= MIN_TURNS else None
