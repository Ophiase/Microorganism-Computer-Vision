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
    """A rhythm in how much it turns: the periodogram's peak against shuffles.

    The 2025 test took the Fourier peak of the signed headings to a fixed
    threshold. A reversal turns by plus or minus pi at random, so signed turns
    hide the rhythm: here it is sought in their size, frame by frame, and the
    threshold is what the same sizes reach once their order is destroyed.
    """
    steps = np.diff(np.array(trajectory.positions).reshape(-1, 2), axis=0)
    headings = np.arctan2(steps[:, 1], steps[:, 0])
    turns = np.abs(np.angle(np.exp(1j * np.diff(headings))))
    if len(turns) < MIN_TURNS * 3:
        return None
    observed = _peak(turns)
    generator = np.random.default_rng(seed + trajectory.index)
    shuffled = np.array([
        _peak(generator.permutation(turns)) for _ in range(SURROGATES)
    ])
    p_value = (1 + np.sum(shuffled >= observed)) / (1 + SURROGATES)
    return Evidence.of_test(
        Hypothesis.PERIODIC,
        str(trajectory.index),
        observed,
        p_value,
        observed / max(float(np.median(shuffled)), FLOOR),
        len(turns),
    )


def _turns(trajectory: Trajectory) -> np.ndarray | None:
    turns = strided_turns(np.array(trajectory.positions).reshape(-1, 2))
    return turns if len(turns) >= MIN_TURNS else None


def _peak(turns: np.ndarray) -> float:
    centred = turns - turns.mean()
    return float(np.max(np.abs(np.fft.rfft(centred))[1:]) ** 2 / len(turns))
