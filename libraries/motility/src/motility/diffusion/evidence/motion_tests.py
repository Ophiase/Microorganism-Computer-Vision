from typing import Final, Protocol

import numpy as np
from scipy import stats
from scipy.optimize import OptimizeResult, least_squares

from motility.diffusion.evidence.result import Evidence, Hypothesis
from motility.diffusion.models.curves import Model, fit, relative_variance
from motility.diffusion.models.selection import akaike_weights
from motility.diffusion.msd import mean_squared_displacement
from motility.tracking.trajectories import Trajectory

LAG_FRACTION: Final = 0.15
MIN_LAGS: Final = 4
MIN_STEPS: Final = 10
FLOOR: Final = 1e-9
ALPHA_MAX: Final = 2.5


class Law(Protocol):
    def __call__(self, coefficients: np.ndarray, /) -> np.ndarray: ...


def anomalous(trajectory: Trajectory) -> Evidence | None:
    """A power law against a straight line for the MSD, both with a noise offset.

    Localisation noise adds a constant to the MSD, which bends a log-log slope
    below one at short lags: a plain regression calls Brownian cells
    subdiffusive. Both laws carry that constant here, are fitted with Qian's
    variance, and their chi-square difference is chi-square with one degree of
    freedom under the null.
    """
    curve = mean_squared_displacement(trajectory, int(len(trajectory) * LAG_FRACTION))
    if len(curve.lags) < MIN_LAGS + 1:
        return None
    lags, squared = np.array(curve.lags, dtype=float), np.array(curve.squared)
    positions = trajectory.frames[-1] - trajectory.frames[0] + 1
    deviation = np.sqrt(relative_variance(lags, positions))
    linear = _fit(lambda c: 4 * c[0] * lags + c[1], squared, deviation, [1.0, 0.1])
    power = _fit(
        lambda c: 4 * c[0] * lags ** c[1] + c[2],
        squared,
        deviation,
        [1.0, 1.0, 0.1],
        upper=[np.inf, ALPHA_MAX, np.inf],
    )
    statistic = max(linear.cost * 2 - power.cost * 2, 0.0)
    return Evidence.of_test(
        Hypothesis.ANOMALOUS,
        str(trajectory.index),
        statistic,
        stats.chi2.sf(statistic, 1),
        float(power.x[1]) - 1,
        len(lags),
    )


def _fit(
    law: Law,
    squared: np.ndarray,
    deviation: np.ndarray,
    start: list[float],
    upper: list[float] | None = None,
) -> OptimizeResult:
    """Least squares of log(law) on log(MSD), each lag by its expected deviation."""
    scale = max(float(squared[0]), FLOOR)
    first = np.array(start) * np.array([scale] + [1.0] * (len(start) - 2) + [scale])
    bounds = (np.full(len(start), FLOOR), np.array(upper or [np.inf] * len(start)))
    return least_squares(
        lambda c: (
            (np.log(np.maximum(law(c), FLOOR)) - np.log(np.maximum(squared, FLOOR)))
            / deviation
        ),
        np.clip(first, bounds[0] * 2, None),
        bounds=bounds,
    )


def non_gaussian(trajectory: Trajectory) -> Evidence | None:
    """Shapiro-Wilk on the steps of one frame, each axis standardised, pooled."""
    steps = np.array(trajectory.displacements).reshape(-1, 2)[
        np.array(trajectory.time_deltas) == 1
    ]
    if len(steps) < MIN_STEPS:
        return None
    deviation = steps.std(axis=0)
    if np.any(deviation < FLOOR):
        return None
    pooled = ((steps - steps.mean(axis=0)) / deviation).ravel()
    result = stats.shapiro(pooled)
    kurtosis = float(stats.kurtosis(pooled))
    return Evidence.of_test(
        Hypothesis.NON_GAUSSIAN,
        str(trajectory.index),
        result.statistic,
        result.pvalue,
        kurtosis,
        len(pooled),
    )


def confined(trajectory: Trajectory) -> Evidence | None:
    """The Akaike weight of a saturating MSD against a linear one: a weight, not a p."""
    curve = mean_squared_displacement(trajectory, int(len(trajectory) * LAG_FRACTION))
    if len(curve.lags) < MIN_LAGS + 1:
        return None
    lags, squared = np.array(curve.lags, dtype=float), np.array(curve.squared)
    positions = trajectory.frames[-1] - trajectory.frames[0] + 1
    fits = {
        model: fit(model, lags, squared, positions)
        for model in (Model.LINEAR, Model.SATURATING)
    }
    weight = akaike_weights(fits)[Model.SATURATING]
    return Evidence.of_test(
        Hypothesis.CONFINED, str(trajectory.index), weight, None, weight, len(lags)
    )
