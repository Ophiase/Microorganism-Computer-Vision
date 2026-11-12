from dataclasses import dataclass

import numpy as np
from scipy import stats

from motility.diffusion.msd import DisplacementCurve

CONFIDENCE = 0.95
FLOOR = 1e-9


@dataclass(frozen=True, slots=True)
class Exponent:
    """The anomalous exponent alpha of MSD ~ t^alpha, with a 95 % interval.

    The interval comes from the regression alone: the MSD lags are correlated,
    so it is narrower than the truth, a lower bound on the uncertainty.
    """

    value: float
    low: float
    high: float


def anomalous_exponent(curve: DisplacementCurve) -> Exponent | None:
    """The slope of log MSD against log lag, or None below three usable lags."""
    squared = np.array(curve.squared)
    lags = np.array(curve.lags, dtype=float)
    usable = squared > FLOOR
    if usable.sum() < 3:
        return None
    regression = stats.linregress(np.log(lags[usable]), np.log(squared[usable]))
    margin = stats.t.ppf((1 + CONFIDENCE) / 2, usable.sum() - 2) * regression.stderr
    slope = float(regression.slope)
    return Exponent(slope, slope - float(margin), slope + float(margin))
