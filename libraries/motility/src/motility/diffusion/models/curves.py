from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from motility.diffusion.models.motion import Motion

FloatArray = NDArray[np.float64]
FLOOR = 1e-9


class Model(StrEnum):
    """A law of the MSD in the lag, each with the motion it stands for."""

    CONSTANT = "constant"
    LINEAR = "linear"
    BALLISTIC = "ballistic"
    SATURATING = "saturating"

    @property
    def motion(self) -> Motion:
        return {
            Model.CONSTANT: Motion.STATIONARY,
            Model.LINEAR: Motion.DIFFUSIVE,
            Model.BALLISTIC: Motion.DIRECTED,
            Model.SATURATING: Motion.CONFINED,
        }[self]


@dataclass(frozen=True, slots=True)
class Fit:
    """A model fitted to an MSD curve on log residuals, with its corrected AIC."""

    model: Model
    coefficients: tuple[float, ...]
    aicc: float


def predict(model: Model, coefficients: FloatArray, lags: FloatArray) -> FloatArray:
    """The MSD of a model at each lag; the offset is the localisation noise.

    constant: c. linear: 4 D t + c. ballistic: 4 D t + v^2 t^2 + c.
    saturating: L^2 (1 - exp(-t / T)) + c.
    """
    match model:
        case Model.CONSTANT:
            return np.full_like(lags, coefficients[0])
        case Model.LINEAR:
            return 4 * coefficients[0] * lags + coefficients[1]
        case Model.BALLISTIC:
            diffusion, speed, offset = coefficients
            return 4 * diffusion * lags + speed**2 * lags**2 + offset
        case Model.SATURATING:
            plateau, time, offset = coefficients
            return plateau * (1 - np.exp(-lags / time)) + offset


def fit(model: Model, lags: FloatArray, squared: FloatArray, positions: int) -> Fit:
    """Weighted least squares on log(MSD), each lag by its expected relative error.

    `positions` is the number of frames the curve was measured over: the long
    lags average fewer pairs, so they weigh less.
    """
    start = np.maximum(_start(model, lags, squared), 2 * FLOOR)
    deviation = np.sqrt(relative_variance(lags, positions))
    result = least_squares(
        lambda coefficients: (
            _log_residuals(model, coefficients, lags, squared) / deviation
        ),
        start,
        bounds=(np.full(len(start), FLOOR), np.full(len(start), np.inf)),
    )
    return Fit(
        model, tuple(float(value) for value in result.x), _aicc(result.fun, len(start))
    )


def relative_variance(lags: FloatArray, positions: int) -> FloatArray:
    """Variance of MSD(n) over MSD(n)^2 for a random walk of N positions (Qian 1991).

    (2 n^2 + 1) / (3 n (N - n + 1)), the variance a time-averaged MSD has when
    its pairs overlap.
    """
    return (2 * lags**2 + 1) / (3 * lags * np.maximum(positions - lags + 1, 1))


def _log_residuals(
    model: Model, coefficients: FloatArray, lags: FloatArray, squared: FloatArray
) -> FloatArray:
    predicted = np.maximum(predict(model, coefficients, lags), FLOOR)
    return np.log(predicted) - np.log(np.maximum(squared, FLOOR))


def _start(model: Model, lags: FloatArray, squared: FloatArray) -> FloatArray:
    slope = max((squared[-1] - squared[0]) / max(lags[-1] - lags[0], 1.0), FLOOR)
    offset = max(squared[0] / 2, FLOOR)
    match model:
        case Model.CONSTANT:
            return np.array([max(float(np.mean(squared)), FLOOR)])
        case Model.LINEAR:
            return np.array([slope / 4, offset])
        case Model.BALLISTIC:
            return np.array([slope / 8, np.sqrt(slope / lags[-1]), offset])
        case Model.SATURATING:
            return np.array([max(squared[-1], FLOOR), lags[-1] / 3, offset])


def _aicc(residuals: FloatArray, parameters: int) -> float:
    """Chi-square plus the small-sample corrected penalty; residuals are weighted."""
    points = len(residuals)
    aic = float(np.sum(residuals**2)) + 2 * parameters
    denominator = points - parameters - 1
    correction = (
        2 * parameters * (parameters + 1) / denominator if denominator > 0 else np.inf
    )
    return float(aic + correction)
