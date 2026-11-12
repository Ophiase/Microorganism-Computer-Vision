from dataclasses import dataclass

import numpy as np

from motility.diffusion.models.curves import Fit, Model, fit
from motility.diffusion.models.exponent import Exponent, anomalous_exponent
from motility.diffusion.models.motion import Motion
from motility.diffusion.models.parameters import ClassificationParameters
from motility.diffusion.models.turning import Turning, turning
from motility.diffusion.msd import mean_squared_displacement
from motility.tracking.trajectories import Trajectory


@dataclass(frozen=True, slots=True)
class Verdict:
    """The motion given to one trajectory, how sure, and the numbers behind it.

    `weights` are the Akaike weights of the four MSD models; `confidence` is the
    weight of the chosen one, or the turning significance for a circular track.
    Diffusion is in pixels squared per frame, speed in pixels per frame.
    """

    track: int
    length: int
    motion: Motion
    confidence: float
    weights: dict[Model, float]
    exponent: Exponent | None
    diffusion: float | None
    speed: float
    turning: Turning


def classify(trajectory: Trajectory, parameters: ClassificationParameters) -> Verdict:
    """Fit the four MSD models, keep the AICc's choice, then test for circling."""
    curve = mean_squared_displacement(
        trajectory, max_lag=int(len(trajectory) * parameters.lag_fraction)
    )
    track_turning = turning(trajectory, parameters.min_step, parameters.turning_stride)
    undetermined = Verdict(
        trajectory.index,
        len(trajectory),
        Motion.UNDETERMINED,
        0.0,
        {},
        None,
        None,
        trajectory.average_speed,
        track_turning,
    )
    if len(trajectory) < parameters.min_track_length or len(curve.lags) < (
        parameters.min_lags
    ):
        return undetermined

    lags = np.array(curve.lags, dtype=float)
    squared = np.array(curve.squared)
    positions = trajectory.frames[-1] - trajectory.frames[0] + 1
    fits = {model: fit(model, lags, squared, positions) for model in Model}
    weights = akaike_weights(fits)
    best = max(weights, key=weights.__getitem__)
    motion, confidence = best.motion, weights[best]
    if best is not Model.CONSTANT and _circles(track_turning, parameters):
        motion, confidence = Motion.CIRCULAR, track_turning.significance

    return Verdict(
        trajectory.index,
        len(trajectory),
        motion,
        confidence,
        weights,
        anomalous_exponent(curve),
        _diffusion(fits),
        trajectory.average_speed,
        track_turning,
    )


def akaike_weights(fits: dict[Model, Fit]) -> dict[Model, float]:
    """exp(-delta/2) of each model's AICc, normalised to sum to one."""
    scores = np.array([item.aicc for item in fits.values()])
    relative = np.exp(-(scores - np.min(scores)) / 2)
    return dict(zip(fits, (relative / relative.sum()).tolist(), strict=True))


def _circles(track_turning: Turning, parameters: ClassificationParameters) -> bool:
    return (
        track_turning.turns >= parameters.circular_turns
        and track_turning.significance >= parameters.circular_significance
    )


def _diffusion(fits: dict[Model, Fit]) -> float:
    return fits[Model.LINEAR].coefficients[0]
