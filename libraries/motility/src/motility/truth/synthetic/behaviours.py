from enum import StrEnum
from typing import Final

import numpy as np
from numpy.typing import NDArray

from motility.diffusion.models.motion import Motion
from motility.truth.synthetic.motions import walk
from motility.truth.synthetic.scene import Kinetics

type Path2D = NDArray[np.float64]
"""Positions, one row per frame, x then y, in pixels."""

MEAN_RUN: Final = 15.0
TUMBLE_MEAN: Final = np.radians(68)
REVERSAL_PERIOD: Final = 10
HURST: Final = 0.3
WAITING_SHAPE: Final = 0.7
FLIGHT_SHAPE: Final = 1.5


class Behaviour(StrEnum):
    """The ways a cell may move, richer than the motion classes, to test the tests."""

    STATIONARY = "stationary"
    BROWNIAN = "brownian"
    SWIMMER = "swimmer"
    RUN_AND_TUMBLE = "run-and-tumble"
    RUN_AND_REVERSE = "run-and-reverse"
    CIRCLING = "circling"
    CONFINED = "confined"
    SUBDIFFUSIVE = "fbm-subdiffusive"
    WAITING = "ctrw-waiting"
    LEVY = "levy-walk"


def behave(
    behaviour: Behaviour,
    kinetics: Kinetics,
    frames: int,
    generator: np.random.Generator,
) -> Path2D:
    """A path of `frames` positions that moves as `behaviour` says, from the origin."""
    simple = {
        Behaviour.STATIONARY: Motion.STATIONARY,
        Behaviour.BROWNIAN: Motion.DIFFUSIVE,
        Behaviour.SWIMMER: Motion.DIRECTED,
        Behaviour.CIRCLING: Motion.CIRCULAR,
        Behaviour.CONFINED: Motion.CONFINED,
    }
    if behaviour in simple:
        return walk(
            simple[behaviour], kinetics, frames, (0.0, 0.0), generator
        ).positions
    match behaviour:
        case Behaviour.RUN_AND_TUMBLE:
            return _swim(_tumbling_headings(frames, generator), kinetics.speed)
        case Behaviour.RUN_AND_REVERSE:
            return _swim(_reversing_headings(frames, generator), kinetics.speed)
        case Behaviour.SUBDIFFUSIVE:
            return _fractional(frames, kinetics.diffusion, generator)
        case Behaviour.WAITING:
            return _waiting(frames, kinetics.diffusion, generator)
        case _:
            return _levy(frames, kinetics.speed, generator)


def _swim(headings: NDArray[np.float64], speed: float) -> Path2D:
    steps = speed * np.column_stack((np.cos(headings), np.sin(headings)))
    return np.vstack(([0.0, 0.0], np.cumsum(steps[:-1], axis=0)))


def _tumbling_headings(
    frames: int, generator: np.random.Generator
) -> NDArray[np.float64]:
    """Runs of exponential length; each tumble turns by about 68 degrees, either way."""
    tumbles = generator.random(frames) < 1 / MEAN_RUN
    angles = generator.normal(TUMBLE_MEAN, TUMBLE_MEAN / 3, frames) * generator.choice(
        (-1, 1), frames
    )
    drift = generator.normal(0, 0.05, frames)
    return generator.uniform(0, 2 * np.pi) + np.cumsum(np.where(tumbles, angles, drift))


def _reversing_headings(
    frames: int, generator: np.random.Generator
) -> NDArray[np.float64]:
    """A reversal every ten frames, give or take one, as run-reverse swimmers do."""
    periods = REVERSAL_PERIOD + generator.integers(-1, 2, frames)
    times = np.cumsum(periods)
    reversals = np.isin(np.arange(frames), times[times < frames])
    drift = generator.normal(0, 0.05, frames)
    return generator.uniform(0, 2 * np.pi) + np.cumsum(
        np.where(reversals, np.pi, drift)
    )


def _fractional(
    frames: int, diffusion: float, generator: np.random.Generator
) -> Path2D:
    """Fractional Brownian motion, Hurst 0.3 and alpha 0.6, by Cholesky factoring."""
    times = np.arange(1, frames, dtype=float)
    covariance = 0.5 * (
        times[:, None] ** (2 * HURST)
        + times[None, :] ** (2 * HURST)
        - np.abs(times[:, None] - times[None, :]) ** (2 * HURST)
    )
    factor = np.linalg.cholesky(covariance + 1e-9 * np.eye(len(times)))
    scale = np.sqrt(2 * diffusion)
    walked = factor @ generator.normal(0, scale, (len(times), 2))
    return np.vstack(([0.0, 0.0], walked))


def _waiting(frames: int, diffusion: float, generator: np.random.Generator) -> Path2D:
    """A continuous-time random walk: Gaussian jumps after heavy-tailed waits."""
    path, position, frame = np.zeros((frames, 2)), np.zeros(2), 0
    while frame < frames:
        wait = int(generator.pareto(WAITING_SHAPE)) + 1
        path[frame : frame + wait] = position
        frame += wait
        position = position + generator.normal(0, np.sqrt(2 * diffusion) * 2, 2)
    return path


def _levy(frames: int, speed: float, generator: np.random.Generator) -> Path2D:
    """A Lévy walk: straight flights at constant speed, their lengths heavy-tailed."""
    headings = np.empty(frames)
    frame = 0
    while frame < frames:
        flight = int(generator.pareto(FLIGHT_SHAPE) * 3) + 1
        headings[frame : frame + flight] = generator.uniform(0, 2 * np.pi)
        frame += flight
    return _swim(headings, speed)
