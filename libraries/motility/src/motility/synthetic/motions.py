from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from motility.diffusion.models.motion import Motion
from motility.synthetic.scene import Kinetics
from motility.tracking.geometry import Position

RESTORING = 0.2
SWIMMING_NOISE = 0.2
SLOWEST_SHARE = 0.2


@dataclass(frozen=True, slots=True)
class Path:
    """Where one synthetic cell is at every frame, and where it points."""

    motion: Motion
    positions: NDArray[np.float64]
    headings: NDArray[np.float64]


def walk(
    motion: Motion,
    kinetics: Kinetics,
    frames: int,
    start: Position,
    generator: np.random.Generator,
) -> Path:
    """The path of a cell of the given motion, from `start`, for `frames` frames."""
    heading = generator.uniform(0, 2 * np.pi)
    match motion:
        case Motion.STATIONARY:
            offsets = generator.normal(0, kinetics.jitter, (frames, 2))
            return _still(motion, np.array(start) + offsets, heading)
        case Motion.DIFFUSIVE:
            steps = generator.normal(0, np.sqrt(2 * kinetics.diffusion), (frames, 2))
            steps[0] = 0
            return _still(motion, np.array(start) + np.cumsum(steps, axis=0), heading)
        case Motion.CONFINED:
            return _still(motion, _well(kinetics, frames, start, generator), heading)
        case Motion.DIRECTED:
            turns = generator.normal(
                0, np.sqrt(2 * kinetics.rotational_diffusion), frames
            )
            return _swim(motion, kinetics, start, heading + np.cumsum(turns), generator)
        case Motion.CIRCULAR:
            sense = generator.choice((-1.0, 1.0))
            drift = sense * kinetics.angular_speed * np.arange(frames)
            return _swim(motion, kinetics, start, heading + drift, generator)
        case Motion.UNDETERMINED:
            raise ValueError("no cell moves in an undetermined way")


def _still(motion: Motion, positions: NDArray[np.float64], heading: float) -> Path:
    return Path(motion, positions, np.full(len(positions), heading))


def _well(
    kinetics: Kinetics, frames: int, start: Position, generator: np.random.Generator
) -> NDArray[np.float64]:
    """An Ornstein-Uhlenbeck walk whose spread settles at the confinement radius."""
    centre = np.array(start)
    deviation = kinetics.confinement * np.sqrt(1 - (1 - RESTORING) ** 2)
    positions = np.empty((frames, 2))
    positions[0] = centre
    for frame in range(1, frames):
        pull = -RESTORING * (positions[frame - 1] - centre)
        positions[frame] = (
            positions[frame - 1] + pull + generator.normal(0, deviation, 2)
        )
    return positions


def _swim(
    motion: Motion,
    kinetics: Kinetics,
    start: Position,
    headings: NDArray[np.float64],
    generator: np.random.Generator,
) -> Path:
    speed = kinetics.speed * max(
        1 + kinetics.speed_spread * generator.normal(), SLOWEST_SHARE
    )
    steps = speed * np.column_stack((np.cos(headings), np.sin(headings)))
    steps += generator.normal(0, SWIMMING_NOISE, steps.shape)
    steps[0] = 0
    return Path(motion, np.array(start) + np.cumsum(steps, axis=0), headings)
