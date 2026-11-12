from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np

from motility.tracking.trajectories import Trajectory

STRIDE: Final = 3
REORIENTATION: Final = np.pi / 3
MIN_STEP: Final = 1.0


@dataclass(frozen=True, slots=True)
class Behaviour:
    """What a track does beyond its motion class, in pixels and frames.

    `angular_velocity` is signed: positive turns from the image's x axis
    towards its y axis, clockwise as the image is shown, y pointing down.
    `radius` is the speed over the angular velocity, the circle it would
    swim. `reorientations` are turns sharper than sixty degrees between two
    moving steps of `STRIDE` frames, a tumble's signature.
    """

    track: int
    path: float
    straightness: float
    angular_velocity: float
    radius: float | None
    reorientations: int
    duration: int


def behaviour(trajectory: Trajectory) -> Behaviour:
    positions = np.array(trajectory.positions).reshape(-1, 2)
    path = float(np.sum(np.hypot(*np.diff(positions, axis=0).T)))
    net = float(np.hypot(*(positions[-1] - positions[0]))) if len(positions) else 0.0
    duration = trajectory.frames[-1] - trajectory.frames[0] if trajectory.frames else 0
    turns = _turns(positions[::STRIDE])
    angular = float(np.sum(turns)) / (len(turns) * STRIDE) if len(turns) else 0.0
    speed = path / max(duration, 1)
    return Behaviour(
        track=trajectory.index,
        path=path,
        straightness=net / path if path > 0 else 0.0,
        angular_velocity=angular,
        radius=speed / abs(angular) if abs(angular) > 1e-6 else None,
        reorientations=int(np.sum(np.abs(turns) > REORIENTATION)),
        duration=duration,
    )


def _turns(positions: np.ndarray) -> np.ndarray:
    """Signed angles between successive moving steps, in (-pi, pi]."""
    steps = np.diff(positions, axis=0)
    moving = steps[np.hypot(steps[:, 0], steps[:, 1]) > MIN_STEP * STRIDE]
    if len(moving) < 2:
        return np.zeros(0)
    headings = np.arctan2(moving[:, 1], moving[:, 0])
    return np.angle(np.exp(1j * np.diff(headings)))


def distance_to_objects(trajectory: Trajectory, objects: np.ndarray) -> float | None:
    """Median distance in pixels from the track to the nearest static object."""
    if not objects.any():
        return None
    distance = cv2.distanceTransform((~objects).astype(np.uint8), cv2.DIST_L2, 5)
    height, width = objects.shape
    samples = [
        distance[min(max(round(y), 0), height - 1), min(max(round(x), 0), width - 1)]
        for x, y in trajectory.positions
    ]
    return float(np.median(samples))
