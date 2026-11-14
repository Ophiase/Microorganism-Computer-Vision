from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from motility.diffusion.evidence.result import Evidence, Hypothesis
from motility.tracking.trajectories import Trajectory

SURROGATES: Final = 199
PLACEMENTS: Final = 30

type Positions = NDArray[np.float64]


class Statistic(Protocol):
    """A number computed on a set of tracks, the distance map of the decor at hand."""

    def __call__(
        self, tracks: Sequence[Positions], distance: NDArray[np.float32], /
    ) -> float: ...


@dataclass(frozen=True, slots=True)
class SurrogateOutcome:
    """The statistic of the real tracks, and of each set of moved ones."""

    observed: float
    null: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SurrogateTest:
    """A statistic of the tracks against the same tracks moved across the decor.

    Each surrogate track is the real one turned about its centre and moved to
    a random place inside the frame: it keeps its shape, its speed and its
    turns, and loses its relation to the decor. It may cross an object; the
    statistics count only the positions outside them. Forbidding the crossing
    kept the surrogates far from the objects, and the test fired on cells that
    merely bounced off them, 40 % of the time. The p-value is the rank of the
    real statistic among the surrogates'.
    """

    statistic: Statistic
    alternative_greater: bool

    def run(
        self,
        trajectories: Sequence[Trajectory],
        objects: NDArray[np.bool_],
        seed: int = 0,
    ) -> SurrogateOutcome | None:
        tracks = [np.array(item.positions) for item in trajectories if len(item) > 1]
        if not tracks or not objects.any():
            return None
        distance = np.asarray(
            cv2.distanceTransform((~objects).astype(np.uint8), cv2.DIST_L2, 5),
            dtype=np.float32,
        )
        generator = np.random.default_rng(seed)
        null = np.array([
            self.statistic(
                [
                    _moved(track, (objects.shape[0], objects.shape[1]), generator)
                    for track in tracks
                ],
                distance,
            )
            for _ in range(SURROGATES)
        ])
        return SurrogateOutcome(self.statistic(tracks, distance), null)

    def evidence(
        self,
        hypothesis: Hypothesis,
        trajectories: Sequence[Trajectory],
        objects: NDArray[np.bool_],
    ) -> Evidence | None:
        outcome = self.run(trajectories, objects)
        if outcome is None:
            return None
        observed, null = outcome.observed, outcome.null
        beyond = null >= observed if self.alternative_greater else null <= observed
        p_value = (1 + int(np.sum(beyond))) / (1 + len(null))
        return Evidence.of_test(
            hypothesis,
            "population",
            observed,
            p_value,
            observed - float(np.mean(null)),
            len(trajectories),
        )


def near_share(radius: float) -> Statistic:
    """The share of all observations within `radius` px of an object."""

    def statistic(tracks: Sequence[Positions], distance: NDArray[np.float32]) -> float:
        values = np.concatenate([_sample(distance, track) for track in tracks])
        outside = values[values > 0]
        return float(np.mean(outside <= radius)) if len(outside) else 0.0

    return statistic


def speed_slope(tracks: Sequence[Positions], distance: NDArray[np.float32]) -> float:
    """The slope of each step's speed against its distance to the nearest object."""
    speeds, distances = [], []
    for track in tracks:
        speeds.append(np.hypot(*np.diff(track, axis=0).T))
        distances.append(_sample(distance, track[:-1]))
    x, y = np.concatenate(distances), np.concatenate(speeds)
    x, y = x[x > 0], y[x > 0]
    return float(np.polyfit(x, y, 1)[0]) if np.ptp(x) > 0 else 0.0


def _sample(distance: NDArray[np.float32], track: Positions) -> NDArray[np.float64]:
    height, width = distance.shape
    columns = np.clip(np.round(track[:, 0]).astype(int), 0, width - 1)
    rows = np.clip(np.round(track[:, 1]).astype(int), 0, height - 1)
    return distance[rows, columns].astype(np.float64)


def _moved(
    track: Positions, frame: tuple[int, int], generator: np.random.Generator
) -> Positions:
    """The track turned about its centre and placed at random inside the frame."""
    height, width = frame
    centred = track - track.mean(axis=0)
    for _ in range(PLACEMENTS):
        angle = generator.uniform(0, 2 * np.pi)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
        turned = centred @ rotation.T
        low, high = -turned.min(axis=0), np.array([width, height]) - turned.max(axis=0)
        if np.all(high > low):
            return turned + generator.uniform(low, high)
    return track
