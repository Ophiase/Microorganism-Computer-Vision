from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from motility.diffusion.evidence.context_tests import SurrogateTest, near_share
from motility.diffusion.evidence.result import Hypothesis
from motility.tracking.trajectories import Trajectory

SIZE: Final = 300
OBSTACLES: Final = ((80.0, 90.0), (210.0, 120.0), (140.0, 230.0))
RADIUS: Final = 25.0
SWIMMERS: Final = 20
FRAMES: Final = 150
SPEED: Final = 2.0
EDGE: Final = 6.0
HUG: Final = 0.3


def obstacles() -> NDArray[np.bool_]:
    region = np.zeros((SIZE, SIZE), dtype=np.uint8)
    for x, y in OBSTACLES:
        cv2.circle(region, (int(x), int(y)), int(RADIUS), 1, -1)
    return region.astype(bool)


def swimmers(follow: bool, generator: np.random.Generator) -> list[Trajectory]:
    """Swimmers among the obstacles: following their edges, or bouncing off them."""
    region = obstacles()
    tracks = []
    for index in range(SWIMMERS):
        position = _free_point(region, generator)
        heading, positions = generator.uniform(0, 2 * np.pi), []
        for _ in range(FRAMES):
            heading = _steer(position, heading, follow) + generator.normal(0, 0.1)
            step = position + SPEED * np.array([np.cos(heading), np.sin(heading)])
            if not _free(step, region):
                heading += np.pi
                continue
            position = step
            positions.append(tuple(position.tolist()))
        tracks.append(Trajectory(index, tuple(range(len(positions))), tuple(positions)))
    return tracks


def rejection_rate(follow: bool, populations: int = 40, seed: int = 0) -> float:
    """How often the trapping test rejects its null over many simulated videos."""
    generator = np.random.default_rng(seed)
    test = SurrogateTest(near_share(EDGE), alternative_greater=True)
    region = obstacles()
    rejected = 0
    for _ in range(populations):
        evidence = test.evidence(
            Hypothesis.TRAPPED, swimmers(follow, generator), region
        )
        rejected += evidence is not None and (evidence.p_value or 1.0) < 0.05
    return rejected / populations


def _steer(position: NDArray[np.float64], heading: float, follow: bool) -> float:
    if not follow:
        return heading
    for x, y in OBSTACLES:
        offset = position - np.array([x, y])
        if np.hypot(*offset) - RADIUS < EDGE:
            tangent = np.arctan2(offset[1], offset[0]) + np.pi / 2 + HUG
            return heading + 0.4 * float(np.angle(np.exp(1j * (tangent - heading))))
    return heading


def _free(point: NDArray[np.float64], region: NDArray[np.bool_]) -> bool:
    x, y = int(point[0]), int(point[1])
    return 0 <= x < SIZE and 0 <= y < SIZE and not region[y, x]


def _free_point(
    region: NDArray[np.bool_], generator: np.random.Generator
) -> NDArray[np.float64]:
    while True:
        point = generator.uniform(0, SIZE, 2)
        if _free(point, region):
            return point
