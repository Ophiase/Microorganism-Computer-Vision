from collections.abc import Callable

import numpy as np

from logic.structure.bounding_box import BoundingBox
from logic.structure.rectangle import Rectangle

DEFAULT_SIZE = 20.0
SPAWN_RANGE = (100.0, 924.0)

Position = tuple[float, float]
Step = Callable[[Position], Position]


class SyntheticGenerator:
    """Walks a virtual cell through a frame and reports the box it would show.

    A cell that leaves the frame is reported as not tracked, and stays out of
    the trajectories the analysis reads.
    """

    def __init__(self, width: int = 1024, height: int = 1024) -> None:
        self.width = width
        self.height = height
        self.current_index = 0

    def brownian_motion(
        self, num_frames: int, speed: float, noise_level: float = 5.0
    ) -> list[BoundingBox]:
        def step(position: Position) -> Position:
            dx = np.random.normal(0, speed) + np.random.normal(0, noise_level)
            dy = np.random.normal(0, speed) + np.random.normal(0, noise_level)
            return (position[0] + dx, position[1] + dy)

        return self._walk(num_frames, step)

    def directed_motion(
        self, num_frames: int, speed: float, angle: float
    ) -> list[BoundingBox]:
        theta = np.radians(angle)
        velocity = (speed * np.cos(theta), speed * np.sin(theta))

        def step(position: Position) -> Position:
            noise = np.random.normal(0, 0.3, 2)
            return (
                position[0] + velocity[0] + noise[0],
                position[1] + velocity[1] + noise[1],
            )

        return self._walk(num_frames, step)

    def confined_motion(
        self, num_frames: int, center: Position, radius: float
    ) -> list[BoundingBox]:
        def step(_: Position) -> Position:
            dx = np.random.uniform(-radius, radius) + np.random.normal(0, 0.5)
            dy = np.random.uniform(-radius, radius) + np.random.normal(0, 0.5)
            return (center[0] + dx, center[1] + dy)

        return self._walk(num_frames, step)

    def ctrw_motion(self, num_frames: int, base_speed: float) -> list[BoundingBox]:
        wait_frames = 0

        def step(position: Position) -> Position:
            nonlocal wait_frames
            if wait_frames > 0:
                wait_frames -= 1
                return position
            dx = np.random.normal(0, base_speed) + np.random.normal(0, 0.3)
            dy = np.random.normal(0, base_speed) + np.random.normal(0, 0.3)
            wait_frames = int(np.random.poisson(5))
            return (position[0] + dx, position[1] + dy)

        return self._walk(num_frames, step)

    def sinusoidal_motion(
        self,
        num_frames: int,
        speed: float,
        amplitude: float,
        frequency: float,
        noise_level: float = 5.0,
    ) -> list[BoundingBox]:
        angle = np.random.uniform(0, 2 * np.pi)
        phase = 0.0

        def step(position: Position) -> Position:
            nonlocal phase
            dx = speed * np.cos(angle) + np.random.normal(0, noise_level)
            dy = speed * np.sin(angle) + np.random.normal(0, noise_level)
            perpendicular = angle + np.pi / 2
            dx += amplitude * np.sin(phase) * np.cos(perpendicular)
            dy += amplitude * np.sin(phase) * np.sin(perpendicular)
            phase += frequency * 2 * np.pi
            return (position[0] + dx, position[1] + dy)

        return self._walk(num_frames, step)

    def _walk(self, num_frames: int, step: Step) -> list[BoundingBox]:
        position = (
            float(np.random.uniform(*SPAWN_RANGE)),
            float(np.random.uniform(*SPAWN_RANGE)),
        )
        trajectory = []
        inside = True
        for _ in range(num_frames):
            if not inside:
                trajectory.append(self._box((0.0, 0.0), tracked=False))
                continue
            proposed = step(position)
            inside = self._inside(proposed)
            position = proposed if inside else position
            trajectory.append(self._box(proposed, tracked=inside))
        self.current_index += 1
        return trajectory

    def _inside(self, position: Position) -> bool:
        return 0 <= position[0] <= self.width and 0 <= position[1] <= self.height

    def _box(self, position: Position, tracked: bool = True) -> BoundingBox:
        half = DEFAULT_SIZE / 2
        return BoundingBox(
            Rectangle(
                round(position[0] - half),
                round(position[1] - half),
                round(DEFAULT_SIZE),
                round(DEFAULT_SIZE),
            ),
            index=self.current_index,
            tracked=tracked,
        )
