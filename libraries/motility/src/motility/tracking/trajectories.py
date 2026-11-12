from dataclasses import dataclass
from itertools import pairwise

from motility.tracking.geometry import Position


@dataclass(frozen=True, slots=True)
class Trajectory:
    """One cell through time: the frames it was seen at and its position at each."""

    index: int
    frames: tuple[int, ...]
    positions: tuple[Position, ...]

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def displacements(self) -> tuple[Position, ...]:
        return tuple(
            (next_x - x, next_y - y)
            for (x, y), (next_x, next_y) in pairwise(self.positions)
        )

    @property
    def time_deltas(self) -> tuple[int, ...]:
        return tuple(later - earlier for earlier, later in pairwise(self.frames))

    @property
    def average_speed(self) -> float:
        """Mean displacement per frame, in pixels per frame."""
        speeds = [
            (dx**2 + dy**2) ** 0.5 / delta
            for (dx, dy), delta in zip(
                self.displacements, self.time_deltas, strict=True
            )
            if delta != 0
        ]
        return sum(speeds) / len(speeds) if speeds else 0.0
