from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Self

from motility.tracking.bounding_box import BoundingBox


@dataclass(frozen=True, slots=True)
class Trajectory:
    """One cell through time: its box per frame, `None` when it is not tracked."""

    boxes: tuple[BoundingBox | None, ...]

    @classmethod
    def from_frames(
        cls, tracked_frames: Sequence[Sequence[BoundingBox]]
    ) -> tuple[Self, ...]:
        """Turn the boxes of each frame into one trajectory per track index."""
        boxes_per_track: dict[int, list[BoundingBox | None]] = {}
        for frame_index, frame in enumerate(tracked_frames):
            for box in frame:
                track = boxes_per_track.setdefault(
                    box.index, [None] * len(tracked_frames)
                )
                if box.tracked:
                    track[frame_index] = box
        return tuple(cls(tuple(boxes)) for _, boxes in sorted(boxes_per_track.items()))

    @property
    def seen(self) -> tuple[tuple[int, BoundingBox], ...]:
        return tuple(
            (frame_index, box)
            for frame_index, box in enumerate(self.boxes)
            if box is not None
        )

    @property
    def frames(self) -> tuple[int, ...]:
        return tuple(frame_index for frame_index, _ in self.seen)

    @property
    def positions(self) -> tuple[tuple[float, float], ...]:
        return tuple(box.centroid for _, box in self.seen)

    @property
    def displacements(self) -> tuple[tuple[float, float], ...]:
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
