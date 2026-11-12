from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

from motility.tracking.observation import Observation
from motility.tracking.trajectories import Trajectory

Column = NDArray[np.generic]


@dataclass(frozen=True, slots=True)
class TrackTable:
    """Every observation of every track, one row each, as parallel columns.

    Positions are centres of mass in pixels, frames are indices in the clip;
    a box size of zero means the source gave none.
    """

    track: NDArray[np.int64]
    frame: NDArray[np.int64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    width: NDArray[np.int64]
    height: NDArray[np.int64]

    @classmethod
    def from_columns(cls, columns: Mapping[str, Column]) -> Self:
        rows = len(columns["track"])
        no_size = np.zeros(rows, dtype=np.int64)
        return cls(
            track=np.asarray(columns["track"], dtype=np.int64),
            frame=np.asarray(columns["frame"], dtype=np.int64),
            x=np.asarray(columns["x"], dtype=np.float64),
            y=np.asarray(columns["y"], dtype=np.float64),
            width=np.asarray(columns.get("width", no_size), dtype=np.int64),
            height=np.asarray(columns.get("height", no_size), dtype=np.int64),
        )

    @classmethod
    def from_observations(cls, observations: Sequence[Observation]) -> Self:
        return cls.from_columns({
            "track": np.array([item.track for item in observations]),
            "frame": np.array([item.frame for item in observations]),
            "x": np.array([item.detection.centroid[0] for item in observations]),
            "y": np.array([item.detection.centroid[1] for item in observations]),
            "width": np.array([
                item.detection.rectangle.width for item in observations
            ]),
            "height": np.array([
                item.detection.rectangle.height for item in observations
            ]),
        })

    @classmethod
    def concatenate(cls, tables: Sequence[Self]) -> Self:
        return cls.from_columns({
            name: np.concatenate([table.columns()[name] for table in tables])
            for name in COLUMNS
        })

    def columns(self) -> dict[str, Column]:
        return {name: getattr(self, name) for name in COLUMNS}

    def __len__(self) -> int:
        return len(self.track)

    @property
    def track_count(self) -> int:
        return len(np.unique(self.track))

    def trajectories(self, min_length: int = 1) -> tuple[Trajectory, ...]:
        """One trajectory per track with at least `min_length` observations."""
        if len(self) == 0:
            return ()
        ordered = self._sorted()
        tracks, starts = np.unique(ordered.track, return_index=True)
        ends = [*starts[1:], len(ordered)]
        return tuple(
            ordered._trajectory(int(track), slice(start, end))
            for track, start, end in zip(tracks, starts, ends, strict=True)
            if end - start >= min_length
        )

    def _sorted(self) -> Self:
        order = np.lexsort((self.frame, self.track))
        return type(self).from_columns({
            name: column[order] for name, column in self.columns().items()
        })

    def _trajectory(self, track: int, rows: slice) -> Trajectory:
        positions = zip(self.x[rows].tolist(), self.y[rows].tolist(), strict=True)
        return Trajectory(track, tuple(self.frame[rows].tolist()), tuple(positions))


COLUMNS = ("track", "frame", "x", "y", "width", "height")
