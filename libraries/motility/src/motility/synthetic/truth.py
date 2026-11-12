from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np

from motility.diffusion.models.motion import Motion
from motility.synthetic.motions import Path
from motility.tracking.table import TrackTable


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """Where every synthetic cell is while it is in the frame, and how it moves.

    The cell's index in the population is its track number in `table`.
    """

    table: TrackTable
    motions: dict[int, Motion]

    @classmethod
    def from_paths(cls, paths: Sequence[Path], width: int, height: int) -> Self:
        tables = [
            _visible_part(index, path, width, height)
            for index, path in enumerate(paths)
        ]
        motions = {index: path.motion for index, path in enumerate(paths)}
        return cls(TrackTable.concatenate(tables), motions)


def _visible_part(index: int, path: Path, width: int, height: int) -> TrackTable:
    x, y = path.positions[:, 0], path.positions[:, 1]
    frames = np.flatnonzero((x >= 0) & (x < width) & (y >= 0) & (y < height))
    return TrackTable.from_columns({
        "track": np.full(len(frames), index),
        "frame": frames,
        "x": x[frames],
        "y": y[frames],
    })
