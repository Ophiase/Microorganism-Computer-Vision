from dataclasses import dataclass
from typing import Final

import numpy as np

from motility.vision.arrays import Frame

EXTREME: Final = 0.08
CANDIDATES: Final = 4
MIN_LENGTH: Final = 8
MAX_THICKNESS: Final = 6
MARGIN_SHARE: Final = 0.3


@dataclass(frozen=True, slots=True)
class ScaleBar:
    """A burnt-in scale bar: its left end, its row and its length in pixels.

    The number written beside it is for whoever reads the frame; with it, the
    pixel size is that number of micrometres over `length`.
    """

    x: int
    y: int
    length: int
    thickness: int


def find_scale_bars(median: Frame) -> tuple[ScaleBar, ...]:
    """Thin horizontal runs of uniform extreme intensity near the frame's border.

    A bar is drawn by software: a few rows of the same white or black, longer
    than they are thick, in a corner. Nothing grown by a cell looks like that.
    """
    height = median.shape[0]
    extreme = (median >= 1 - EXTREME) | (median <= EXTREME)
    runs = [
        (row, start, length)
        for row in range(height)
        for start, length in _runs(extreme[row])
        if length >= MIN_LENGTH and _near_border(row, start, length, median.shape)
    ]
    bars = [bar for bar in _merge_rows(runs) if _stands_out(bar, extreme)]
    return tuple(sorted(bars, key=lambda bar: -bar.length)[:CANDIDATES])


def _stands_out(bar: ScaleBar, extreme: np.ndarray) -> bool:
    """The rows just above and below the bar are not extreme: a bar is thin."""
    span = slice(bar.x, bar.x + bar.length)
    above, below = bar.y - 2, bar.y + bar.thickness + 1
    if above < 0 or below >= extreme.shape[0]:
        return False
    return bool(extreme[above, span].mean() < 0.5 and extreme[below, span].mean() < 0.5)


def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], row, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [
        (int(start), int(end - start))
        for start, end in zip(edges[::2], edges[1::2], strict=True)
    ]


def _near_border(row: int, start: int, length: int, shape: tuple[int, ...]) -> bool:
    height, width = shape
    vertical = row < height * MARGIN_SHARE or row > height * (1 - MARGIN_SHARE)
    horizontal = (
        start + length < width * MARGIN_SHARE
        or start > width * (1 - MARGIN_SHARE) - length
    )
    return vertical and horizontal and length < width / 3


def _merge_rows(runs: list[tuple[int, int, int]]) -> list[ScaleBar]:
    """Stack the runs of the same span on consecutive rows into one bar."""
    bars: list[ScaleBar] = []
    for row, start, length in runs:
        previous = next(
            (
                bar
                for bar in bars
                if bar.x == start
                and bar.length == length
                and bar.y + bar.thickness == row
            ),
            None,
        )
        if previous is None:
            bars.append(ScaleBar(start, row, length, 1))
            continue
        bars[bars.index(previous)] = ScaleBar(
            start, previous.y, length, previous.thickness + 1
        )
    return [bar for bar in bars if bar.thickness <= MAX_THICKNESS]
