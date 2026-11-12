from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from motility.tracking.table import TrackTable

UNREACHABLE = 1e9


@dataclass(frozen=True, slots=True)
class FrameMatch:
    """Which predicted track each true cell was paired with at one frame."""

    frame: int
    pairs: tuple[tuple[int, int], ...]
    distances: tuple[float, ...]
    truths: int
    predictions: int


def match_frames(
    truth: TrackTable, predicted: TrackTable, radius: float
) -> tuple[FrameMatch, ...]:
    """Pair true and predicted positions frame by frame, within `radius` pixels.

    Each frame is an independent assignment, solved by the Hungarian algorithm
    on the distances: a simplification of CLEAR MOT, which would prefer to keep
    the pairs of the previous frame.
    """
    frames = np.union1d(truth.frame, predicted.frame)
    return tuple(_match(truth, predicted, int(frame), radius) for frame in frames)


def _match(
    truth: TrackTable, predicted: TrackTable, frame: int, radius: float
) -> FrameMatch:
    true_rows = np.flatnonzero(truth.frame == frame)
    predicted_rows = np.flatnonzero(predicted.frame == frame)
    if len(true_rows) == 0 or len(predicted_rows) == 0:
        return FrameMatch(frame, (), (), len(true_rows), len(predicted_rows))
    distance = cdist(
        np.column_stack((truth.x[true_rows], truth.y[true_rows])),
        np.column_stack((predicted.x[predicted_rows], predicted.y[predicted_rows])),
    )
    cost = np.where(distance <= radius, distance, UNREACHABLE)
    rows, columns = linear_sum_assignment(cost)
    kept = [
        (row, column)
        for row, column in zip(rows, columns, strict=True)
        if cost[row, column] <= radius
    ]
    return FrameMatch(
        frame,
        tuple(
            (
                int(truth.track[true_rows[row]]),
                int(predicted.track[predicted_rows[column]]),
            )
            for row, column in kept
        ),
        tuple(float(distance[row, column]) for row, column in kept),
        len(true_rows),
        len(predicted_rows),
    )


def majority_truth(matches: tuple[FrameMatch, ...]) -> dict[int, int]:
    """The true cell each predicted track was most often paired with."""
    votes: dict[int, Counter[int]] = {}
    for match in matches:
        for true_cell, track in match.pairs:
            votes.setdefault(track, Counter())[true_cell] += 1
    return {track: counter.most_common(1)[0][0] for track, counter in votes.items()}
