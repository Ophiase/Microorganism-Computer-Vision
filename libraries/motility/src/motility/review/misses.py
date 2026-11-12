from collections.abc import Sequence
from typing import Final

import numpy as np
from scipy.spatial import KDTree

from motility.pipeline.parameters import PipelineParameters
from motility.review.suspects import Suspect, SuspectKind
from motility.tracking.kalman_filter import KalmanTracker
from motility.tracking.table import TrackTable
from motility.vision.arrays import Video
from motility.vision.detection import DetectionParameters, detect

RELAX: Final = 0.6
MIN_FRAMES: Final = 5
UNCOVERED: Final = 0.8
MOVING: Final = 2.0


def relaxed_tracks(foreground: Video, parameters: PipelineParameters) -> TrackTable:
    """Tracks of a looser detection: a lower cut, half the smallest area, no opening.

    What it finds and the pipeline did not is where a cell may have been missed.
    """
    detection = _relaxed(parameters.detection)
    tracker = KalmanTracker(parameters.tracking)
    observations = [
        observation
        for frame in range(len(foreground))
        for observation in tracker.step(detect(foreground[frame], detection), frame)
    ]
    return TrackTable.from_observations(observations)


def find_misses(
    tracks: TrackTable, relaxed: TrackTable, radius: float
) -> tuple[Suspect, ...]:
    """Relaxed tracks that move, last five frames, and that no kept track comes near.

    What does not move is left out: still cells are found in the median, and a
    relaxed track that stays put is mostly texture, a halo or a cell's edge.
    The farther it travels, the higher it ranks.
    """
    covered = _covered(tracks, relaxed, radius)
    suspects = []
    for trajectory in relaxed.trajectories(min_length=MIN_FRAMES):
        rows = np.flatnonzero(relaxed.track == trajectory.index)
        extent = float(max(np.ptp(relaxed.x[rows]), np.ptp(relaxed.y[rows])))
        if covered[rows].mean() > 1 - UNCOVERED or extent < MOVING * radius:
            continue
        middle = rows[len(rows) // 2]
        frame, x, y = int(relaxed.frame[middle]), relaxed.x[middle], relaxed.y[middle]
        suspects.append(
            Suspect(
                kind=SuspectKind.MISSED,
                track=trajectory.index,
                frame=frame,
                x=float(x),
                y=float(y),
                score=extent,
                note=(
                    f"{len(trajectory)} frames over {extent:.0f} px that no track"
                    f" covers, at ({x:.0f}, {y:.0f})"
                ),
                suggestion=f"force --frame {frame} --x {x:.0f} --y {y:.0f}",
            )
        )
    return tuple(sorted(suspects, key=lambda item: -item.score))


def adopt(
    tracks: TrackTable,
    relaxed: TrackTable,
    points: Sequence[tuple[int, float, float]],
    radius: float,
) -> TrackTable:
    """Each relaxed track through one of the (frame, x, y) points joins as a new track.

    Its frames already covered by a kept track are left out: forcing a cell
    never duplicates one.
    """
    columns = tracks.columns()
    covered = _covered(tracks, relaxed, radius)
    next_index = int(tracks.track.max()) + 1 if len(tracks) else 0
    for frame, x, y in points:
        found = _through(relaxed, frame, x, y, radius)
        if found is None:
            continue
        rows = (relaxed.track == found) & ~covered
        added = {name: column[rows] for name, column in relaxed.columns().items()}
        added["track"] = np.full(rows.sum(), next_index)
        columns = {
            name: np.concatenate((columns[name], added[name])) for name in columns
        }
        next_index += 1
    return TrackTable.from_columns(columns)


def _relaxed(detection: DetectionParameters) -> DetectionParameters:
    return detection.model_copy(
        update={
            "threshold": detection.threshold * RELAX,
            "min_area": max(detection.min_area // 2, 1),
            "opening": 0,
        }
    )


def _covered(tracks: TrackTable, relaxed: TrackTable, radius: float) -> np.ndarray:
    """For each relaxed row, whether a kept observation of its frame is near it."""
    covered = np.zeros(len(relaxed), dtype=bool)
    for frame in np.unique(relaxed.frame):
        mine = np.flatnonzero(tracks.frame == frame)
        theirs = np.flatnonzero(relaxed.frame == frame)
        if len(mine) == 0:
            continue
        tree = KDTree(np.column_stack((tracks.x[mine], tracks.y[mine])))
        distances, _ = tree.query(
            np.column_stack((relaxed.x[theirs], relaxed.y[theirs]))
        )
        covered[theirs] = distances <= radius
    return covered


def _through(
    relaxed: TrackTable, frame: int, x: float, y: float, radius: float
) -> int | None:
    rows = np.flatnonzero(relaxed.frame == frame)
    if len(rows) == 0:
        return None
    distances = np.hypot(relaxed.x[rows] - x, relaxed.y[rows] - y)
    nearest = int(np.argmin(distances))
    return int(relaxed.track[rows[nearest]]) if distances[nearest] <= radius else None
