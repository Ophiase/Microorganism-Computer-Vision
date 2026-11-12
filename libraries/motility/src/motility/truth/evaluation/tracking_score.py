from collections import Counter

import numpy as np
from pydantic import BaseModel, Field

from motility.tracking.table import TrackTable
from motility.truth.evaluation.matching import FrameMatch

MOSTLY = 0.8


class TrackingScore(BaseModel):
    """How close the tracks are to the truth, over every frame of a clip."""

    mota: float = Field(
        description="1 - (misses + false alarms + identity switches) / true positions."
    )
    precision: float = Field(
        description="Share of the predicted positions that are true."
    )
    recall: float = Field(description="Share of the true positions that were found.")
    identity_switches: int = Field(description="Times a true cell changed track.")
    localisation_error: float = Field(description="Mean distance of the pairs, px.")
    mostly_tracked: float = Field(
        description="Share of true cells one track follows for 80 % of their frames."
    )
    true_positions: int
    predicted_positions: int


def score_tracking(matches: tuple[FrameMatch, ...], truth: TrackTable) -> TrackingScore:
    truths = sum(match.truths for match in matches)
    predictions = sum(match.predictions for match in matches)
    found = sum(len(match.pairs) for match in matches)
    switches = _identity_switches(matches)
    distances = [distance for match in matches for distance in match.distances]
    return TrackingScore(
        mota=1 - ((truths - found) + (predictions - found) + switches) / max(truths, 1),
        precision=found / max(predictions, 1),
        recall=found / max(truths, 1),
        identity_switches=switches,
        localisation_error=float(np.mean(distances)) if distances else 0.0,
        mostly_tracked=_mostly_tracked(matches, truth),
        true_positions=truths,
        predicted_positions=predictions,
    )


def _identity_switches(matches: tuple[FrameMatch, ...]) -> int:
    last: dict[int, int] = {}
    switches = 0
    for match in matches:
        for true_cell, track in match.pairs:
            if true_cell in last and last[true_cell] != track:
                switches += 1
            last[true_cell] = track
    return switches


def _mostly_tracked(matches: tuple[FrameMatch, ...], truth: TrackTable) -> float:
    """Cells whose most frequent track covers 80 % of the frames the cell is visible."""
    cells, visible = np.unique(truth.track, return_counts=True)
    if len(cells) == 0:
        return 0.0
    tracks: dict[int, Counter[int]] = {}
    for match in matches:
        for true_cell, track in match.pairs:
            tracks.setdefault(true_cell, Counter())[track] += 1
    followed = sum(
        1
        for true_cell, frames in zip(cells.tolist(), visible.tolist(), strict=True)
        if true_cell in tracks
        and tracks[true_cell].most_common(1)[0][1] >= MOSTLY * frames
    )
    return followed / len(cells)
