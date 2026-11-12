from collections.abc import Sequence
from dataclasses import replace
from enum import StrEnum
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from motility.diffusion.models.motion import Motion
from motility.diffusion.models.selection import Verdict
from motility.tracking.table import TrackTable


class Action(StrEnum):
    DROP = "drop"
    SPLIT = "split"
    JOIN = "join"
    LABEL = "label"


class Correction(BaseModel):
    """One judgement on one track, by whoever looked, and why.

    drop removes the track; split starts a new track at `frame`; join appends
    `other` to `track`, the first keeping any frame both hold; label overrides
    the verdict with `motion`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Action
    track: int = Field(ge=0)
    other: int | None = Field(
        default=None, ge=0, description="join: the track appended."
    )
    frame: int | None = Field(
        default=None, ge=0, description="split: first frame of the new track."
    )
    motion: Motion | None = Field(default=None, description="label: the motion seen.")
    reason: str = Field(
        min_length=1, description="What was seen that the pipeline missed."
    )
    by: str = Field(
        default="agent", description="Who judged: agent, or a person's name."
    )

    @model_validator(mode="after")
    def _has_what_its_action_needs(self) -> Self:
        needed = {
            Action.JOIN: self.other,
            Action.SPLIT: self.frame,
            Action.LABEL: self.motion,
        }
        if self.action in needed and needed[self.action] is None:
            raise ValueError(f"a {self.action.value} needs its other, frame or motion")
        return self


def correct_tracks(tracks: TrackTable, corrections: Sequence[Correction]) -> TrackTable:
    """The table after every drop, split and join, applied in order."""
    columns = {name: column.copy() for name, column in tracks.columns().items()}
    for correction in corrections:
        columns = _apply(columns, correction)
    return TrackTable.from_columns(columns)


def relabel(
    verdicts: Sequence[Verdict], corrections: Sequence[Correction]
) -> tuple[Verdict, ...]:
    """The verdicts with every label applied, each certain: its confidence is 1."""
    labels = {
        item.track: item.motion
        for item in corrections
        if item.action is Action.LABEL and item.motion is not None
    }
    return tuple(
        replace(verdict, motion=labels[verdict.track], confidence=1.0)
        if verdict.track in labels
        else verdict
        for verdict in verdicts
    )


def _apply(
    columns: dict[str, np.ndarray], correction: Correction
) -> dict[str, np.ndarray]:
    track = columns["track"]
    match correction.action:
        case Action.DROP:
            return _keep(columns, track != correction.track)
        case Action.SPLIT:
            later = (track == correction.track) & (
                columns["frame"] >= (correction.frame or 0)
            )
            columns["track"] = np.where(later, track.max() + 1, track)
            return columns
        case Action.JOIN:
            mine = set(columns["frame"][track == correction.track].tolist())
            theirs = track == correction.other
            duplicate = theirs & np.isin(columns["frame"], list(mine))
            columns["track"] = np.where(theirs, correction.track, track)
            return _keep(columns, ~duplicate)
        case Action.LABEL:
            return columns


def _keep(columns: dict[str, np.ndarray], rows: np.ndarray) -> dict[str, np.ndarray]:
    return {name: column[rows] for name, column in columns.items()}
