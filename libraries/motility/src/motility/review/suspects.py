from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import Final

import numpy as np
from pydantic import BaseModel, Field

from motility.diffusion.models.motion import Motion
from motility.diffusion.models.selection import Verdict
from motility.tracking.table import TrackTable
from motility.tracking.trajectories import Trajectory

JUMP_RATIO: Final = 4.0
MIN_JUMP: Final = 6.0
GAP_FRAMES: Final = 10
GAP_SLACK: Final = 6.0
UNSURE: Final = 0.6
FAST_RATIO: Final = 3.0
SHOWN: Final = 40


class SuspectKind(StrEnum):
    JUMP = "jump"
    GAP = "gap"
    UNSURE = "unsure"
    FAST = "fast"
    MISSED = "missed"


class Suspect(BaseModel):
    """Something a pipeline may have got wrong, ranked for an agent to look at."""

    kind: SuspectKind
    track: int
    other: int | None = Field(
        default=None, description="gap: the track that may continue it."
    )
    frame: int | None = Field(default=None, description="Where to look.")
    x: float | None = Field(default=None, description="missed: where, in the frame.")
    y: float | None = Field(default=None, description="missed: where, in the frame.")
    score: float = Field(description="How suspicious, comparable within a kind.")
    note: str
    suggestion: str = Field(description="The correction to try if the film agrees.")


def find_suspects(
    tracks: TrackTable, verdicts: Sequence[Verdict]
) -> tuple[Suspect, ...]:
    """Jumps, broken tracks, unsure verdicts and outliers, the worst of each first."""
    trajectories = {item.index: item for item in tracks.trajectories(min_length=2)}
    found = [
        *_jumps(trajectories.values()),
        *_gaps(trajectories),
        *_unsure(verdicts),
        *_fast(verdicts),
    ]
    found.sort(key=lambda suspect: -suspect.score)
    return tuple(found[:SHOWN])


def _jumps(trajectories: Iterable[Trajectory]) -> list[Suspect]:
    """A step far longer than the track's usual one: two cells swapped."""
    suspects = []
    for trajectory in trajectories:
        steps = np.hypot(*np.diff(np.array(trajectory.positions), axis=0).T)
        if len(steps) < 4:
            continue
        usual = max(float(np.median(steps)), 1.0)
        worst = int(np.argmax(steps))
        ratio = float(steps[worst]) / usual
        if ratio >= JUMP_RATIO and steps[worst] >= MIN_JUMP:
            frame = trajectory.frames[worst + 1]
            suspects.append(
                Suspect(
                    kind=SuspectKind.JUMP,
                    track=trajectory.index,
                    frame=frame,
                    score=ratio,
                    note=f"a step of {steps[worst]:.0f} px, {ratio:.0f} x its usual",
                    suggestion=f"split {trajectory.index} at {frame}",
                )
            )
    return suspects


def _gaps(trajectories: dict[int, Trajectory]) -> list[Suspect]:
    """A track that starts where and when another one ended, as if continuing it."""
    endings = [item for item in trajectories.values() if len(item) >= 3]
    return [
        suspect
        for ending in endings
        for starting in trajectories.values()
        if (suspect := _continuation(ending, starting)) is not None
    ]


def _continuation(ending: Trajectory, starting: Trajectory) -> Suspect | None:
    """`starting` if it begins within a few frames on the path `ending` was on."""
    delay = starting.frames[0] - ending.frames[-1]
    if starting.index == ending.index or not 0 < delay <= GAP_FRAMES:
        return None
    position = np.array(ending.positions[-1])
    elapsed = max(ending.frames[-1] - ending.frames[-3], 1)
    velocity = (position - np.array(ending.positions[-3])) / elapsed
    expected = position + velocity * delay
    distance = float(np.hypot(*(np.array(starting.positions[0]) - expected)))
    if distance > GAP_SLACK + float(np.hypot(*velocity)) * delay / 2:
        return None
    return Suspect(
        kind=SuspectKind.GAP,
        track=ending.index,
        other=starting.index,
        frame=starting.frames[0],
        score=1 / (1 + distance) + 1 / delay,
        note=(
            f"{starting.index} starts {delay} frames later,"
            f" {distance:.1f} px from where {ending.index} was heading"
        ),
        suggestion=f"join {ending.index} {starting.index}",
    )


def _unsure(verdicts: Sequence[Verdict]) -> list[Suspect]:
    return [
        Suspect(
            kind=SuspectKind.UNSURE,
            track=item.track,
            score=1 - item.confidence,
            note=f"{item.motion.value} at {item.confidence:.2f}",
            suggestion=f"look at {item.track} and label it if its motion is clear",
        )
        for item in verdicts
        if item.motion is not Motion.UNDETERMINED and item.confidence < UNSURE
    ]


def _fast(verdicts: Sequence[Verdict]) -> list[Suspect]:
    classified = [item for item in verdicts if item.motion is not Motion.UNDETERMINED]
    if not classified:
        return []
    typical = float(np.median([item.speed for item in classified]))
    return [
        Suspect(
            kind=SuspectKind.FAST,
            track=item.track,
            score=item.speed / max(typical, 1e-9),
            note=f"{item.speed:.1f} px/frame against a median of {typical:.1f}",
            suggestion=f"look at {item.track}: a swap between cells often looks fast",
        )
        for item in classified
        if item.speed > FAST_RATIO * typical
    ]
