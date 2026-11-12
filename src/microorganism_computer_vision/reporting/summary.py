from collections.abc import Sequence
from typing import Self

import numpy as np
from motility.diffusion import Motion, Verdict
from motility.pipeline import PipelineParameters, PipelineResult, Scale
from pydantic import BaseModel, Field

SHORT_TRACKS = 0.8
STATIONARY = 0.5
FLICKER = 0.5
MIN_DETECTIONS = 2.0


class Spread(BaseModel):
    """A distribution in three numbers: the median and the 10th and 90th percentiles."""

    median: float | None
    low: float | None
    high: float | None

    @classmethod
    def of(cls, values: Sequence[float]) -> Self:
        if not values:
            return cls(median=None, low=None, high=None)
        low, median, high = np.percentile(values, (10, 50, 90)).tolist()
        return cls(median=median, low=low, high=high)


class RunSummary(BaseModel):
    """What one run found, in the numbers an expert reads before the pictures."""

    frames: int
    detections_per_frame: Spread
    tracks: int
    track_length: Spread = Field(description="Observations per track.")
    classified: int = Field(description="Tracks long enough to be given a motion.")
    motions: dict[Motion, int]
    speed: Spread = Field(description="Mean speed of the classified tracks.")
    speed_unit: str
    alpha: Spread = Field(description="Anomalous exponent of the classified tracks.")
    diffusion: Spread = Field(
        description="Diffusion coefficient of the diffusive tracks."
    )
    diffusion_unit: str
    hints: tuple[str, ...] = Field(
        description="What the numbers suggest to look at; a hint, never a verdict."
    )

    @classmethod
    def of(
        cls, result: PipelineResult, parameters: PipelineParameters, scale: Scale
    ) -> Self:
        counts = [len(frame) for frame in result.detections]
        lengths = [verdict.length for verdict in result.verdicts]
        classified = [
            item for item in result.verdicts if item.motion is not Motion.UNDETERMINED
        ]
        motions = {
            motion: sum(item.motion is motion for item in classified)
            for motion in Motion
        }
        motions[Motion.UNDETERMINED] = len(result.verdicts) - len(classified)
        return cls(
            frames=len(result.detections),
            detections_per_frame=Spread.of(counts),
            tracks=len(result.verdicts),
            track_length=Spread.of(lengths),
            classified=len(classified),
            motions=motions,
            speed=Spread.of([scale.speed(item.speed) for item in classified]),
            speed_unit=scale.speed_unit,
            alpha=Spread.of([
                item.exponent.value for item in classified if item.exponent
            ]),
            diffusion=Spread.of([
                scale.diffusion(item.diffusion)
                for item in classified
                if item.motion is Motion.DIFFUSIVE and item.diffusion is not None
            ]),
            diffusion_unit=scale.diffusion_unit,
            hints=hints(counts, result.verdicts, parameters),
        )


def hints(
    counts: Sequence[int], verdicts: Sequence[Verdict], parameters: PipelineParameters
) -> tuple[str, ...]:
    """The usual failures, read off the numbers, each with the knob to try first."""
    found = []
    mean = float(np.mean(counts)) if counts else 0.0
    if mean < MIN_DETECTIONS:
        found.append(
            "almost nothing is detected: open a look; lower detection.threshold or"
            " widen the preprocessing window, and check preprocessing.polarity"
        )
    if mean and float(np.std(counts)) > FLICKER * mean:
        found.append(
            "the detections per frame vary a lot: cells cross the threshold in and"
            " out; smooth more, or lower detection.threshold"
        )
    minimum = parameters.classification.min_track_length
    if (
        verdicts
        and np.mean([item.length < minimum for item in verdicts]) > SHORT_TRACKS
    ):
        found.append(
            f"most tracks are shorter than {minimum} observations: the tracker loses"
            " them; raise tracking.max_assignment_distance or max_missed_frames,"
            " or look for blinking detections"
        )
    classified = [item for item in verdicts if item.motion is not Motion.UNDETERMINED]
    stationary = sum(item.motion is Motion.STATIONARY for item in classified)
    if classified and stationary > STATIONARY * len(classified):
        found.append(
            "most classified tracks are stationary: dust or stuck cells; try"
            " preprocessing.background = temporal-median"
        )
    return tuple(found)
