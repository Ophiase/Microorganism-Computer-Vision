from collections.abc import Sequence
from typing import Final, Self

import numpy as np
from motility.diffusion import Motion
from motility.diffusion.behaviour import Behaviour, behaviour, distance_to_objects
from motility.vision.regions.mask import static_objects
from pydantic import BaseModel, Field
from scipy.stats import binomtest

from microorganism_computer_vision.media.replay import Replay
from microorganism_computer_vision.reporting.summary import Spread

MOTILE: Final = frozenset({Motion.DIRECTED, Motion.CIRCULAR})


class Population(BaseModel):
    """The behaviour of every classified cell of one run, in the run's units."""

    run: str
    source: str
    classified: int
    motions: dict[Motion, int]
    motile_share: float = Field(
        description="Directed and circular over all classified."
    )
    motile_speed: Spread = Field(description="Mean speed of the motile tracks.")
    motile_speeds: tuple[float, ...]
    circling_share: float = Field(description="Circular over all motile tracks.")
    clockwise: int = Field(description="Circular tracks turning clockwise as shown.")
    anticlockwise: int
    chirality_p: float | None = Field(
        description="Two-sided binomial p-value of as many turning each way."
    )
    circle_radius: Spread = Field(description="Radius of the circular tracks.")
    reorientations_per_second: float | None = Field(
        description="Turns sharper than 60 degrees per second of motile swimming."
    )
    straightness: Spread = Field(description="Net over path length, directed tracks.")
    distance_motile: Spread = Field(description="To the nearest static object.")
    distance_still: Spread = Field(description="Stationary and diffusive tracks, same.")
    speed_unit: str
    length_unit: str

    @classmethod
    def of(cls, replay: Replay) -> Self:
        verdicts = {item.track: item for item in replay.result.verdicts}
        classified = [
            item for item in verdicts.values() if item.motion is not Motion.UNDETERMINED
        ]
        trajectories = {
            item.index: item for item in replay.result.tracks.trajectories()
        }
        motile = [item for item in classified if item.motion in MOTILE]
        circling = [
            behaviour(trajectories[item.track])
            for item in classified
            if item.motion is Motion.CIRCULAR
        ]
        clockwise = sum(item.angular_velocity > 0 for item in circling)
        scale = replay.scale
        objects = _objects(replay)
        return cls(
            run=str(replay.folder.path),
            source=replay.folder.source.stem,
            classified=len(classified),
            motions={
                motion: sum(item.motion is motion for item in classified)
                for motion in Motion
            },
            motile_share=len(motile) / len(classified) if classified else 0.0,
            motile_speed=Spread.of([scale.speed(item.speed) for item in motile]),
            motile_speeds=tuple(scale.speed(item.speed) for item in motile),
            circling_share=len(circling) / len(motile) if motile else 0.0,
            clockwise=clockwise,
            anticlockwise=len(circling) - clockwise,
            chirality_p=binomtest(clockwise, len(circling)).pvalue
            if circling
            else None,
            circle_radius=Spread.of([
                item.radius * scale.length for item in circling if item.radius
            ]),
            reorientations_per_second=_rate(
                [behaviour(trajectories[item.track]) for item in motile], scale.time
            ),
            straightness=Spread.of([
                behaviour(trajectories[item.track]).straightness
                for item in classified
                if item.motion is Motion.DIRECTED
            ]),
            distance_motile=_distances(motile, trajectories, objects, scale.length),
            distance_still=_distances(
                [item for item in classified if item.motion not in MOTILE],
                trajectories,
                objects,
                scale.length,
            ),
            speed_unit=scale.speed_unit,
            length_unit=scale.length_unit,
        )


def _objects(replay: Replay) -> np.ndarray | None:
    preprocessing = replay.configuration.preprocessing
    if preprocessing.mask.static is None:
        return None
    return static_objects(replay.video, preprocessing.mask.static, 0)


def _rate(behaviours: Sequence[Behaviour], seconds_per_frame: float) -> float | None:
    duration = sum(item.duration for item in behaviours) * seconds_per_frame
    return (
        sum(item.reorientations for item in behaviours) / duration if duration else None
    )


def _distances(
    verdicts: Sequence, trajectories: dict, objects: np.ndarray | None, length: float
) -> Spread:
    if objects is None:
        return Spread.of([])
    values = [
        distance_to_objects(trajectories[item.track], objects) for item in verdicts
    ]
    return Spread.of([value * length for value in values if value is not None])
