from dataclasses import dataclass
from pathlib import Path
from typing import Self

from motility.diffusion import Motion, classify
from motility.pipeline import PipelineParameters, PipelineResult, Scale, run_pipeline
from motility.review import (
    Correction,
    adopt,
    correct_tracks,
    forced_points,
    relabel,
    relaxed_tracks,
)
from motility.tracking import TrackTable
from motility.vision import Video

from microorganism_computer_vision.configuration import Configuration
from microorganism_computer_vision.media.run_folder import RunFolder
from microorganism_computer_vision.media.video import read_recording


@dataclass(frozen=True, slots=True)
class Replay:
    """A run computed again from its folder, every correction applied."""

    folder: RunFolder
    configuration: Configuration
    video: Video
    result: PipelineResult
    scale: Scale

    @classmethod
    def of(cls, path: Path) -> Self:
        folder = RunFolder(path)
        configuration = folder.configuration
        recording = read_recording(folder.source, configuration.clip)
        raw = run_pipeline(recording.video, configuration)
        corrections = folder.corrections()
        tracks = correct_tracks(_forced(raw, configuration, corrections), corrections)
        verdicts = relabel(
            [
                classify(item, configuration.classification)
                for item in tracks.trajectories()
            ],
            corrections,
        )
        result = PipelineResult(raw.foreground, raw.detections, tracks, verdicts)
        scale = Scale.from_acquisition(configuration.acquisition, recording.frame_rate)
        return cls(folder, configuration, recording.video, result, scale)

    @property
    def motions(self) -> dict[int, Motion]:
        return {verdict.track: verdict.motion for verdict in self.result.verdicts}

    @property
    def radius(self) -> float:
        return match_radius(self.configuration)

    def relaxed(self) -> TrackTable:
        return relaxed_tracks(self.result.foreground, self.configuration)


def match_radius(parameters: PipelineParameters) -> float:
    """Two observations closer than half the tracker's gate are one cell."""
    return parameters.tracking.max_assignment_distance / 2


def _forced(
    raw: PipelineResult,
    parameters: PipelineParameters,
    corrections: tuple[Correction, ...],
) -> TrackTable:
    """The raw tracks and every cell forced in, each as a new track."""
    points = forced_points(corrections)
    if not points:
        return raw.tracks
    relaxed = relaxed_tracks(raw.foreground, parameters)
    return adopt(raw.tracks, relaxed, points, match_radius(parameters))
