import argparse
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from motility.pipeline import Scale, run_pipeline
from motility.review import find_misses, relaxed_tracks
from pydantic import BaseModel

from microorganism_computer_vision.commands.base import (
    add_configuration,
    clip_of,
    configuration_of,
    frames_of,
    zoom_of,
)
from microorganism_computer_vision.configuration import Configuration
from microorganism_computer_vision.media.folders import RUNS
from microorganism_computer_vision.media.replay import match_radius
from microorganism_computer_vision.media.run_folder import Workspace, source_key
from microorganism_computer_vision.media.video import Clip, UsageError, read_recording
from microorganism_computer_vision.reporting.inspection import side_by_side
from microorganism_computer_vision.reporting.pictures.overlay import TrackOverlay
from microorganism_computer_vision.reporting.summary import RunSummary
from microorganism_computer_vision.reporting.writer import RunWriter

SWEEP_CLIP: Final = Clip(stop=60)


class RunReport(BaseModel):
    folder: Path
    summary: RunSummary
    pictures: tuple[Path, ...]
    next: str


class RunCommand:
    name = "run"
    summary = "run a configuration on a video; write the tracks, looks and report"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "source", type=Path, help="a video, TIFF stack or image folder"
        )
        add_configuration(parser)
        parser.add_argument(
            "--clip", help="frames START:STOP; overrides the configuration"
        )
        parser.add_argument("--looks", help="frames to look at, e.g. 0,40,80")
        parser.add_argument("--zoom", help="enlarge X,Y,SIZE of every look")

    def execute(self, arguments: argparse.Namespace) -> RunReport:
        configuration = configuration_of(arguments)
        clip = clip_of(arguments.clip)
        if clip is not None:
            configuration = configuration.model_copy(update={"clip": clip})
        recording = read_recording(arguments.source, configuration.clip)
        result = run_pipeline(recording.video, configuration)
        scale = Scale.from_acquisition(configuration.acquisition, recording.frame_rate)
        folder = Workspace(RUNS, source_key(arguments.source)).next_folder()
        record = RunWriter(folder, str(arguments.source), configuration, scale).write(
            f"{arguments.source.name}, run {folder.name}",
            recording.video,
            result,
            frames_of(arguments.looks),
            zoom_of(arguments.zoom),
        )
        return RunReport(
            folder=record.folder,
            summary=record.summary,
            pictures=record.pictures,
            next=(
                "open the looks; change one parameter with --set, say why with"
                " --reason, run again; `mcv history` shows the journal"
            ),
        )


class SweepRow(BaseModel):
    value: str
    detections_per_frame: float
    tracks: int
    long_tracks: int
    misses: int


class SweepReport(BaseModel):
    knob: str
    picture: Path
    rows: tuple[SweepRow, ...]


class SweepCommand:
    name = "sweep"
    summary = "one parameter over several values, side by side on the same view"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "source", type=Path, help="a video, TIFF stack or image folder"
        )
        add_configuration(parser)
        parser.add_argument("knob", help="section.key, e.g. detection.threshold")
        parser.add_argument("values", help="comma-separated, e.g. 0.5,0.4,0.3")
        parser.add_argument("--clip", help="frames START:STOP, default :60")
        parser.add_argument(
            "--frame", type=int, help="the frame shown, default the middle"
        )
        parser.add_argument(
            "--zoom", help="X,Y,SIZE of the view, default the whole frame"
        )

    def execute(self, arguments: argparse.Namespace) -> SweepReport:
        base = configuration_of(arguments)
        clip = clip_of(arguments.clip) or SWEEP_CLIP
        video = read_recording(arguments.source, clip).video
        frame = arguments.frame if arguments.frame is not None else len(video) // 2
        if not 0 <= frame < len(video):
            raise UsageError(
                f"frame {frame} is outside the clip of {len(video)} frames"
            )
        zoom = zoom_of(arguments.zoom)
        rows, images = [], []
        for value in arguments.values.split(","):
            configuration = base.with_overrides([f"{arguments.knob}={value}"])
            row, image = _try(video, configuration, frame, value)
            rows.append(row)
            images.append(zoom.apply(image) if zoom else image)
        picture = RUNS / source_key(arguments.source) / f"sweep_{arguments.knob}.png"
        picture.parent.mkdir(parents=True, exist_ok=True)
        captions = [
            f"{arguments.knob} = {row.value}: {row.misses} missed" for row in rows
        ]
        cv2.imwrite(str(picture), side_by_side(images, captions))
        return SweepReport(knob=arguments.knob, picture=picture, rows=tuple(rows))


def _try(
    video: np.ndarray, configuration: Configuration, frame: int, value: str
) -> tuple[SweepRow, np.ndarray]:
    result = run_pipeline(video, configuration)
    minimum = configuration.classification.min_track_length
    misses = find_misses(
        result.tracks,
        relaxed_tracks(result.foreground, configuration),
        match_radius(configuration),
    )
    row = SweepRow(
        value=value,
        detections_per_frame=float(np.mean([len(item) for item in result.detections])),
        tracks=len(result.verdicts),
        long_tracks=sum(item.length >= minimum for item in result.verdicts),
        misses=len(misses),
    )
    motions = {item.track: item.motion for item in result.verdicts}
    return row, TrackOverlay(result, motions, labels=False).draw(video[frame], frame)
