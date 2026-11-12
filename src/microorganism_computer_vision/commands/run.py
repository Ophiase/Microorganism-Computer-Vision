import argparse
from pathlib import Path

from motility.pipeline import Scale, run_pipeline
from pydantic import BaseModel

from microorganism_computer_vision.commands.base import (
    add_configuration,
    clip_of,
    configuration_of,
    frames_of,
    zoom_of,
)
from microorganism_computer_vision.media.folders import RUNS
from microorganism_computer_vision.media.run_folder import Workspace, source_key
from microorganism_computer_vision.media.video import read_recording
from microorganism_computer_vision.reporting.summary import RunSummary
from microorganism_computer_vision.reporting.writer import RunWriter


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
