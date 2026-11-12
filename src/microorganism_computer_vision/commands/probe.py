import argparse
from pathlib import Path
from typing import Final

import cv2
from motility.pipeline import suggest
from motility.vision.appearance import AppearanceReport, measure_appearance
from pydantic import BaseModel

from microorganism_computer_vision.commands.base import clip_of
from microorganism_computer_vision.configuration import Configuration
from microorganism_computer_vision.media.folders import RUNS
from microorganism_computer_vision.media.video import Clip, read_recording
from microorganism_computer_vision.media.workspace import source_key
from microorganism_computer_vision.reporting.looks import probe_look

DEFAULT_CLIP: Final = Clip(stop=100)


class ProbeReport(BaseModel):
    source: str
    frame_rate: float | None
    appearance: AppearanceReport
    look: Path
    suggested: Path
    next: str


class ProbeCommand:
    name = "probe"
    summary = "measure a video and suggest a first configuration"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "source", type=Path, help="a video, TIFF stack or image folder"
        )
        parser.add_argument("--clip", help="frames START:STOP, default :100")

    def execute(self, arguments: argparse.Namespace) -> ProbeReport:
        source: Path = arguments.source
        clip = clip_of(arguments.clip) or DEFAULT_CLIP
        recording = read_recording(source, clip)
        appearance = measure_appearance(recording.video)
        folder = RUNS / source_key(source) / "probe"
        folder.mkdir(parents=True, exist_ok=True)
        look = folder / "probe.png"
        cv2.imwrite(str(look), probe_look(recording.video))
        (folder / "appearance.json").write_text(appearance.model_dump_json(indent=2))
        suggested = Configuration(
            **suggest(appearance).model_dump(),
            clip=clip,
            reason=_reason(appearance),
        ).save(folder / "suggested.toml")
        return ProbeReport(
            source=str(source),
            frame_rate=recording.frame_rate,
            appearance=appearance,
            look=look,
            suggested=suggested,
            next=f"open {look}, then `mcv run {source} --config {suggested}`",
        )


def _reason(appearance: AppearanceReport) -> str:
    return (
        f"suggested by the probe: {appearance.polarity.value} objects of about"
        f" {appearance.object_length:.0f} x {appearance.object_width:.0f} px,"
        f" contrast {appearance.contrast:.2f} over noise {appearance.noise:.3f},"
        f" static share {appearance.static_share:.2f}"
    )
