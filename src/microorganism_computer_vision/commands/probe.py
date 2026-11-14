import argparse
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import yaml
from motility.pipeline import suggest
from motility.vision import Video
from motility.vision.measures.appearance import AppearanceReport, measure_appearance
from motility.vision.measures.scale_bar import find_scale_bars
from motility.vision.regions.mask import ignored_region
from pydantic import BaseModel

from microorganism_computer_vision.commands.base import (
    add_configuration,
    clip_of,
    configuration_of,
)
from microorganism_computer_vision.configuration import Configuration
from microorganism_computer_vision.media.extract import process as download_study
from microorganism_computer_vision.media.folders import CATALOGUE, RUNS
from microorganism_computer_vision.media.run_folder import source_key
from microorganism_computer_vision.media.video import Clip, read_recording
from microorganism_computer_vision.reporting.pictures.looks import (
    mask_preview,
    probe_look,
)

DEFAULT_CLIP: Final = Clip(stop=100)
CROP_MARGIN: Final = 25


class ScaleBarCandidate(BaseModel):
    x: int
    y: int
    length: int
    crop: Path


class ProbeReport(BaseModel):
    source: str
    frame_rate: float | None
    appearance: AppearanceReport
    scale_bars: tuple[ScaleBarCandidate, ...]
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
            scale_bars=_scale_bars(recording.video, folder),
            look=look,
            suggested=suggested,
            next=(
                f"open {look} and the scale bar crops: read the length written"
                " beside the bar, set acquisition.pixel_size to it over the bar's"
                f" length, then `mcv run {source} --config {suggested}`"
            ),
        )


def _reason(appearance: AppearanceReport) -> str:
    return (
        f"suggested by the probe: {appearance.polarity.value} objects of about"
        f" {appearance.object_length:.0f} x {appearance.object_width:.0f} px,"
        f" contrast {appearance.contrast:.2f} over noise {appearance.noise:.3f},"
        f" static share {appearance.static_share:.2f}"
    )


def _scale_bars(video: Video, folder: Path) -> tuple[ScaleBarCandidate, ...]:
    """The burnt-in bars the probe suspects, each with an enlarged crop to read."""
    median = np.median(video, axis=0)
    candidates = []
    for index, bar in enumerate(find_scale_bars(median), start=1):
        top, left = max(bar.y - CROP_MARGIN, 0), max(bar.x - CROP_MARGIN, 0)
        window = median[
            top : bar.y + CROP_MARGIN, left : bar.x + bar.length + CROP_MARGIN
        ]
        crop = folder / f"scale_bar_{index}.png"
        image = (np.clip(window, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(
            str(crop),
            cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST),
        )
        candidates.append(
            ScaleBarCandidate(x=bar.x, y=bar.y, length=bar.length, crop=crop)
        )
    return tuple(candidates)


class MaskReport(BaseModel):
    preview: Path
    ignored_share: float
    next: str


class MaskCommand:
    name = "mask"
    summary = "draw what a configuration's mask hides, to check it before a run"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "source", type=Path, help="a video, TIFF stack or image folder"
        )
        add_configuration(parser)

    def execute(self, arguments: argparse.Namespace) -> MaskReport:
        configuration = configuration_of(arguments)
        video = read_recording(arguments.source, configuration.clip).video
        mask = configuration.preprocessing.mask
        ignored = ignored_region(video, mask)
        folder = RUNS / source_key(arguments.source)
        folder.mkdir(parents=True, exist_ok=True)
        preview = folder / "mask.png"
        cv2.imwrite(str(preview), mask_preview(np.median(video, axis=0), ignored, mask))
        return MaskReport(
            preview=preview,
            ignored_share=float(ignored.mean()),
            next=(
                f"open {preview}: red is ignored. Mask an object the level missed"
                " with --set 'preprocessing.mask.ignore=[{x=..., y=..., radius=...}]',"
                " give back a cell with preprocessing.mask.keep"
            ),
        )


class Sources(BaseModel):
    count: int
    sources: tuple[dict[str, object], ...]


class SourcesCommand:
    name = "sources"
    summary = "the catalogue of motility videos, their truth, licence and trust"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--with-truth", action="store_true", help="only those with a truth"
        )

    def execute(self, arguments: argparse.Namespace) -> Sources:
        entries = yaml.safe_load(CATALOGUE.read_text())
        if arguments.with_truth:
            entries = [entry for entry in entries if _has_truth(entry)]
        return Sources(count=len(entries), sources=tuple(entries))


class Fetched(BaseModel):
    videos: tuple[Path, ...]


class FetchCommand:
    name = "fetch"
    summary = "download the nine videos of the 2025 study into data/ (asks the network)"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """The study's addresses are in settings/dataset.toml."""

    def execute(self, _arguments: argparse.Namespace, /) -> Fetched:
        return Fetched(videos=download_study())


def _has_truth(entry: dict[str, object]) -> bool:
    return not str(entry.get("truth", "none")).startswith("none")


def _has_truth(entry: dict[str, object]) -> bool:
    return not str(entry.get("truth", "none")).startswith("none")
