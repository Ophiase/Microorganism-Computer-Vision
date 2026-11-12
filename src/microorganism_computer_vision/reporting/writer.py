from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from motility.pipeline import PipelineResult, Scale
from motility.vision import Video

from microorganism_computer_vision.configuration import Configuration
from microorganism_computer_vision.media.tables import write_tracks, write_verdicts
from microorganism_computer_vision.reporting.figures import distributions
from microorganism_computer_vision.reporting.looks import Zoom, look, trajectory_map
from microorganism_computer_vision.reporting.markdown import report
from microorganism_computer_vision.reporting.summary import RunSummary

LOOKS: Final = 3


@dataclass(frozen=True, slots=True)
class RunRecord:
    """A written run: its folder, its numbers, and the pictures an agent opens."""

    folder: Path
    summary: RunSummary
    pictures: tuple[Path, ...]


class RunWriter:
    """Lays a run out as a folder: the numbers as JSON and CSV, the eyes as PNG.

    Everything a later reader needs is inside, the configuration with its reason
    first, so a folder alone can be judged, rerun or compared.
    """

    def __init__(
        self, folder: Path, configuration: Configuration, scale: Scale
    ) -> None:
        self._folder = folder
        self._configuration = configuration
        self._scale = scale

    def write(
        self,
        title: str,
        video: Video,
        result: PipelineResult,
        look_frames: Sequence[int] = (),
        zoom: Zoom | None = None,
    ) -> RunRecord:
        self._folder.mkdir(parents=True, exist_ok=True)
        summary = RunSummary.of(result, self._configuration, self._scale)
        self._configuration.save(self._folder / "configuration.toml")
        write_tracks(self._folder / "tracks.csv", result.tracks)
        write_verdicts(self._folder / "verdicts.csv", result.verdicts, self._scale)
        (self._folder / "summary.json").write_text(summary.model_dump_json(indent=2))
        pictures = (
            *self._looks(
                video, result, look_frames or _spread_frames(len(video)), zoom
            ),
            self._trajectories(video, result),
            distributions(
                result.verdicts, self._scale, self._folder / "distributions.png"
            ),
        )
        page = report(
            title,
            self._configuration.reason,
            summary,
            result.verdicts,
            self._scale,
            pictures,
        )
        (self._folder / "report.md").write_text(page)
        return RunRecord(self._folder, summary, pictures)

    def _looks(
        self,
        video: Video,
        result: PipelineResult,
        frames: Sequence[int],
        zoom: Zoom | None,
    ) -> tuple[Path, ...]:
        motions = {verdict.track: verdict.motion for verdict in result.verdicts}
        written = []
        for frame in frames:
            detection = self._configuration.detection
            image = look(video, result, detection, motions, frame, zoom)
            path = self._folder / f"look_{frame:04d}.png"
            cv2.imwrite(str(path), image)
            written.append(path)
        return tuple(written)

    def _trajectories(self, video: Video, result: PipelineResult) -> Path:
        motions = {verdict.track: verdict.motion for verdict in result.verdicts}
        image = trajectory_map(
            np.median(video, axis=0), result.tracks.trajectories(), motions
        )
        path = self._folder / "trajectories.png"
        cv2.imwrite(str(path), image)
        return path


def _spread_frames(count: int) -> tuple[int, ...]:
    return tuple(sorted({round(index) for index in np.linspace(0, count - 1, LOOKS)}))
