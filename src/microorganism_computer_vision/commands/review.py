import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
from motility.diffusion import Motion, Verdict, classify
from motility.pipeline import PipelineResult, Scale, run_pipeline
from motility.review import (
    Action,
    Correction,
    Suspect,
    correct_tracks,
    find_suspects,
    relabel,
)
from motility.vision import Video
from pydantic import BaseModel

from microorganism_computer_vision.configuration import Configuration
from microorganism_computer_vision.media.run_folder import RunFolder
from microorganism_computer_vision.media.video import read_recording
from microorganism_computer_vision.reporting.inspection import Inspector
from microorganism_computer_vision.reporting.summary import RunSummary
from microorganism_computer_vision.reporting.writer import RunWriter


@dataclass(frozen=True, slots=True)
class Replay:
    """A run computed again from its folder, every correction applied."""

    folder: RunFolder
    configuration: Configuration
    video: Video
    result: PipelineResult
    scale: Scale

    @classmethod
    def of(cls, path: Path) -> Replay:
        folder = RunFolder(path)
        configuration = folder.configuration
        recording = read_recording(folder.source, configuration.clip)
        raw = run_pipeline(recording.video, configuration)
        corrections = folder.corrections()
        tracks = correct_tracks(raw.tracks, corrections)
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

    def inspector(self) -> Inspector:
        return Inspector(self.video, self.result, self.motions)


class SuspectsReport(BaseModel):
    sheet: Path
    suspects: tuple[Suspect, ...]
    next: str


class SuspectsCommand:
    name = "suspects"
    summary = "rank what may be wrong in a run, and draw it for review"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("run", type=Path, help="a run folder")

    def execute(self, arguments: argparse.Namespace) -> SuspectsReport:
        replay = Replay.of(arguments.run)
        suspects = find_suspects(replay.result.tracks, replay.result.verdicts)
        sheet = arguments.run / "suspects.png"
        if suspects:
            cv2.imwrite(str(sheet), replay.inspector().sheet(suspects))
        return SuspectsReport(
            sheet=sheet,
            suspects=suspects,
            next=(
                f"open {sheet}; `mcv inspect {arguments.run} TRACK` films one up"
                f" close; `mcv correct {arguments.run} ...` records what you see"
            ),
        )


class InspectReport(BaseModel):
    film: Path
    verdicts: tuple[dict[str, object], ...]


class InspectCommand:
    name = "inspect"
    summary = "film tracks up close, with every track in view numbered"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("run", type=Path, help="a run folder")
        parser.add_argument("tracks", type=int, nargs="+", help="track numbers")

    def execute(self, arguments: argparse.Namespace) -> InspectReport:
        replay = Replay.of(arguments.run)
        names = "_".join(map(str, arguments.tracks))
        film = arguments.run / f"inspect_{names}.png"
        cv2.imwrite(str(film), replay.inspector().film(arguments.tracks))
        wanted = set(arguments.tracks)
        return InspectReport(
            film=film,
            verdicts=tuple(
                _numbers(item, replay.scale)
                for item in replay.result.verdicts
                if item.track in wanted
            ),
        )


class CorrectReport(BaseModel):
    corrections: tuple[Correction, ...]
    summary: RunSummary


class CorrectCommand:
    name = "correct"
    summary = "record a judgement on a track, with its reason, and rewrite the run"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("run", type=Path, help="a run folder")
        parser.add_argument("action", choices=[item.value for item in Action])
        parser.add_argument("track", type=int)
        parser.add_argument("--other", type=int, help="join: the track appended")
        parser.add_argument(
            "--frame", type=int, help="split: first frame of the new track"
        )
        parser.add_argument("--motion", choices=[item.value for item in Motion])
        parser.add_argument("--reason", required=True, help="what you saw")
        parser.add_argument("--by", default="agent", help="who judged")

    def execute(self, arguments: argparse.Namespace) -> CorrectReport:
        correction = Correction(
            action=Action(arguments.action),
            track=arguments.track,
            other=arguments.other,
            frame=arguments.frame,
            motion=Motion(arguments.motion) if arguments.motion else None,
            reason=arguments.reason,
            by=arguments.by,
        )
        corrections = RunFolder(arguments.run).append(correction)
        replay = Replay.of(arguments.run)
        record = RunWriter(
            arguments.run, str(replay.folder.source), replay.configuration, replay.scale
        ).write(f"{replay.folder.source.name}, corrected", replay.video, replay.result)
        return CorrectReport(corrections=corrections, summary=record.summary)


def _numbers(verdict: Verdict, scale: Scale) -> dict[str, object]:
    exponent = verdict.exponent
    return {
        "track": verdict.track,
        "length": verdict.length,
        "motion": verdict.motion.value,
        "confidence": round(verdict.confidence, 3),
        "alpha": None if exponent is None else round(exponent.value, 3),
        f"speed ({scale.speed_unit})": round(scale.speed(verdict.speed), 3),
        "turns": round(verdict.turning.turns, 2),
    }
