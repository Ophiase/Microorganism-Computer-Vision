import argparse
from pathlib import Path

import cv2
from motility.diffusion import Motion, Verdict
from motility.pipeline import Scale
from motility.review import Action, Correction, Suspect, find_misses, find_suspects
from pydantic import BaseModel

from microorganism_computer_vision.media.replay import Replay
from microorganism_computer_vision.media.run_folder import RunFolder
from microorganism_computer_vision.media.video import UsageError
from microorganism_computer_vision.reporting.inspection import Inspector
from microorganism_computer_vision.reporting.summary import RunSummary
from microorganism_computer_vision.reporting.writer import RunWriter


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
            cv2.imwrite(str(sheet), _inspector(replay).sheet(suspects))
        return SuspectsReport(
            sheet=sheet,
            suspects=suspects,
            next=(
                f"open {sheet}; `mcv inspect {arguments.run} TRACK` films one up"
                f" close; `mcv correct {arguments.run} ...` records what you see"
            ),
        )


class MissesCommand:
    name = "misses"
    summary = "where a looser detection follows something no track covers"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("run", type=Path, help="a run folder")

    def execute(self, arguments: argparse.Namespace) -> SuspectsReport:
        replay = Replay.of(arguments.run)
        misses = find_misses(replay.result.tracks, replay.relaxed(), replay.radius)
        sheet = arguments.run / "misses.png"
        if misses:
            cv2.imwrite(str(sheet), _inspector(replay).sheet(misses))
        return SuspectsReport(
            sheet=sheet,
            suspects=misses,
            next=(
                f"open {sheet}; where a cell is plainly there, `mcv correct"
                f" {arguments.run} force 0 --frame F --x X --y Y --reason ...`"
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
        cv2.imwrite(str(film), _inspector(replay).film(arguments.tracks))
        wanted = set(arguments.tracks)
        return InspectReport(
            film=film,
            verdicts=tuple(
                _numbers(item, replay.scale)
                for item in replay.result.verdicts
                if item.track in wanted
            ),
        )


class GalleryReport(BaseModel):
    gallery: Path
    tracks: tuple[int, ...]


class GalleryCommand:
    name = "gallery"
    summary = "a small view of every track of one motion, to judge them all at once"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("run", type=Path, help="a run folder")
        parser.add_argument("motion", choices=[item.value for item in Motion])
        parser.add_argument("--limit", type=int, default=64, help="tracks shown")

    def execute(self, arguments: argparse.Namespace) -> GalleryReport:
        replay = Replay.of(arguments.run)
        tracks = tuple(
            item.track
            for item in replay.result.verdicts
            if item.motion is Motion(arguments.motion)
        )[: arguments.limit]
        if not tracks:
            raise UsageError(f"no {arguments.motion} track in {arguments.run}")
        gallery = arguments.run / f"gallery_{arguments.motion}.png"
        cv2.imwrite(str(gallery), _inspector(replay).gallery(tracks))
        return GalleryReport(gallery=gallery, tracks=tracks)


class CorrectReport(BaseModel):
    corrections: tuple[Correction, ...]
    summary: RunSummary


class CorrectCommand:
    name = "correct"
    summary = "record a judgement on a track, with its reason, and rewrite the run"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("run", type=Path, help="a run folder")
        parser.add_argument("action", choices=[item.value for item in Action])
        parser.add_argument("tracks", type=int, nargs="+", help="one or more tracks")
        parser.add_argument("--other", type=int, help="join: the track appended")
        parser.add_argument(
            "--frame", type=int, help="split: first frame of the new track"
        )
        parser.add_argument("--motion", choices=[item.value for item in Motion])
        parser.add_argument("--x", type=float, help="force: where the cell is, px")
        parser.add_argument("--y", type=float, help="force: where the cell is, px")
        parser.add_argument("--reason", required=True, help="what you saw")
        parser.add_argument("--by", default="agent", help="who judged")

    def execute(self, arguments: argparse.Namespace) -> CorrectReport:
        folder = RunFolder(arguments.run)
        for track in arguments.tracks:
            corrections = folder.append(
                Correction(
                    action=Action(arguments.action),
                    track=track,
                    other=arguments.other,
                    frame=arguments.frame,
                    motion=Motion(arguments.motion) if arguments.motion else None,
                    x=arguments.x,
                    y=arguments.y,
                    reason=arguments.reason,
                    by=arguments.by,
                )
            )
        replay = Replay.of(arguments.run)
        record = RunWriter(
            arguments.run, str(replay.folder.source), replay.configuration, replay.scale
        ).write(f"{replay.folder.source.name}, corrected", replay.video, replay.result)
        return CorrectReport(corrections=corrections, summary=record.summary)


def _inspector(replay: Replay) -> Inspector:
    return Inspector(replay.video, replay.result, replay.motions)


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
