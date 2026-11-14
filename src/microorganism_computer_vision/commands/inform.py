import argparse
import json
from pathlib import Path
from typing import Final

from motility.diffusion import Motion
from motility.diffusion.evidence import Evidence, Hypothesis
from motility.diffusion.evidence.context_tests import (
    SurrogateTest,
    near_share,
    speed_slope,
)
from motility.tracking import Trajectory
from motility.truth.synthetic import SceneParameters
from pydantic import BaseModel

from microorganism_computer_vision.configuration import Configuration
from microorganism_computer_vision.media.folders import RUNS
from microorganism_computer_vision.media.replay import Replay
from microorganism_computer_vision.media.run_folder import (
    HistoryEntry,
    Workspace,
    source_key,
)
from microorganism_computer_vision.media.video import UsageError
from microorganism_computer_vision.reporting.pictures.charts.context import (
    NullPanel,
    context,
)
from microorganism_computer_vision.reporting.pictures.charts.population import (
    population as population_figure,
)
from microorganism_computer_vision.reporting.pictures.palette import MOTION_HEX
from microorganism_computer_vision.reporting.population import MOTILE, Population

GUIDE: Final = Path("docs/guide.md")
PASSIVE: Final = frozenset({Motion.DIFFUSIVE, Motion.CONFINED})


class GuideCommand:
    name = "guide"
    summary = "how an agent calibrates the toolkit: read this first"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """The guide takes no argument."""

    def execute(self, _arguments: argparse.Namespace, /) -> str:
        return GUIDE.read_text()


class SchemaCommand:
    name = "schema"
    summary = "every parameter with its description and bounds, as JSON Schema"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--scene", action="store_true", help="the synthetic scene's instead"
        )

    def execute(self, arguments: argparse.Namespace) -> str:
        model = SceneParameters if arguments.scene else Configuration
        return json.dumps(model.model_json_schema(), indent=2, ensure_ascii=False)


class History(BaseModel):
    source: str
    runs: tuple[HistoryEntry, ...]


class HistoryCommand:
    name = "history"
    summary = "the runs of a source, each with the reason it was made"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "source", type=Path, help="the video, as given to `mcv run`"
        )

    def execute(self, arguments: argparse.Namespace) -> History:
        stem = source_key(arguments.source)
        return History(source=stem, runs=Workspace(RUNS, stem).history())


class Statistics(BaseModel):
    figure: Path
    populations: tuple[Population, ...]


class StatsCommand:
    name = "stats"
    summary = "the behaviour of every cell across runs: motile share, speeds, chirality"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("runs", type=Path, nargs="+", help="run folders to compare")
        parser.add_argument(
            "--name", default="population", help="the folder of the output"
        )

    def execute(self, arguments: argparse.Namespace) -> Statistics:
        populations = tuple(Population.of(Replay.of(run)) for run in arguments.runs)
        folder = RUNS.parent / "stats" / arguments.name
        folder.mkdir(parents=True, exist_ok=True)
        statistics = Statistics(
            figure=population_figure(populations, folder / "population.png"),
            populations=populations,
        )
        (folder / "stats.json").write_text(statistics.model_dump_json(indent=2))
        return statistics


class Interactions(BaseModel):
    figure: Path
    evidence: tuple[Evidence, ...]


class InteractionsCommand:
    name = "interactions"
    summary = "test whether the decor holds or slows the cells, against surrogates"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "run", type=Path, help="a run folder whose mask marks the decor"
        )
        parser.add_argument("--radius", type=float, default=10.0, help="near, in px")

    def execute(self, arguments: argparse.Namespace) -> Interactions:
        replay = Replay.of(arguments.run)
        objects = replay.result.ignored
        if objects is None:
            raise UsageError("the run has no mask: there is no decor to test against")
        groups = {
            "swimmers": (MOTILE, MOTION_HEX[Motion.DIRECTED]),
            "non-motile": (PASSIVE, MOTION_HEX[Motion.DIFFUSIVE]),
        }
        tests = {
            Hypothesis.TRAPPED: (
                SurrogateTest(near_share(arguments.radius), True),
                "share near objects",
            ),
            Hypothesis.SLOWED: (SurrogateTest(speed_slope, False), "px/frame per px"),
        }
        panels, evidence = [], []
        for group, (motions, colour) in groups.items():
            tracks = _of_motions(replay, motions)
            for hypothesis, (test, unit) in tests.items():
                outcome = test.run(tracks, objects)
                found = test.evidence(hypothesis, tracks, objects)
                if outcome is None or found is None:
                    continue
                evidence.append(found.model_copy(update={"subject": group}))
                panels.append(
                    NullPanel(
                        f"{group}: {hypothesis.value}",
                        unit,
                        outcome.observed,
                        outcome.null,
                        found.p_value or 1.0,
                        colour,
                    )
                )
        if not panels:
            raise UsageError("no classified track to test")
        figure = context(panels, arguments.run / "interactions.png")
        return Interactions(figure=figure, evidence=tuple(evidence))


def _of_motions(replay: Replay, motions: frozenset[Motion]) -> list[Trajectory]:
    wanted = {item.track for item in replay.result.verdicts if item.motion in motions}
    return [
        item for item in replay.result.tracks.trajectories() if item.index in wanted
    ]
