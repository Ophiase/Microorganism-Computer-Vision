import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tomli_w
from motility.pipeline import Scale, twin
from motility.truth.evaluation.power import RejectionRates, rejection_rates
from motility.truth.synthetic import Kinetics, Population, SceneParameters, simulate
from motility.vision.measures.appearance import AppearanceReport
from pydantic import BaseModel

from microorganism_computer_vision.benchmark.search import (
    RandomSearch,
    SearchReport,
    Strategy,
)
from microorganism_computer_vision.benchmark.space import SearchSpace
from microorganism_computer_vision.benchmark.suite import Bench, BenchReport, load_scene
from microorganism_computer_vision.commands.base import (
    add_configuration,
    configuration_of,
)
from microorganism_computer_vision.configuration import override
from microorganism_computer_vision.media.folders import BENCH_SCENES, RUNS
from microorganism_computer_vision.media.tables import write_tracks
from microorganism_computer_vision.reporting.writer import RunWriter

SYNTHETIC = RUNS.parent / "synthetic"
BENCH_RUNS = RUNS.parent / "bench"


class SynthReport(BaseModel):
    frames: Path
    truth: Path
    scene: Path
    next: str


class SynthCommand:
    name = "synth"
    summary = "render a synthetic video with its truth, from a scene or like a probe"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("name", help="the folder name under output/synthetic/")
        source = parser.add_mutually_exclusive_group()
        source.add_argument("--scene", type=Path, help="a scene TOML file")
        source.add_argument(
            "--like", type=Path, help="an appearance.json from `mcv probe`: its twin"
        )
        parser.add_argument(
            "--set",
            action="append",
            default=[],
            metavar="SECTION.KEY=VALUE",
            help="change one scene parameter, e.g. population.directed=30",
        )

    def execute(self, arguments: argparse.Namespace) -> SynthReport:
        scene = override(self._scene(arguments), arguments.set)
        folder = SYNTHETIC / arguments.name
        frames = folder / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        simulation = simulate(scene)
        images = np.round(simulation.video * 255).astype(np.uint8)
        for index in range(len(images)):
            cv2.imwrite(str(frames / f"{index:04d}.png"), images[index])
        truth = write_tracks(folder / "truth.csv", simulation.truth.table)
        motions = {
            str(cell): motion.value for cell, motion in simulation.truth.motions.items()
        }
        (folder / "motions.json").write_text(json.dumps(motions, indent=2))
        scene_path = folder / "scene.toml"
        scene_path.write_text(tomli_w.dumps(scene.model_dump(mode="json")))
        return SynthReport(
            frames=frames,
            truth=truth,
            scene=scene_path,
            next=f"`mcv bench --scene {scene_path}` scores a configuration on it",
        )

    @staticmethod
    def _scene(arguments: argparse.Namespace) -> SceneParameters:
        if arguments.scene:
            return load_scene(arguments.scene)
        if arguments.like:
            appearance = AppearanceReport.model_validate_json(
                arguments.like.read_text()
            )
            return twin(appearance, Population(), Kinetics())
        return SceneParameters()


class BenchCommand:
    name = "bench"
    summary = "score a configuration on synthetic scenes against their truth"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        add_configuration(parser)
        _scenes(parser)
        parser.add_argument(
            "--write",
            action="store_true",
            help="write a run folder per scene, with looks",
        )

    def execute(self, arguments: argparse.Namespace) -> BenchReport:
        configuration = configuration_of(arguments)
        bench = _bench(arguments)
        if arguments.write:
            for name in bench.names:
                scored = bench.run(name, configuration)
                scale = Scale.from_acquisition(configuration.acquisition, None)
                RunWriter(
                    BENCH_RUNS / name, f"scene:{name}", configuration, scale
                ).write(
                    f"bench scene {name}", bench.simulation(name).video, scored.result
                )
        return bench.score(configuration)


class SearchCommand:
    name = "search"
    summary = "the baseline: a random search with the truth as its score"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        add_configuration(parser)
        parser.add_argument(
            "--strategy", choices=[item.value for item in Strategy], default="blind"
        )
        parser.add_argument(
            "--budget", type=int, default=30, help="configurations tried"
        )
        parser.add_argument("--seed", type=int, default=0)
        _scenes(parser)
        parser.add_argument(
            "--out", type=Path, help="write the best configuration there"
        )

    def execute(self, arguments: argparse.Namespace) -> SearchReport:
        search = RandomSearch(
            _bench(arguments),
            SearchSpace(),
            Strategy(arguments.strategy),
        )
        report = search.run(
            configuration_of(arguments), arguments.budget, arguments.seed
        )
        if arguments.out:
            report.best_configuration.save(arguments.out)
        return report


def _scenes(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--split",
        default="tuning",
        help="a folder of settings/bench/: tuning or evaluation",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        action="append",
        default=[],
        help="a scene TOML; repeatable",
    )


def _bench(arguments: argparse.Namespace) -> Bench:
    if arguments.scene:
        return Bench({path.stem: load_scene(path) for path in arguments.scene})
    return Bench.from_folder(BENCH_SCENES / arguments.split)


class ValidateCommand:
    name = "validate"
    summary = "each hypothesis test's false positives and power on known behaviours"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--frames", type=int, default=100, help="frames per track")
        parser.add_argument(
            "--cells", type=int, default=100, help="tracks per behaviour"
        )
        parser.add_argument("--seed", type=int, default=0)

    def execute(self, arguments: argparse.Namespace) -> RejectionRates:
        return rejection_rates(
            Kinetics(),
            frames=arguments.frames,
            cells=arguments.cells,
            seed=arguments.seed,
        )
