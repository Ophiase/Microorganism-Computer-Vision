import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tomli_w
from motility.pipeline import Scale, twin
from motility.truth.synthetic import Kinetics, Population, SceneParameters, simulate
from motility.vision.appearance import AppearanceReport
from pydantic import BaseModel

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
        parser.add_argument(
            "--write",
            action="store_true",
            help="write a run folder per scene, with looks",
        )

    def execute(self, arguments: argparse.Namespace) -> BenchReport:
        configuration = configuration_of(arguments)
        bench = (
            Bench({path.stem: load_scene(path) for path in arguments.scene})
            if arguments.scene
            else Bench.from_folder(BENCH_SCENES / arguments.split)
        )
        if arguments.write:
            for name in bench.names:
                scored = bench.run(name, configuration)
                scale = Scale.from_acquisition(configuration.acquisition, None)
                RunWriter(BENCH_RUNS / name, configuration, scale).write(
                    f"bench scene {name}", bench.simulation(name).video, scored.result
                )
        return bench.score(configuration)
