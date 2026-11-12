import argparse
from pathlib import Path

from microorganism_computer_vision.benchmark.search import (
    RandomSearch,
    SearchReport,
    Strategy,
)
from microorganism_computer_vision.benchmark.space import SearchSpace
from microorganism_computer_vision.benchmark.suite import Bench, load_scene
from microorganism_computer_vision.commands.base import (
    add_configuration,
    configuration_of,
)
from microorganism_computer_vision.media.folders import BENCH_SCENES


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
        parser.add_argument("--split", default="tuning")
        parser.add_argument(
            "--scene",
            type=Path,
            action="append",
            default=[],
            help="search on these scene TOML files instead of a split; repeatable",
        )
        parser.add_argument(
            "--out", type=Path, help="write the best configuration there"
        )

    def execute(self, arguments: argparse.Namespace) -> SearchReport:
        bench = (
            Bench({path.stem: load_scene(path) for path in arguments.scene})
            if arguments.scene
            else Bench.from_folder(BENCH_SCENES / arguments.split)
        )
        search = RandomSearch(
            bench,
            SearchSpace(),
            Strategy(arguments.strategy),
        )
        report = search.run(
            configuration_of(arguments), arguments.budget, arguments.seed
        )
        if arguments.out:
            report.best_configuration.save(arguments.out)
        return report
