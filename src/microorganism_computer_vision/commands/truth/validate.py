import argparse
from pathlib import Path

from motility.truth.evaluation.power import RejectionRates, rejection_rates
from motility.truth.synthetic import Kinetics
from pydantic import BaseModel

from microorganism_computer_vision.media.folders import RUNS
from microorganism_computer_vision.reporting.pictures.charts.power import power


class Validation(BaseModel):
    figure: Path
    rates: RejectionRates


class ValidateCommand:
    name = "validate"
    summary = "each hypothesis test's false positives and power on known behaviours"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--frames", type=int, default=100, help="frames per track")
        parser.add_argument(
            "--cells", type=int, default=100, help="tracks per behaviour"
        )
        parser.add_argument("--seed", type=int, default=0)

    def execute(self, arguments: argparse.Namespace) -> Validation:
        rates = rejection_rates(
            Kinetics(),
            frames=arguments.frames,
            cells=arguments.cells,
            seed=arguments.seed,
        )
        folder = RUNS.parent / "validate"
        folder.mkdir(parents=True, exist_ok=True)
        return Validation(figure=power(rates, folder / "power.png"), rates=rates)
