import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Self

from motility.pipeline import Evaluation, PipelineParameters, ScoredRun, evaluate
from motility.truth.synthetic import SceneParameters, Simulation, simulate
from pydantic import BaseModel, Field

from microorganism_computer_vision.media.video import UsageError

MIN_RADIUS: Final = 4.0


class SceneScore(BaseModel):
    scene: str
    objective: float = Field(description="Mean of the MOTA and the accuracy.")
    evaluation: Evaluation


class BenchReport(BaseModel):
    """A configuration scored on every scene of a split, and on their mean."""

    objective: float = Field(
        description=(
            "Mean over the scenes of (MOTA + classification accuracy) / 2: the"
            " tracks must be right, and so must the verdicts drawn from them."
        )
    )
    mota: float
    accuracy: float
    scenes: tuple[SceneScore, ...]


class Bench:
    """A set of synthetic scenes, each simulated once, that scores configurations."""

    def __init__(self, scenes: Mapping[str, SceneParameters]) -> None:
        self._scenes = dict(scenes)
        self._simulations: dict[str, Simulation] = {}

    @classmethod
    def from_folder(cls, folder: Path) -> Self:
        paths = sorted(folder.glob("*.toml"))
        if not paths:
            raise UsageError(f"no scene in {folder}")
        return cls({path.stem: load_scene(path) for path in paths})

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._scenes)

    def simulation(self, name: str) -> Simulation:
        if name not in self._simulations:
            self._simulations[name] = simulate(self._scenes[name])
        return self._simulations[name]

    def run(self, name: str, parameters: PipelineParameters) -> ScoredRun:
        radius = max(self._scenes[name].appearance.length, MIN_RADIUS)
        return evaluate(self.simulation(name), parameters, radius)

    def score(self, parameters: PipelineParameters) -> BenchReport:
        scores = tuple(
            _scene_score(name, self.run(name, parameters).evaluation)
            for name in self._scenes
        )
        count = len(scores)
        return BenchReport(
            objective=sum(score.objective for score in scores) / count,
            mota=sum(score.evaluation.tracking.mota for score in scores) / count,
            accuracy=sum(score.evaluation.classification.accuracy for score in scores)
            / count,
            scenes=scores,
        )


def load_scene(path: Path) -> SceneParameters:
    with path.open("rb") as file:
        return SceneParameters.model_validate(tomllib.load(file))


def _scene_score(name: str, evaluation: Evaluation) -> SceneScore:
    objective = (evaluation.tracking.mota + evaluation.classification.accuracy) / 2
    return SceneScore(scene=name, objective=objective, evaluation=evaluation)
