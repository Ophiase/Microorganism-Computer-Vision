from enum import StrEnum
from typing import Final

import numpy as np
from pydantic import BaseModel, Field

from microorganism_computer_vision.benchmark.space import SearchSpace
from microorganism_computer_vision.benchmark.suite import Bench
from microorganism_computer_vision.configuration import Configuration

REPORTED_TRIALS: Final = 10


class Strategy(StrEnum):
    BLIND = "blind"
    AROUND = "around"


class Trial(BaseModel):
    index: int
    objective: float
    mota: float
    accuracy: float


class SearchReport(BaseModel):
    """What a search tried and the best it found, for the same budget as anyone."""

    strategy: Strategy
    budget: int
    seed: int
    best: Trial
    best_configuration: Configuration
    curve: tuple[float, ...] = Field(description="The best objective after each trial.")
    top: tuple[Trial, ...]


class RandomSearch:
    """The baseline an agent must beat: draws and keeps the best on the bench.

    `blind` draws every knob uniformly over the space; `around` takes steps from
    the current best, starting at the base configuration, as a local search does.
    """

    def __init__(self, bench: Bench, space: SearchSpace, strategy: Strategy) -> None:
        self._bench = bench
        self._space = space
        self._strategy = strategy

    def run(self, base: Configuration, budget: int, seed: int) -> SearchReport:
        generator = np.random.default_rng(seed)
        best_configuration, best = base, self._trial(0, base)
        trials, curve = [best], [best.objective]
        for index in range(1, budget):
            candidate = self._propose(best_configuration, base, generator)
            trial = self._trial(index, candidate)
            trials.append(trial)
            if trial.objective > best.objective:
                best_configuration, best = candidate, trial
            curve.append(best.objective)
        ranked = sorted(trials, key=lambda trial: -trial.objective)[:REPORTED_TRIALS]
        reason = f"{self._strategy.value} search, best of {budget} on the bench"
        return SearchReport(
            strategy=self._strategy,
            budget=budget,
            seed=seed,
            best=best,
            best_configuration=best_configuration.model_copy(update={"reason": reason}),
            curve=tuple(curve),
            top=tuple(ranked),
        )

    def _propose(
        self, best: Configuration, base: Configuration, generator: np.random.Generator
    ) -> Configuration:
        if self._strategy is Strategy.BLIND:
            return self._space.blind(base, generator)
        return self._space.around(best, generator)

    def _trial(self, index: int, configuration: Configuration) -> Trial:
        report = self._bench.score(configuration)
        return Trial(
            index=index,
            objective=report.objective,
            mota=report.mota,
            accuracy=report.accuracy,
        )
