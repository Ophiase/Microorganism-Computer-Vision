from typing import Final

import numpy as np
from pydantic import BaseModel, Field

from motility.diffusion.evidence import Evidence, Hypothesis, supports, test_tracks
from motility.tracking.trajectories import Trajectory
from motility.truth.synthetic.behaviours import Behaviour, behave
from motility.truth.synthetic.scene import Kinetics

LOCALISATION: Final = 0.3


class RejectionRates(BaseModel):
    """How often each test rejects its null on cells whose true behaviour is known.

    On a behaviour that satisfies a test's null, the rate is its false positive
    rate and should sit near the level; on one that breaks it, it is the
    test's power.
    """

    frames: int
    cells: int
    rates: dict[Behaviour, dict[Hypothesis, float]] = Field(
        description="True behaviour, then hypothesis, then the share rejecting."
    )


def rejection_rates(
    kinetics: Kinetics, frames: int = 80, cells: int = 100, seed: int = 0
) -> RejectionRates:
    """Run the battery on `cells` tracks of each behaviour, localisation noise added."""
    generator = np.random.default_rng(seed)
    rates = {}
    for behaviour in Behaviour:
        trajectories = [
            _noisy(behave(behaviour, kinetics, frames, generator), index, generator)
            for index in range(cells)
        ]
        evidence = test_tracks(trajectories)
        rates[behaviour] = {
            hypothesis: _share([
                item for item in evidence if item.hypothesis == hypothesis
            ])
            for hypothesis in Hypothesis
        }
    return RejectionRates(frames=frames, cells=cells, rates=rates)


def _noisy(
    positions: np.ndarray, index: int, generator: np.random.Generator
) -> Trajectory:
    observed = positions + generator.normal(0, LOCALISATION, positions.shape)
    return Trajectory(
        index, tuple(range(len(observed))), tuple(map(tuple, observed.tolist()))
    )


def _share(evidence: list[Evidence]) -> float:
    return sum(supports(item) for item in evidence) / len(evidence) if evidence else 0.0
