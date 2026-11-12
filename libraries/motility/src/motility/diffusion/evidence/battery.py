from collections.abc import Sequence
from typing import Final, Protocol

from motility.diffusion.evidence.motion_tests import anomalous, confined, non_gaussian
from motility.diffusion.evidence.result import Evidence, Hypothesis, benjamini_hochberg
from motility.diffusion.evidence.turning_tests import (
    chiral,
    periodic,
    persistent,
    tumbling,
)
from motility.tracking.trajectories import Trajectory

LEVEL: Final = 0.05
WEIGHT: Final = 0.9


class TrackTest(Protocol):
    def __call__(self, trajectory: Trajectory, /) -> Evidence | None: ...


TRACK_TESTS: Final[dict[Hypothesis, TrackTest]] = {
    Hypothesis.ANOMALOUS: anomalous,
    Hypothesis.PERSISTENT: persistent,
    Hypothesis.CHIRAL: chiral,
    Hypothesis.NON_GAUSSIAN: non_gaussian,
    Hypothesis.TUMBLING: tumbling,
    Hypothesis.PERIODIC: periodic,
    Hypothesis.CONFINED: confined,
}


def test_tracks(trajectories: Sequence[Trajectory]) -> tuple[Evidence, ...]:
    """Every hypothesis on the tracks it applies to, corrected for how many those are.

    Tumbling asks whether a swimmer turns more sharply than its rotational
    diffusion allows: it is tested only on the tracks found persistent first.
    """
    results: list[Evidence] = []
    for hypothesis, test in TRACK_TESTS.items():
        eligible = _eligible(hypothesis, trajectories, results)
        found = [item for item in (test(trajectory) for trajectory in eligible) if item]
        tested = [item for item in found if item.p_value is not None]
        adjusted = benjamini_hochberg([item.p_value or 0.0 for item in tested])
        results += [
            item.model_copy(update={"q_value": q})
            for item, q in zip(tested, adjusted, strict=True)
        ]
        results += [item for item in found if item.p_value is None]
    return tuple(results)


def _eligible(
    hypothesis: Hypothesis, trajectories: Sequence[Trajectory], results: list[Evidence]
) -> Sequence[Trajectory]:
    if hypothesis is not Hypothesis.TUMBLING:
        return trajectories
    swimmers = {
        item.subject
        for item in results
        if item.hypothesis == Hypothesis.PERSISTENT and supports(item)
    }
    return [item for item in trajectories if str(item.index) in swimmers]


def supports(evidence: Evidence, level: float = LEVEL) -> bool:
    """Whether a result rejects its null: q below `level`, or a weight above 0.9."""
    if evidence.q_value is not None:
        return evidence.q_value < level
    if evidence.p_value is None:
        return evidence.statistic > WEIGHT
    return evidence.p_value < level
