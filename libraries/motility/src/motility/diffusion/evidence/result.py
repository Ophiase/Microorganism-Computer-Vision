from collections.abc import Sequence
from enum import StrEnum
from typing import Self

import numpy as np
from pydantic import BaseModel, Field


class Hypothesis(StrEnum):
    """An alternative to a null model of motion, each with the test that decides it."""

    ANOMALOUS = "anomalous"
    PERSISTENT = "persistent"
    CHIRAL = "chiral"
    NON_GAUSSIAN = "non-gaussian"
    TUMBLING = "tumbling"
    PERIODIC = "periodic"
    CONFINED = "confined"

    @property
    def null(self) -> str:
        return NULLS[self]


NULLS: dict[Hypothesis, str] = {
    Hypothesis.ANOMALOUS: "the MSD grows linearly: alpha = 1 (t-test on its slope)",
    Hypothesis.PERSISTENT: "no memory of heading: mean cos of turns = 0 (t-test, > 0)",
    Hypothesis.CHIRAL: "no preferred sense of turning: mean signed turn = 0 (t-test)",
    Hypothesis.NON_GAUSSIAN: "steps are Gaussian (Shapiro-Wilk on standardised steps)",
    Hypothesis.TUMBLING: (
        "turns are rotational diffusion: sharp turns as rare as a Gaussian of the"
        " turns' robust spread allows (one-sided binomial)"
    ),
    Hypothesis.PERIODIC: (
        "turns come in no rhythm: the periodogram's peak is no higher than in"
        " shuffled turns (permutation test)"
    ),
    Hypothesis.CONFINED: (
        "the MSD keeps growing: a saturating law is not preferred over a linear"
        " one (Akaike weight, no p-value)"
    ),
}


class Evidence(BaseModel):
    """The outcome of one test on one track or one population."""

    hypothesis: str
    subject: str = Field(description="The track number, or the run for a population.")
    statistic: float
    p_value: float | None = Field(
        description="Under the null; None for a model weight."
    )
    q_value: float | None = Field(
        default=None, description="Benjamini-Hochberg across the run's tracks."
    )
    effect: float = Field(description="The size of the departure from the null.")
    samples: int

    @classmethod
    def of_test(
        cls,
        hypothesis: Hypothesis,
        subject: str,
        statistic: float,
        p_value: float | None,
        effect: float,
        samples: int,
    ) -> Self:
        return cls(
            hypothesis=hypothesis.value,
            subject=subject,
            statistic=float(statistic),
            p_value=None if p_value is None else float(p_value),
            effect=float(effect),
            samples=samples,
        )


def benjamini_hochberg(p_values: Sequence[float]) -> tuple[float, ...]:
    """False discovery rate adjusted p-values, in the order given."""
    if not p_values:
        return ()
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return tuple(result.tolist())
