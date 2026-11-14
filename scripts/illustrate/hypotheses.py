from dataclasses import dataclass
from typing import Final

import numpy as np
from matplotlib.axes import Axes
from motility.diffusion.evidence import TRACK_TESTS, Hypothesis
from motility.tracking import Trajectory
from motility.truth.synthetic.behaviours import Behaviour, behave_oriented
from motility.truth.synthetic.scene import Kinetics

from illustrate.style import BLUE, GREY, INK, ORANGE, blank, chart, panels, save

FRAMES: Final = 100
TRACKS: Final = 100
SHOWN: Final = 4
NOISE: Final = 0.3
AXIS_NOISE: Final = 0.15
KINETICS: Final = Kinetics()


@dataclass(frozen=True, slots=True)
class Card:
    """A hypothesis drawn: its null, its alternative, and the test's statistic."""

    slug: str
    hypothesis: Hypothesis
    null: Behaviour
    alternative: Behaviour
    statistic: str


CARDS: Final = (
    Card(
        "anomalous",
        Hypothesis.ANOMALOUS,
        Behaviour.BROWNIAN,
        Behaviour.SWIMMER,
        "likelihood ratio, chi-square",
    ),
    Card(
        "persistent",
        Hypothesis.PERSISTENT,
        Behaviour.BROWNIAN,
        Behaviour.SWIMMER,
        "t of the mean cos of turns",
    ),
    Card(
        "chiral",
        Hypothesis.CHIRAL,
        Behaviour.SWIMMER,
        Behaviour.CIRCLING,
        "t of the mean signed turn",
    ),
    Card(
        "tumbling",
        Hypothesis.TUMBLING,
        Behaviour.SWIMMER,
        Behaviour.RUN_AND_TUMBLE,
        "sharp turns counted",
    ),
    Card(
        "rhythmic",
        Hypothesis.PERIODIC,
        Behaviour.RUN_AND_TUMBLE,
        Behaviour.RUN_AND_REVERSE,
        "variation of the intervals",
    ),
    Card(
        "confined",
        Hypothesis.CONFINED,
        Behaviour.BROWNIAN,
        Behaviour.CONFINED,
        "Akaike weight of saturation",
    ),
    Card(
        "heavy-tails",
        Hypothesis.NON_GAUSSIAN,
        Behaviour.BROWNIAN,
        Behaviour.WAITING,
        "Shapiro-Wilk W",
    ),
    Card(
        "body-frame",
        Hypothesis.BODY_FRAME,
        Behaviour.BROWNIAN,
        Behaviour.BODY_FRAME,
        "D along over D across",
    ),
)


def _tracks(behaviour: Behaviour, generator: np.random.Generator) -> list[Trajectory]:
    """Tracks seen through the observation chain: noisy centres and axes."""
    tracks = []
    for index in range(TRACKS):
        path = behave_oriented(behaviour, KINETICS, FRAMES, generator)
        seen = path.positions + generator.normal(0, NOISE, path.positions.shape)
        angles = path.angles + generator.normal(0, AXIS_NOISE, len(path.angles))
        tracks.append(
            Trajectory(
                index,
                tuple(range(FRAMES)),
                tuple(map(tuple, seen.tolist())),
                tuple(angles.tolist()),
                2.5,
            )
        )
    return tracks


def _statistics(card: Card, tracks: list[Trajectory]) -> list[float]:
    test = TRACK_TESTS[card.hypothesis]
    values = [
        evidence.statistic for track in tracks if (evidence := test(track)) is not None
    ]
    return [value for value in values if np.isfinite(value)]


def _paths(axes: Axes, tracks: list[Trajectory], colour: str, title: str) -> None:
    for track in tracks[:SHOWN]:
        positions = np.array(track.positions)
        axes.plot(
            positions[:, 0], positions[:, 1], color=colour, linewidth=1, alpha=0.85
        )
        axes.plot(*positions[0], "o", color=INK, markersize=2.5)
    blank(axes, title)


def draw_card(card: Card, seed: int) -> None:
    generator = np.random.default_rng(seed)
    nulls, alternatives = (
        _tracks(card.null, generator),
        _tracks(card.alternative, generator),
    )
    figure, (left, middle, right) = panels(3, size=3.6)
    _paths(left, nulls, GREY, f"H0: {card.null.value}")
    _paths(middle, alternatives, BLUE, f"H1: {card.alternative.value}")
    for values, colour, label in (
        (_statistics(card, nulls), GREY, "H0"),
        (_statistics(card, alternatives), ORANGE, "H1"),
    ):
        if values:
            right.hist(values, bins=25, color=colour, alpha=0.65, label=label)
    right.legend(fontsize=8, frameon=False)
    chart(right, "the test's statistic", card.statistic, "tracks")
    save(
        figure,
        f"hypothesis-{card.slug}",
        f"{TRACKS} tracks of each, {FRAMES} frames, 0.3 px noise, as the pipeline sees",
    )


def draw() -> None:
    for seed, card in enumerate(CARDS):
        draw_card(card, seed)
