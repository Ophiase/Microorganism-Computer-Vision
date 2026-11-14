from pathlib import Path
from typing import Final

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from motility.diffusion.evidence import Hypothesis
from motility.truth.evaluation.power import RejectionRates
from motility.truth.synthetic.behaviours import Behaviour

from microorganism_computer_vision.reporting.pictures.palette import (
    INK,
    MUTED_INK,
    SURFACE,
)

MADE_FOR: Final = {
    Behaviour.RUN_AND_TUMBLE: Hypothesis.TUMBLING,
    Behaviour.RUN_AND_REVERSE: Hypothesis.PERIODIC,
    Behaviour.CIRCLING: Hypothesis.CHIRAL,
    Behaviour.CONFINED: Hypothesis.CONFINED,
    Behaviour.SUBDIFFUSIVE: Hypothesis.ANOMALOUS,
    Behaviour.WAITING: Hypothesis.NON_GAUSSIAN,
    Behaviour.SWIMMER: Hypothesis.PERSISTENT,
    Behaviour.BODY_FRAME: Hypothesis.BODY_FRAME,
}
"""The test built to find each behaviour: its cell is boxed, and should be dark."""

NULL: Final = Behaviour.BROWNIAN
"""A cell that only diffuses: every test should stay silent on it."""

IN_WORDS: Final = {
    Behaviour.BROWNIAN: "only diffuses: false alarms",
    Behaviour.STATIONARY: "does not move",
    Behaviour.SWIMMER: "swims straight",
    Behaviour.RUN_AND_TUMBLE: "runs, then turns sharply",
    Behaviour.RUN_AND_REVERSE: "runs back and forth",
    Behaviour.CIRCLING: "swims in circles",
    Behaviour.CONFINED: "held in a cage",
    Behaviour.SUBDIFFUSIVE: "spreads slower than diffusion",
    Behaviour.WAITING: "diffuses with long pauses",
    Behaviour.LEVY: "runs of any length",
    Behaviour.BODY_FRAME: "a rod diffusing along its axis",
}
"""Each simulated behaviour as a reader says it, without the physicist's name."""

ASKS: Final = {
    Hypothesis.ANOMALOUS: "spreads at\nan odd rate",
    Hypothesis.PERSISTENT: "keeps its\nheading",
    Hypothesis.CHIRAL: "turns\none way",
    Hypothesis.NON_GAUSSIAN: "steps of\nuneven size",
    Hypothesis.TUMBLING: "turns\nsharply",
    Hypothesis.PERIODIC: "has a\nrhythm",
    Hypothesis.CONFINED: "stays in\na cage",
    Hypothesis.BODY_FRAME: "moves along\nits body",
}
"""What each test asks of a track, under its name."""

SAYS_YES: Final = LinearSegmentedColormap.from_list("yes", ["#fcfcfb", "#2a78d6"])
SHOWN: Final = (
    Hypothesis.ANOMALOUS,
    Hypothesis.PERSISTENT,
    Hypothesis.CHIRAL,
    Hypothesis.NON_GAUSSIAN,
    Hypothesis.TUMBLING,
    Hypothesis.PERIODIC,
    Hypothesis.CONFINED,
    Hypothesis.BODY_FRAME,
)


def power(rates: RejectionRates, path: Path) -> Path:
    """How often each test says yes on cells of known behaviour, a cell per pair.

    Dark is yes. The top row is a diffusing cell, where every cell should stay
    white; a boxed cell is the test made for that behaviour, and should be dark.
    """
    behaviours = [NULL, *(item for item in rates.rates if item is not NULL)]
    shares = np.array([
        [rates.rates[behaviour].get(test, 0.0) for test in SHOWN]
        for behaviour in behaviours
    ])
    figure, axes = plt.subplots(figsize=(10, 0.45 * len(behaviours) + 2.0))
    figure.patch.set_facecolor(SURFACE)
    axes.imshow(shares, cmap=SAYS_YES, vmin=0, vmax=1, aspect="auto")
    for row, behaviour in enumerate(behaviours):
        for column, test in enumerate(SHOWN):
            share = shares[row, column]
            axes.text(
                column,
                row,
                f"{share:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color=SURFACE if share > 0.6 else MUTED_INK,
            )
            if MADE_FOR.get(behaviour) is test:
                axes.add_patch(_box(row, column, INK))
    axes.add_patch(
        Rectangle((-0.5, -0.5), len(SHOWN), 1, fill=False, edgecolor="#e34948", lw=2)
    )
    axes.set_xticks(
        range(len(SHOWN)), [f"{test.value}\n{ASKS[test]}" for test in SHOWN]
    )
    axes.set_yticks(
        range(len(behaviours)), [IN_WORDS.get(item, item.value) for item in behaviours]
    )
    axes.set_ylabel("the simulated cell truly...", color=INK, fontsize=9)
    axes.set_xlabel("the hypothesis tested", color=INK, fontsize=9, labelpad=8)
    axes.xaxis.set_label_position("top")
    axes.tick_params(colors=MUTED_INK, labelsize=8, length=0)
    axes.xaxis.tick_top()
    for side in axes.spines.values():
        side.set_visible(False)
    figure.suptitle(
        "How often each test says yes, over 100 simulated tracks per row",
        color=INK,
        fontsize=11,
        x=0.02,
        ha="left",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(figure)
    return path


def _box(row: int, column: int, colour: str) -> Rectangle:
    return Rectangle(
        (column - 0.5, row - 0.5), 1, 1, fill=False, edgecolor=colour, lw=2.2
    )
