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
    figure, axes = plt.subplots(figsize=(9, 0.45 * len(behaviours) + 1.6))
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
    axes.set_xticks(range(len(SHOWN)), [test.value for test in SHOWN], rotation=30)
    axes.set_yticks(
        range(len(behaviours)),
        [
            f"{item.value} (false alarms)" if item is NULL else item.value
            for item in behaviours
        ],
    )
    axes.tick_params(colors=MUTED_INK, labelsize=8, length=0)
    axes.xaxis.tick_top()
    for side in axes.spines.values():
        side.set_visible(False)
    axes.set_title(
        "How often each test says yes, on cells whose behaviour is known",
        color=INK,
        fontsize=10,
        loc="left",
        pad=40,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(figure)
    return path


def _box(row: int, column: int, colour: str) -> Rectangle:
    return Rectangle(
        (column - 0.5, row - 0.5), 1, 1, fill=False, edgecolor=colour, lw=2.2
    )
