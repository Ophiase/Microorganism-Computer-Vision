from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from microorganism_computer_vision.reporting.pictures.charts.style import style
from microorganism_computer_vision.reporting.pictures.palette import (
    INK,
    MUTED_INK,
    SURFACE,
)

PANEL: Final = (4.2, 3.2)
NULL: Final = "#b9b8b2"


@dataclass(frozen=True, slots=True)
class NullPanel:
    """One test against its surrogates: their spread, the real value, its rank."""

    title: str
    unit: str
    observed: float
    null: NDArray[np.float64]
    p_value: float
    colour: str


def context(panels: Sequence[NullPanel], path: Path) -> Path:
    """Each test's surrogates as a histogram, the real tracks' value as a line."""
    figure, axes = plt.subplots(
        1, len(panels), figsize=(PANEL[0] * len(panels), PANEL[1])
    )
    figure.patch.set_facecolor(SURFACE)
    for item, panel in zip(np.atleast_1d(axes), panels, strict=True):
        item.hist(panel.null, bins=25, color=NULL, edgecolor=SURFACE)
        item.axvline(panel.observed, color=panel.colour, linewidth=2.5)
        item.text(
            panel.observed,
            item.get_ylim()[1] * 0.92,
            f"  real, p = {panel.p_value:.3f}",
            color=INK,
            fontsize=8,
        )
        style(item, panel.title)
        item.set_xlabel(panel.unit, color=MUTED_INK, fontsize=8)
    figure.suptitle(
        "grey: the same tracks moved at random across the frame; line: the real ones",
        color=MUTED_INK,
        fontsize=9,
        x=0.01,
        ha="left",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(figure)
    return path
