from collections.abc import Sequence
from pathlib import Path
from typing import Final

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from motility.diffusion import Motion

from microorganism_computer_vision.reporting.pictures.charts.style import style
from microorganism_computer_vision.reporting.pictures.palette import (
    INK,
    MOTION_HEX,
    SERIES,
    SURFACE,
)
from microorganism_computer_vision.reporting.population import Population

SIZE: Final = (13, 3.8)
JITTER: Final = 0.12
SHOWN: Final = tuple(motion for motion in Motion if motion is not Motion.UNDETERMINED)
REFERENCES: Final = ((1.0, "diffusive"), (2.0, "ballistic"))


def population(populations: Sequence[Population], path: Path) -> Path:
    """One row per run: motile share, swimmers' speeds, chirality, sharp turns."""
    names = [item.source for item in populations]
    figure, axes = plt.subplots(
        1,
        4,
        figsize=(15, 0.5 * len(names) + 1.8),
        sharey=True,
        gridspec_kw={"width_ratios": (3, 5, 3, 3)},
    )
    figure.patch.set_facecolor(SURFACE)
    rows = range(len(names))
    shares = [item.motile_share for item in populations]
    bars = axes[0].barh(rows, shares, color=SERIES, height=0.6)
    axes[0].bar_label(
        bars,
        labels=[
            f"{value:.0%} of {item.classified}"
            for value, item in zip(shares, populations, strict=True)
        ],
        padding=3,
        fontsize=8,
        color=INK,
    )
    axes[0].set_xlim(0, 1.3)
    style(axes[0], "motile share")
    generator = np.random.default_rng(0)
    for row, item in enumerate(populations):
        speeds = item.motile_speeds
        axes[1].scatter(
            speeds,
            row + generator.uniform(-JITTER, JITTER, len(speeds)),
            s=10,
            color=MOTION_HEX[Motion.DIRECTED],
            edgecolors=SURFACE,
            linewidths=0.5,
        )
    style(axes[1], f"swimmers' speed ({populations[0].speed_unit})")
    clockwise = [item.clockwise for item in populations]
    anticlockwise = [item.anticlockwise for item in populations]
    axes[2].barh(
        rows,
        clockwise,
        color=MOTION_HEX[Motion.CIRCULAR],
        height=0.6,
        label="clockwise",
    )
    axes[2].barh(
        rows,
        anticlockwise,
        left=clockwise,
        color=SURFACE,
        edgecolor=MOTION_HEX[Motion.CIRCULAR],
        height=0.6,
        label="anticlockwise",
    )
    axes[2].legend(fontsize=8, frameon=False, loc="lower right")
    style(axes[2], "circling tracks, by sense")
    rates = [item.reorientations_per_second or 0.0 for item in populations]
    axes[3].barh(rows, rates, color=SERIES, height=0.6)
    style(axes[3], "sharp turns per second swimming")
    axes[0].set_yticks(list(rows), names)
    axes[0].set_ylim(len(names) - 0.5, -0.5)
    figure.tight_layout()
    figure.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(figure)
    return path
