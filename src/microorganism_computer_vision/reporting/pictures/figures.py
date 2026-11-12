from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from motility.diffusion import Motion, Verdict
from motility.pipeline import Scale

from microorganism_computer_vision.reporting.pictures.palette import (
    INK,
    MOTION_HEX,
    MUTED_INK,
    SERIES,
    SURFACE,
)
from microorganism_computer_vision.reporting.population import Population

SIZE: Final = (13, 3.8)
JITTER: Final = 0.12
SHOWN: Final = tuple(motion for motion in Motion if motion is not Motion.UNDETERMINED)
REFERENCES: Final = ((1.0, "diffusive"), (2.0, "ballistic"))


def distributions(verdicts: Sequence[Verdict], scale: Scale, path: Path) -> Path:
    """One dot per classified track, one row per motion: speeds, exponents, counts.

    Dots rather than histograms: a video holds tens of tracks, too few for bins,
    and the row says which motion each value belongs to.
    """
    classified = [item for item in verdicts if item.motion is not Motion.UNDETERMINED]
    figure, (speeds, exponents, counts) = plt.subplots(
        1, 3, figsize=SIZE, sharey=False, gridspec_kw={"width_ratios": (5, 5, 3)}
    )
    figure.patch.set_facecolor(SURFACE)
    _strip(speeds, classified, lambda item: scale.speed(item.speed))
    _style(speeds, f"mean speed per track ({scale.speed_unit})")
    _strip(
        exponents,
        classified,
        lambda item: item.exponent.value if item.exponent else None,
    )
    _style(exponents, "anomalous exponent α, MSD ∝ t^α")
    for value, label in REFERENCES:
        exponents.axvline(value, color=MUTED_INK, linewidth=1, linestyle="--")
        exponents.text(
            value, len(SHOWN) - 0.45, f" {label}", color=MUTED_INK, fontsize=8
        )
    _counts(counts, classified)
    figure.tight_layout()
    figure.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(figure)
    return path


def _strip(
    axes: Axes, classified: Sequence[Verdict], value: Callable[[Verdict], float | None]
) -> None:
    generator = np.random.default_rng(0)
    for row, motion in enumerate(SHOWN):
        values = [
            measured
            for item in classified
            if item.motion is motion and (measured := value(item)) is not None
        ]
        heights = row + generator.uniform(-JITTER, JITTER, len(values))
        axes.scatter(
            values,
            heights,
            s=22,
            color=MOTION_HEX[motion],
            edgecolors=SURFACE,
            linewidths=0.8,
        )
    axes.set_yticks(range(len(SHOWN)), [motion.value for motion in SHOWN])
    axes.set_ylim(len(SHOWN) - 0.5, -0.5)


def _counts(axes: Axes, classified: Sequence[Verdict]) -> None:
    counts = [sum(item.motion is motion for item in classified) for motion in SHOWN]
    colours = [MOTION_HEX[motion] for motion in SHOWN]
    bars = axes.barh(range(len(SHOWN)), counts, color=colours, height=0.6)
    axes.bar_label(bars, padding=3, color=INK, fontsize=9)
    axes.set_yticks(range(len(SHOWN)), [motion.value for motion in SHOWN])
    axes.set_ylim(len(SHOWN) - 0.5, -0.5)
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    _style(axes, f"tracks per motion, {len(classified)} classified")


def _style(axes: Axes, title: str) -> None:
    axes.set_facecolor(SURFACE)
    axes.set_title(title, color=INK, fontsize=10, loc="left")
    axes.tick_params(colors=MUTED_INK, labelsize=8)
    axes.grid(axis="x", color="#e4e3df", linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(MUTED_INK)


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
    _style(axes[0], "motile share")
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
    _style(axes[1], f"swimmers' speed ({populations[0].speed_unit})")
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
    _style(axes[2], "circling tracks, by sense")
    rates = [item.reorientations_per_second or 0.0 for item in populations]
    axes[3].barh(rows, rates, color=SERIES, height=0.6)
    _style(axes[3], "sharp turns per second swimming")
    axes[0].set_yticks(list(rows), names)
    axes[0].set_ylim(len(names) - 0.5, -0.5)
    figure.tight_layout()
    figure.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(figure)
    return path
