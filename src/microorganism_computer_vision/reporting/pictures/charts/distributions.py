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

from microorganism_computer_vision.reporting.pictures.charts.style import style
from microorganism_computer_vision.reporting.pictures.palette import (
    INK,
    MOTION_HEX,
    MUTED_INK,
    SURFACE,
)

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
    style(speeds, f"mean speed per track ({scale.speed_unit})")
    _strip(
        exponents,
        classified,
        lambda item: item.exponent.value if item.exponent else None,
    )
    style(exponents, "anomalous exponent α, MSD ∝ t^α")
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
    style(axes, f"tracks per motion, {len(classified)} classified")
