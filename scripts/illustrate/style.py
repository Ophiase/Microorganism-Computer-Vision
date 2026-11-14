from collections.abc import Sequence
from pathlib import Path
from typing import Final

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

OUTPUT: Final = Path("docs/figures/model")
SURFACE: Final = "#fcfcfb"
INK: Final = "#0b0b0b"
MUTED: Final = "#52514e"
GRID: Final = "#e4e3df"
BLUE: Final = "#2a78d6"
ORANGE: Final = "#eb6834"
GREEN: Final = "#1baf7a"
YELLOW: Final = "#eda100"
PINK: Final = "#e87ba4"
VIOLET: Final = "#6250d6"
RED: Final = "#e34948"
GREY: Final = "#8c8b86"


def panels(columns: int, rows: int = 1, size: float = 3.6) -> tuple[Figure, list[Axes]]:
    """A row of square panels on the surface, their frames faint."""
    figure, axes = plt.subplots(rows, columns, figsize=(size * columns, size * rows))
    figure.patch.set_facecolor(SURFACE)
    flat = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
    for item in flat:
        item.set_facecolor(SURFACE)
        for side in item.spines.values():
            side.set_color(GRID)
    return figure, flat


def blank(axes: Axes, title: str) -> None:
    """A panel for a drawing: equal axes, no ticks, a title on the left."""
    axes.set_title(title, color=INK, fontsize=10, loc="left")
    axes.set_aspect("equal", adjustable="datalim")
    axes.set_xticks([])
    axes.set_yticks([])


def chart(axes: Axes, title: str, x_label: str, y_label: str) -> None:
    """A panel for a curve: a faint grid, quiet axes."""
    axes.set_title(title, color=INK, fontsize=10, loc="left")
    axes.set_xlabel(x_label, color=MUTED, fontsize=9)
    axes.set_ylabel(y_label, color=MUTED, fontsize=9)
    axes.tick_params(colors=MUTED, labelsize=8)
    axes.grid(color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)


def save(figure: Figure, name: str, caption: str | None = None) -> Path:
    if caption:
        figure.suptitle(caption, color=MUTED, fontsize=9, x=0.01, ha="left")
    figure.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"{name}.png"
    figure.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(figure)
    return path


def shades(count: int) -> Sequence[float]:
    """Opacities from strong to faint, so one path in a bundle stands out."""
    return [1.0] + [0.35] * (count - 1)
