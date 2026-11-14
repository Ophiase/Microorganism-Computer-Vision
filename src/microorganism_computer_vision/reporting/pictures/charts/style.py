from matplotlib.axes import Axes

from microorganism_computer_vision.reporting.pictures.palette import (
    INK,
    MUTED_INK,
    SURFACE,
)


def style(axes: Axes, title: str) -> None:
    axes.set_facecolor(SURFACE)
    axes.set_title(title, color=INK, fontsize=10, loc="left")
    axes.tick_params(colors=MUTED_INK, labelsize=8)
    axes.grid(axis="x", color="#e4e3df", linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(MUTED_INK)
