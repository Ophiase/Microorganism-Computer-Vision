from typing import Final

import numpy as np
from matplotlib.axes import Axes
from motility.truth.synthetic.behaviours import Behaviour, behave
from motility.truth.synthetic.scene import Kinetics
from numpy.typing import NDArray

from illustrate.style import (
    BLUE,
    GREEN,
    GREY,
    INK,
    MUTED,
    ORANGE,
    YELLOW,
    blank,
    chart,
    panels,
    save,
    shades,
)

FRAMES: Final = 240
BUNDLE: Final = 6
KINETICS: Final = Kinetics(
    diffusion=0.6,
    speed=2.0,
    rotational_diffusion=0.003,
    angular_speed=0.05,
    confinement=4.0,
)

type Path2D = NDArray[np.float64]


def body_frame(
    frames: int, generator: np.random.Generator
) -> tuple[Path2D, NDArray[np.float64]]:
    """Kicks four times larger along a slowly turning axis than across it."""
    axis = np.cumsum(generator.normal(0, np.sqrt(2 * 0.002), frames))
    along = generator.normal(0, np.sqrt(2 * 1.6), frames)
    across = generator.normal(0, np.sqrt(2 * 0.1), frames)
    steps = np.column_stack((
        along * np.cos(axis) - across * np.sin(axis),
        along * np.sin(axis) + across * np.cos(axis),
    ))
    return np.cumsum(steps, axis=0), axis


def _bundle(axes: Axes, paths: list[Path2D], colour: str, title: str) -> None:
    for path, alpha in zip(paths, shades(len(paths)), strict=True):
        axes.plot(path[:, 0], path[:, 1], color=colour, linewidth=1.1, alpha=alpha)
        axes.plot(*path[0], "o", color=INK, markersize=2.5, alpha=alpha)
    blank(axes, title)


def _paths(behaviour: Behaviour, seed: int) -> list[Path2D]:
    generator = np.random.default_rng(seed)
    return [behave(behaviour, KINETICS, FRAMES, generator) for _ in range(BUNDLE)]


def passive() -> None:
    figure, (brownian, body, confined) = panels(3)
    _bundle(brownian, _paths(Behaviour.BROWNIAN, 1), ORANGE, "Brownian: equal kicks")
    generator = np.random.default_rng(2)
    rods = [body_frame(FRAMES, generator) for _ in range(BUNDLE)]
    _bundle(body, [path for path, _ in rods], ORANGE, "body-frame: kicks along the rod")
    path, axis = rods[0]
    for index in range(0, FRAMES, 30):
        x, y = path[index]
        dx, dy = 2.5 * np.cos(axis[index]), 2.5 * np.sin(axis[index])
        body.plot(
            [x - dx, x + dx],
            [y - dy, y + dy],
            color=INK,
            linewidth=3,
            solid_capstyle="round",
        )
    wells = [
        behave(Behaviour.CONFINED, KINETICS, FRAMES, generator) + offset
        for offset in ((0, 0), (24, 6), (8, 22), (30, 28))
    ]
    _bundle(confined, wells, GREEN, "confined: pulled back to a centre")
    save(
        figure,
        "models-passive",
        f"{BUNDLE} cells, {FRAMES} frames each; black bars: the rod's axis",
    )


def active() -> None:
    figure, (swimmer, chiral, tumble, reverse) = panels(4)
    _bundle(swimmer, _paths(Behaviour.SWIMMER, 3), BLUE, "active swimmer")
    _bundle(chiral, _paths(Behaviour.CIRCLING, 4), YELLOW, "chiral swimmer")
    _bundle(tumble, _paths(Behaviour.RUN_AND_TUMBLE, 5), BLUE, "run-and-tumble")
    _bundle(reverse, _paths(Behaviour.RUN_AND_REVERSE, 6)[:2], BLUE, "run-and-reverse")
    save(
        figure,
        "models-active",
        "self-propelled at the same speed; only the turning differs",
    )


def anomalous() -> None:
    figure, (fractional, waiting, levy) = panels(3)
    _bundle(
        fractional, _paths(Behaviour.SUBDIFFUSIVE, 7)[:3], GREY, "fractional, alpha 0.6"
    )
    generator = np.random.default_rng(8)
    path = behave(Behaviour.WAITING, KINETICS, FRAMES, generator)
    stops, counts = np.unique(path, axis=0, return_counts=True)
    waiting.plot(path[:, 0], path[:, 1], color=GREY, linewidth=1.1)
    waiting.scatter(stops[:, 0], stops[:, 1], s=counts * 3, color=MUTED, alpha=0.5)
    blank(waiting, "waiting times: discs grow with the wait")
    _bundle(
        levy, _paths(Behaviour.LEVY, 9)[:3], GREY, "Lévy walk: heavy-tailed flights"
    )
    save(figure, "models-anomalous", "the same number of frames as above")


def msd() -> None:
    figure, (axes,) = panels(1, size=5.2)
    lags = np.unique(np.logspace(0, np.log10(FRAMES // 4), 20).astype(int))
    curves = {
        Behaviour.BROWNIAN: ORANGE,
        Behaviour.SWIMMER: BLUE,
        Behaviour.CONFINED: GREEN,
        Behaviour.SUBDIFFUSIVE: GREY,
        Behaviour.CIRCLING: YELLOW,
    }
    for behaviour, colour in curves.items():
        paths = _paths(behaviour, 10) + _paths(behaviour, 11)
        values = [
            np.mean([np.mean(np.sum((p[lag:] - p[:-lag]) ** 2, axis=1)) for p in paths])
            for lag in lags
        ]
        axes.loglog(lags, values, color=colour, linewidth=2, label=behaviour.value)
    for slope, label in ((1, "alpha 1"), (2, "alpha 2")):
        axes.loglog(
            lags,
            2 * lags.astype(float) ** slope,
            color=MUTED,
            linestyle="--",
            linewidth=0.8,
        )
        axes.text(lags[-1], 2 * lags[-1] ** slope, f" {label}", color=MUTED, fontsize=8)
    axes.legend(fontsize=8, frameon=False)
    chart(axes, "mean squared displacement", "lag (frames)", "MSD (px²)")
    save(figure, "models-msd", "each law's signature: its slope on a log-log plot")


def draw() -> None:
    passive()
    active()
    anomalous()
    msd()
