from typing import Final

import numpy as np
from matplotlib.patches import Circle
from numpy.typing import NDArray

from illustrate.style import (
    BLUE,
    GREEN,
    GREY,
    INK,
    MUTED,
    RED,
    blank,
    chart,
    panels,
    save,
)

type Array = NDArray[np.float64]

REPULSION: Final = 1.0
ALIGNMENT: Final = 3.0
ATTRACTION: Final = 6.0
OBSTACLE: Final = 4.0
HUG: Final = 0.3


def zones() -> None:
    figure, (picture, curves) = panels(2, size=4.4)
    for radius, colour, label in (
        (ATTRACTION, GREEN, "attraction"),
        (ALIGNMENT, BLUE, "alignment"),
        (REPULSION, RED, "repulsion"),
    ):
        picture.add_patch(Circle((0, 0), radius, color=colour, alpha=0.15))
        picture.text(0, radius - 0.6, label, ha="center", color=colour, fontsize=9)
    picture.arrow(0, 0, 0.9, 0.0, color=INK, width=0.08, head_width=0.35)
    generator = np.random.default_rng(1)
    for _ in range(14):
        place = generator.uniform(-6.5, 6.5, 2)
        angle = generator.uniform(0, 2 * np.pi)
        picture.arrow(
            *place,
            0.7 * np.cos(angle),
            0.7 * np.sin(angle),
            color=GREY,
            width=0.05,
            head_width=0.25,
        )
    blank(picture, "the zones around a cell")
    picture.set_xlim(-7, 7)
    picture.set_ylim(-7, 7)
    distance = np.linspace(0, 8, 300)
    curves.plot(
        distance,
        -np.exp(-((distance / REPULSION) ** 2) * 2),
        color=RED,
        label="repulsion",
    )
    curves.plot(
        distance,
        np.exp(-(((distance - ALIGNMENT) / 1.2) ** 2)),
        color=BLUE,
        label="alignment",
    )
    curves.plot(
        distance,
        0.6 * np.exp(-(((distance - ATTRACTION) / 1.5) ** 2)),
        color=GREEN,
        label="attraction",
    )
    curves.axhline(0, color=MUTED, linewidth=0.8)
    curves.legend(fontsize=8, frameon=False)
    chart(
        curves,
        "how much a neighbour weighs, by distance",
        "distance to the neighbour (cell lengths)",
        "weight",
    )
    save(
        figure,
        "interactions-zones",
        "each neighbour pushes, turns or pulls a cell, by how far it is",
    )


def _vicsek(
    noise: float, steps: int, generator: np.random.Generator
) -> tuple[Array, Array, Array]:
    """Cells at constant speed take their neighbours' mean heading, plus noise."""
    count, size, speed, radius = 120, 25.0, 0.3, 2.0
    positions = generator.uniform(0, size, (count, 2))
    headings = generator.uniform(-np.pi, np.pi, count)
    start, order = (positions.copy(), headings.copy()), []
    for _ in range(steps):
        offset = positions[:, None] - positions[None]
        offset -= size * np.round(offset / size)
        near = np.hypot(offset[..., 0], offset[..., 1]) < radius
        mean = np.angle(near @ np.exp(1j * headings))
        headings = mean + generator.uniform(-noise / 2, noise / 2, count)
        positions = (
            positions + speed * np.column_stack((np.cos(headings), np.sin(headings)))
        ) % size
        order.append(abs(np.mean(np.exp(1j * headings))))
    return (
        np.vstack((start[0], positions)),
        np.concatenate((start[1], headings)),
        np.array(order),
    )


def alignment() -> None:
    figure, (before, after, curve) = panels(3, size=4.0)
    generator = np.random.default_rng(2)
    stacked, headings, quiet = _vicsek(0.5, 250, generator)
    half = len(stacked) // 2
    for axes, rows, title in (
        (before, slice(0, half), "at the start"),
        (after, slice(half, None), "250 steps later"),
    ):
        axes.quiver(
            stacked[rows, 0],
            stacked[rows, 1],
            np.cos(headings[rows]),
            np.sin(headings[rows]),
            color=BLUE,
            scale=30,
            width=0.006,
        )
        blank(axes, title)
    _, _, loud = _vicsek(4.0, 250, np.random.default_rng(3))
    curve.plot(quiet, color=BLUE, label="little noise: they align")
    curve.plot(loud, color=GREY, label="much noise: they do not")
    curve.legend(fontsize=8, frameon=False)
    curve.set_ylim(0, 1.05)
    chart(curve, "polar order", "steps", "|mean heading|")
    save(
        figure,
        "interactions-alignment",
        "each cell takes the mean heading of the cells within two lengths",
    )


def chase() -> None:
    figure, (axes,) = panels(1, size=4.4)
    generator = np.random.default_rng(5)
    prey, hunter = np.array([0.0, 0.0]), np.array([-10.0, -6.0])
    heading, prey_path, hunter_path = 0.0, [], []
    for _ in range(260):
        heading += generator.normal(0, 0.15)
        prey = prey + 0.25 * np.array([np.cos(heading), np.sin(heading)])
        towards = prey - hunter
        hunter = hunter + 0.28 * towards / np.hypot(*towards)
        prey_path.append(prey.copy())
        hunter_path.append(hunter.copy())
    for path, colour, label in (
        (np.array(prey_path), BLUE, "the chased"),
        (np.array(hunter_path), RED, "the chaser"),
    ):
        axes.plot(path[:, 0], path[:, 1], color=colour, linewidth=1.4, label=label)
        axes.plot(*path[0], "o", color=colour)
    axes.legend(fontsize=8, frameon=False)
    blank(axes, "one cell steers towards another")
    save(
        figure,
        "interactions-chase",
        "a chase: velocity correlated with the direction to the other, at a lag",
    )


def draw() -> None:
    zones()
    alignment()
    chase()
