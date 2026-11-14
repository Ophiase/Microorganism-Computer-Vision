from typing import Final

import numpy as np
from matplotlib.patches import Circle
from numpy.typing import NDArray

from illustrate.style import (
    BLUE,
    GREEN,
    INK,
    ORANGE,
    PINK,
    YELLOW,
    blank,
    panels,
    save,
)

type Array = NDArray[np.float64]

REPULSION: Final = 1.0
ALIGNMENT: Final = 3.0
ATTRACTION: Final = 6.0
OBSTACLE: Final = 4.0
HUG: Final = 0.3


def _swim_near_obstacle(follow: bool, generator: np.random.Generator) -> Array:
    """A swimmer sent towards the obstacle: it turns along the edge, or bounces off."""
    position = np.array([-12.0, generator.uniform(-3, 3)])
    heading, path = generator.normal(0, 0.15), []
    for _ in range(320):
        distance = np.hypot(*position) - OBSTACLE
        if follow and distance < 1.2:
            tangent = np.arctan2(position[1], position[0]) + np.pi / 2 + HUG
            heading += 0.4 * np.angle(np.exp(1j * (tangent - heading)))
        heading += generator.normal(0, 0.05)
        step = position + 0.2 * np.array([np.cos(heading), np.sin(heading)])
        if np.hypot(*step) <= OBSTACLE:
            heading += np.pi + generator.normal(0, 0.5)
            continue
        position = step
        path.append(position.copy())
    return np.array(path)


def _chemotaxis(generator: np.random.Generator) -> Array:
    """Run-and-tumble whose tumbles are rarer while climbing towards the source at 0."""
    position, heading, path = (
        generator.uniform(-15, 15, 2),
        generator.uniform(0, 2 * np.pi),
        [],
    )
    for _ in range(900):
        climbing = np.dot(-position, [np.cos(heading), np.sin(heading)]) > 0
        if generator.random() < (0.01 if climbing else 0.06):
            heading = generator.uniform(0, 2 * np.pi)
        position = position + 0.2 * np.array([np.cos(heading), np.sin(heading)])
        path.append(position.copy())
    return np.array(path)


def decor() -> None:
    figure, (control, follow, gradient) = panels(3, size=4.0)
    generator = np.random.default_rng(4)
    for axes, follows, colour, title in (
        (control, False, ORANGE, "no interaction"),
        (follow, True, YELLOW, "turning to the edge"),
    ):
        axes.add_patch(Circle((0, 0), OBSTACLE, color=PINK, alpha=0.5))
        for _ in range(5):
            path = _swim_near_obstacle(follows, generator)
            axes.plot(path[:, 0], path[:, 1], color=colour, linewidth=1.1, alpha=0.85)
            axes.plot(*path[0], "o", color=INK, markersize=2.5)
        blank(axes, f"near an obstacle: {title}")
        axes.set_xlim(-14, 14)
        axes.set_ylim(-14, 14)
    gradient.scatter([0], [0], s=300, color=GREEN, alpha=0.4)
    for _ in range(4):
        path = _chemotaxis(generator)
        gradient.plot(path[:, 0], path[:, 1], color=BLUE, linewidth=0.9, alpha=0.7)
    blank(gradient, "tumbling less while climbing to a source")
    save(
        figure,
        "interactions-decor",
        "the decor changes the law: an edge traps, a gradient attracts",
    )


def draw() -> None:
    decor()
