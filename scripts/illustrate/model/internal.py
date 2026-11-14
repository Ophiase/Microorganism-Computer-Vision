from typing import Final

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from numpy.typing import NDArray

from illustrate.style import BLUE, ORANGE, PINK, YELLOW, blank, chart, panels, save

type Array = NDArray[np.float64]

FRAMES: Final = 600
LEAVE_SWIMMING: Final = 0.01
LEAVE_STUCK: Final = 0.02
SPEED: Final = 1.5
BEHAVIOURS: Final = (ORANGE, BLUE, YELLOW)


def _switching(generator: np.random.Generator) -> tuple[Array, NDArray[np.bool_]]:
    """A two-state hidden Markov chain: swimming, or stuck and only jittering."""
    swimming, heading, position = True, 0.0, np.zeros(2)
    path, states = [], []
    for _ in range(FRAMES):
        if generator.random() < (LEAVE_SWIMMING if swimming else LEAVE_STUCK):
            swimming = not swimming
            heading = generator.uniform(0, 2 * np.pi)
        heading += generator.normal(0, 0.08)
        if swimming:
            position = position + SPEED * np.array([np.cos(heading), np.sin(heading)])
        position = position + generator.normal(0, 0.2, 2)
        path.append(position.copy())
        states.append(swimming)
    return np.array(path), np.array(states)


def switching() -> None:
    figure, (picture, timeline) = panels(2, size=4.4)
    path, states = _switching(np.random.default_rng(1))
    segments = np.stack((path[:-1], path[1:]), axis=1)
    colours = [BLUE if state else PINK for state in states[1:]]
    picture.add_collection(
        LineCollection(list(segments), colors=colours, linewidths=1.3)
    )
    for stop in _stops(states):
        picture.scatter(
            *path[stop.start], s=6 * (stop.stop - stop.start), color=PINK, alpha=0.6
        )
    picture.autoscale()
    blank(picture, "blue: swimming; pink discs: stuck, by how long")
    timeline.fill_between(
        range(FRAMES), states.astype(float), step="mid", color=BLUE, alpha=0.6
    )
    timeline.set_yticks([0, 1], ["stuck", "swimming"])
    chart(timeline, "the hidden state over time", "frame", "")
    save(
        figure,
        "internal-switching",
        "a hidden two-state chain decides which law the cell follows",
    )


def _stops(states: NDArray[np.bool_]) -> list[slice]:
    """The runs of frames in the stuck state."""
    edges = np.flatnonzero(np.diff(np.concatenate(([1], states.astype(int), [1]))))
    return [
        slice(int(start), int(stop))
        for start, stop in zip(edges[::2], edges[1::2], strict=True)
    ]


def _weights(generator: np.random.Generator) -> Array:
    """Softmax of three slow Ornstein-Uhlenbeck drifts: weights that sum to one."""
    latent, values = np.zeros(3), []
    for _ in range(FRAMES):
        latent = latent - 0.01 * latent + generator.normal(0, 0.12, 3)
        exponentials = np.exp(2 * latent)
        values.append(exponentials / exponentials.sum())
    return np.array(values)


def _mixed_path(weights: Array, generator: np.random.Generator) -> Array:
    """Each step mixes a Brownian kick, a straight and a turning swim, by weight."""
    heading, position, path = 0.0, np.zeros(2), []
    for brownian, swimmer, circler in weights:
        heading += circler * 0.12 + generator.normal(0, 0.03)
        swim = (
            (swimmer + circler) * SPEED * np.array([np.cos(heading), np.sin(heading)])
        )
        position = position + swim + brownian * generator.normal(0, 1.5, 2)
        path.append(position.copy())
    return np.array(path)


def mixture() -> None:
    figure, (picture, area) = panels(2, size=4.4)
    generator = np.random.default_rng(2)
    weights = _weights(generator)
    path = _mixed_path(weights, generator)
    rgb = np.array([to_rgb(colour) for colour in BEHAVIOURS])
    colours = weights[1:] @ rgb
    segments = np.stack((path[:-1], path[1:]), axis=1)
    picture.add_collection(
        LineCollection(list(segments), colors=np.clip(colours, 0, 1), linewidths=1.3)
    )
    picture.autoscale()
    blank(picture, "the path, coloured by its mixture")
    area.stackplot(
        range(FRAMES),
        weights.T,
        colors=BEHAVIOURS,
        labels=("Brownian", "swimming", "circling"),
        alpha=0.85,
    )
    area.legend(fontsize=8, frameon=False, loc="upper right")
    area.set_ylim(0, 1)
    chart(area, "the weights of the three behaviours", "frame", "weight")
    save(
        figure,
        "internal-mixture",
        "a slow hidden vector, normalised into weights, mixes three basic laws",
    )


def draw() -> None:
    switching()
    mixture()
