from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse, Polygon
from numpy.typing import NDArray

from illustrate.style import BLUE, GREY, INK, ORANGE, PINK, blank, panels, save

VIDEO: Final = Path("data/342843.avi")
FRAME: Final = 40
CROP: Final = 18
FILAMENT: Final = 5
WORM_POINTS: Final = 13
BRIGHT: Final = 170

type Points = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Rod:
    """A crop around one rod of the video, and the rod's own pixels in it."""

    crop: np.ndarray
    mask: np.ndarray


def _rod() -> Rod | None:
    """A crop of one dark rod of the study's video, and its mask."""
    if not VIDEO.exists():
        return None
    capture = cv2.VideoCapture(str(VIDEO))
    frames = []
    for _ in range(FRAME + 1):
        frames.append(cv2.cvtColor(capture.read()[1], cv2.COLOR_BGR2GRAY))
    capture.release()
    median = np.median(np.stack(frames), axis=0)
    darker = (median - frames[FRAME].astype(float)) > 12
    count, labels, statistics, centroids = cv2.connectedComponentsWithStats(
        darker.astype(np.uint8)
    )
    height, width = darker.shape
    rods = [
        label
        for label in range(1, count)
        if 18 <= statistics[label][4] <= 80
        and max(statistics[label][2:4]) >= 2 * min(statistics[label][2:4])
        and CROP < centroids[label][0] < width - CROP
        and CROP < centroids[label][1] < height - CROP
        and _clear(median, centroids[label])
    ]
    if not rods:
        return None
    x, y = (round(value) for value in centroids[rods[0]])
    window = (slice(y - CROP, y + CROP), slice(x - CROP, x + CROP))
    return Rod(frames[FRAME][window], labels[window] == rods[0])


def _clear(median: NDArray[np.float64], centre: NDArray[np.float64]) -> bool:
    """No bright bead or host cell in the window around a candidate rod."""
    x, y = (round(value) for value in centre)
    return float(median[y - CROP : y + CROP, x - CROP : x + CROP].max()) < BRIGHT


def _axis(mask: NDArray[np.bool_]) -> tuple[Points, Points, float]:
    """Centre, unit direction of the long axis, and half the length along it."""
    rows, columns = np.nonzero(mask)
    points = np.column_stack((columns, rows)).astype(float)
    centre = points.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov((points - centre).T))
    direction = vectors[:, -1]
    reach = float(np.abs((points - centre) @ direction).max())
    return centre, direction, reach


def _filament(centre: Points, direction: Points, reach: float) -> Points:
    return centre + np.outer(np.linspace(-reach, reach, FILAMENT), direction)


def _real_row(axes: list[Axes]) -> None:
    found = _rod()
    if found is None:
        for item in axes:
            blank(item, "the study's video is not in data/")
        return
    crop = found.crop
    centre, direction, reach = _axis(found.mask)
    titles = ("a rod, as filmed", "point", "oriented point", "filament")
    for index, (item, title) in enumerate(zip(axes, titles, strict=True)):
        item.imshow(crop, cmap="gray", interpolation="nearest", vmin=60, vmax=200)
        blank(item, title)
        if index >= 1:
            item.plot(*centre, "o", color=PINK, markersize=7)
        if index == 2:
            item.arrow(
                *centre,
                *(direction * reach * 1.4),
                color=BLUE,
                width=0.35,
                head_width=1.6,
            )
        if index == 3:
            points = _filament(centre, direction, reach)
            item.plot(
                points[:, 0],
                points[:, 1],
                "-o",
                color=ORANGE,
                markersize=4,
                linewidth=1.5,
            )


def _worm() -> tuple[Points, Points]:
    """A centreline bent by a sine wave, and the outline of a tapered body around it."""
    arc = np.linspace(0, 1, 200)
    centre = np.column_stack((arc * 10, 1.2 * np.sin(2 * np.pi * arc * 1.1)))
    tangent = np.gradient(centre, axis=0)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    width = 0.45 * np.sin(np.pi * np.clip(arc, 0.02, 0.98)) ** 0.5
    outline = np.vstack((
        centre + normal * width[:, None],
        (centre - normal * width[:, None])[::-1],
    ))
    return centre, outline


def _resample(curve: Points, count: int) -> Points:
    lengths = np.concatenate((
        [0.0],
        np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1)),
    ))
    targets = np.linspace(0, lengths[-1], count)
    return np.column_stack([
        np.interp(targets, lengths, curve[:, axis]) for axis in (0, 1)
    ])


def _worm_row(axes: list[Axes]) -> None:
    centre, outline = _worm()
    titles = (
        "a worm's body, seen from above",
        "filament: points equally spaced",
        "outline: points of the contour",
        "ellipse: a centre, two axes, an angle",
    )
    for item, title in zip(axes, titles, strict=True):
        item.add_patch(
            Polygon(outline, closed=True, color=GREY, alpha=0.45, linewidth=0)
        )
        blank(item, title)
        item.set_xlim(-0.8, 10.8)
        item.set_ylim(-3.5, 3.5)
    skeleton = _resample(centre, WORM_POINTS)
    axes[1].plot(skeleton[:, 0], skeleton[:, 1], "-o", color=ORANGE, markersize=4)
    axes[1].plot(*skeleton[0], "o", color=INK, markersize=6)
    contour = _resample(np.vstack((outline, outline[:1])), 40)
    axes[2].plot(contour[:, 0], contour[:, 1], "o", color=BLUE, markersize=3)
    axes[3].add_patch(
        Ellipse((5, 0), 11, 3.4, angle=0, fill=False, color=PINK, linewidth=2)
    )


def draw() -> None:
    figure, axes = panels(4, rows=2, size=3.3)
    _real_row(axes[:4])
    _worm_row(axes[4:])
    save(
        figure,
        "representation",
        "top: a rod of the study's video, three ways to see it;"
        " bottom: a skeleton is one way among several to describe a shape",
    )
