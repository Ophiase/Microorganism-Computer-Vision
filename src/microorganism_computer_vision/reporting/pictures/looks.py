from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np
from cv2.typing import MatLike
from motility.diffusion import Motion
from motility.pipeline import PipelineResult
from motility.vision import DetectionParameters, Frame, Video, mask

from microorganism_computer_vision.reporting.pictures.overlay import TrackOverlay

type Image = MatLike
"""A colour picture, rows by columns by blue, green, red bytes."""

ZOOM_SCALE: Final = 3


@dataclass(frozen=True, slots=True)
class Zoom:
    """A square of the frame to enlarge in every view: its centre and its side, px."""

    x: int
    y: int
    size: int

    def apply(self, image: Image) -> Image:
        half = self.size // 2
        top, left = max(self.y - half, 0), max(self.x - half, 0)
        window = image[top : top + self.size, left : left + self.size]
        return cv2.resize(
            window, None, fx=ZOOM_SCALE, fy=ZOOM_SCALE, interpolation=cv2.INTER_NEAREST
        )


TAIL = 15
BANNER = 22
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)


def look(
    video: Video,
    result: PipelineResult,
    detection: DetectionParameters,
    motions: Mapping[int, Motion],
    frame: int,
    zoom: Zoom | None = None,
) -> Image:
    """Four views of one frame, what an expert checks: raw, foreground, mask, tracks.

    The tracks are drawn as their last positions, coloured by their verdict,
    over the raw frame; red boxes are this frame's detections.
    """
    views = {
        f"raw, frame {frame}": grey(video[frame]),
        "foreground": grey(result.foreground[frame]),
        "mask": grey(mask(result.foreground[frame], detection) / 255),
        "cells and tracks": TrackOverlay(result, motions).draw(video[frame], frame),
    }
    tiles = [
        _titled(zoom.apply(image) if zoom else image, title)
        for title, image in views.items()
    ]
    return np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))


def grey(frame: Frame) -> Image:
    bytes_frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(bytes_frame, cv2.COLOR_GRAY2BGR)


def _titled(image: Image, title: str) -> Image:
    banner = np.zeros((BANNER, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, title, (6, 16), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    return np.vstack((banner, image))


def probe_look(video: Video) -> Image:
    """What a video holds before any parameter: a frame, what stays, what moves.

    The motion view is the largest departure of each pixel from its median over
    the clip, stretched to the full range: swimmers draw their paths, dust
    stays black.
    """
    background = np.median(video, axis=0)
    departure = np.max(np.abs(video - background), axis=0)
    views = {
        "first frame": grey(video[0]),
        "median background": grey(background),
        "motion: largest departure from the median": grey(_stretched(departure)),
        "spread over time": grey(_stretched(np.std(video, axis=0))),
    }
    tiles = [_titled(image, title) for title, image in views.items()]
    return np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))


def _stretched(frame: np.ndarray) -> np.ndarray:
    high = float(np.percentile(frame, 99.5))
    return np.clip(frame / high, 0, 1) if high > 0 else frame
