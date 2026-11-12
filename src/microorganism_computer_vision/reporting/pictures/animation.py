from collections.abc import Mapping
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from motility.diffusion import Motion
from motility.pipeline import PipelineResult
from motility.vision import Video
from PIL import Image as PillowImage

from microorganism_computer_vision.reporting.pictures.overlay import (
    TAIL_OPACITY,
    TrackOverlay,
)
from microorganism_computer_vision.reporting.pictures.palette import (
    LOST,
    OVERLAY_COLOURS,
)

PANEL_WIDTH: Final = 480
GAP: Final = 4
MAX_FRAMES: Final = 48
FRAME_MILLISECONDS: Final = 120
TAIL_WIDTH: Final = 3
BLENDS: Final = (48, 96, 144, 192)
GREYS: Final = 24


def animate(
    video: Video, result: PipelineResult, motions: Mapping[int, Motion], path: Path
) -> Path:
    """The video as it is beside the same with its cells marked, as a GIF.

    Long clips keep one frame in n. Every frame shares one palette: grey levels
    for the video, the verdict colours for the boxes, and those colours blended
    over a few greys for the translucent tails; the marks are drawn after the
    downscale and without antialiasing, so they stay sharp.
    """
    step = max(1, -(-len(video) // MAX_FRAMES))
    scale = min(1.0, PANEL_WIDTH / video.shape[2])
    overlay = TrackOverlay(
        result, motions, scale=scale, thickness=TAIL_WIDTH, antialiased=False
    )
    palette = _palette()
    images = [
        _quantized(_pair(video, overlay, frame), palette)
        for frame in range(0, len(video), step)
    ]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=FRAME_MILLISECONDS,
        loop=0,
        optimize=True,
    )
    return path


def _pair(video: Video, overlay: TrackOverlay, frame: int) -> np.ndarray:
    """The raw frame, a white gap, the marked frame."""
    marked = overlay.draw(video[frame], frame)
    raw = cv2.resize(
        (np.clip(video[frame], 0, 1) * 255).astype(np.uint8),
        (marked.shape[1], marked.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    gap = np.full((marked.shape[0], GAP, 3), 255, dtype=np.uint8)
    return np.hstack((cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR), gap, marked))


def _palette() -> PillowImage.Image:
    levels = np.linspace(0, 255, GREYS).round().astype(int)
    greys = [(level, level, level) for level in levels]
    marks = [
        (red, green, blue) for blue, green, red in (*OVERLAY_COLOURS.values(), LOST)
    ]
    blends = [
        tuple(
            round(TAIL_OPACITY * channel + (1 - TAIL_OPACITY) * grey)
            for channel in mark
        )
        for mark in marks
        for grey in BLENDS
    ]
    entries = [channel for colour in (*greys, *marks, *blends) for channel in colour]
    palette = PillowImage.new("P", (1, 1))
    palette.putpalette(entries + entries[-3:] * (256 - len(entries) // 3))
    return palette


def _quantized(image: np.ndarray, palette: PillowImage.Image) -> PillowImage.Image:
    rgb = PillowImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return rgb.quantize(palette=palette, dither=PillowImage.Dither.NONE)
