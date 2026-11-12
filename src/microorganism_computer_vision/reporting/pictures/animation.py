from collections.abc import Mapping
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from motility.diffusion import Motion
from motility.pipeline import PipelineResult
from motility.vision import Video
from PIL import Image as PillowImage

from microorganism_computer_vision.reporting.pictures.overlay import TrackOverlay
from microorganism_computer_vision.reporting.pictures.palette import OVERLAY_COLOURS

MAX_WIDTH: Final = 640
MAX_FRAMES: Final = 48
FRAME_MILLISECONDS: Final = 120
TAIL_WIDTH: Final = 2
GREYS: Final = 24


def animate(
    video: Video, result: PipelineResult, motions: Mapping[int, Motion], path: Path
) -> Path:
    """The video with a box and a tail on every cell, coloured by verdict, as a GIF.

    Long clips keep one frame in n. Every frame shares one palette: grey levels
    for the video and the exact verdict colours for the marks, which are drawn
    after the downscale and without antialiasing, so they stay sharp and true.
    """
    step = max(1, -(-len(video) // MAX_FRAMES))
    scale = min(1.0, MAX_WIDTH / video.shape[2])
    overlay = TrackOverlay(
        result, motions, scale=scale, thickness=TAIL_WIDTH, antialiased=False
    )
    palette = _palette()
    images = [
        _quantized(overlay.draw(video[frame], frame), palette)
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


def _palette() -> PillowImage.Image:
    levels = np.linspace(0, 255, GREYS).round().astype(int)
    greys = [(level, level, level) for level in levels]
    marks = [(red, green, blue) for blue, green, red in OVERLAY_COLOURS.values()]
    entries = [channel for colour in (*greys, *marks) for channel in colour]
    palette = PillowImage.new("P", (1, 1))
    palette.putpalette(entries + entries[-3:] * (256 - len(entries) // 3))
    return palette


def _quantized(image: np.ndarray, palette: PillowImage.Image) -> PillowImage.Image:
    rgb = PillowImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return rgb.quantize(palette=palette, dither=PillowImage.Dither.NONE)
