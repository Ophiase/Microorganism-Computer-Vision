from collections.abc import Mapping
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from motility.diffusion import Motion
from motility.pipeline import PipelineResult
from motility.vision import Video
from PIL import Image as PillowImage

from microorganism_computer_vision.reporting.pictures.looks import tracks_view
from microorganism_computer_vision.reporting.pictures.palette import MOTION_HEX

MAX_WIDTH: Final = 560
MAX_FRAMES: Final = 48
FRAME_MILLISECONDS: Final = 120
TAIL_WIDTH: Final = 2
COLOURS: Final = 32
SWATCH: Final = 8


def animate(
    video: Video, result: PipelineResult, motions: Mapping[int, Motion], path: Path
) -> Path:
    """The video with its tracks drawn over it, coloured by verdict, as a GIF.

    Long clips keep one frame in n, and every frame shares one small palette,
    so the file stays near three megabytes; the palette is drawn with a swatch of
    the verdict colours, which come out exact.
    """
    step = max(1, -(-len(video) // MAX_FRAMES))
    frames = [
        PillowImage.fromarray(_tracked(video, result, motions, frame))
        for frame in range(0, len(video), step)
    ]
    palette = _palette(frames[len(frames) // 2])
    images = [
        frame.quantize(palette=palette, dither=PillowImage.Dither.NONE)
        for frame in frames
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


def _palette(frame: PillowImage.Image) -> PillowImage.Image:
    swatch = PillowImage.new("RGB", (SWATCH * len(MOTION_HEX), SWATCH))
    for index, colour in enumerate(MOTION_HEX.values()):
        swatch.paste(colour, (index * SWATCH, 0, (index + 1) * SWATCH, SWATCH))
    sample = PillowImage.new("RGB", (frame.width, frame.height + SWATCH))
    sample.paste(frame, (0, 0))
    sample.paste(swatch, (0, frame.height))
    return sample.quantize(colors=COLOURS, method=PillowImage.Quantize.MEDIANCUT)


def _tracked(
    video: Video, result: PipelineResult, motions: Mapping[int, Motion], frame: int
) -> np.ndarray:
    image = tracks_view(video[frame], result, motions, frame, thickness=TAIL_WIDTH)
    scale = min(1.0, MAX_WIDTH / image.shape[1])
    if scale < 1.0:
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
