from collections.abc import Mapping
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from motility.diffusion import Motion
from motility.pipeline import PipelineResult
from motility.vision import Video
from PIL import Image as PillowImage

from microorganism_computer_vision.reporting.pictures.looks import grey, tracks_view

MAX_WIDTH: Final = 720
MAX_FRAMES: Final = 120
FRAME_MILLISECONDS: Final = 100
GAP: Final = 4


def animate(
    video: Video, result: PipelineResult, motions: Mapping[int, Motion], path: Path
) -> Path:
    """The raw video beside its tracks, coloured by verdict, as a looping GIF.

    Long clips keep one frame in n so the file stays small; the tails still
    cover the frames skipped, so a track is never seen jumping.
    """
    step = max(1, -(-len(video) // MAX_FRAMES))
    images = [
        PillowImage.fromarray(_side_by_side(video, result, motions, frame))
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


def _side_by_side(
    video: Video, result: PipelineResult, motions: Mapping[int, Motion], frame: int
) -> np.ndarray:
    raw = grey(video[frame])
    tracked = tracks_view(video[frame], result, motions, frame)
    gap = np.full((raw.shape[0], GAP, 3), 255, dtype=np.uint8)
    pair = np.hstack((raw, gap, tracked))
    scale = min(1.0, MAX_WIDTH / pair.shape[1])
    if scale < 1.0:
        pair = cv2.resize(pair, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(pair, cv2.COLOR_BGR2RGB)
