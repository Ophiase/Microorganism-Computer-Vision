from collections.abc import Mapping, Sequence
from typing import Final

import cv2
import numpy as np
from motility.diffusion import Motion
from motility.tracking import Trajectory
from motility.vision import Frame

from microorganism_computer_vision.reporting.pictures.looks import FONT, WHITE, Image
from microorganism_computer_vision.reporting.pictures.palette import MOTION_COLOURS

LEGEND_ROW: Final = 18
SMALL_WIDTH: Final = 640
DIMMED: Final = 0.6


def trajectory_map(
    background: Frame,
    trajectories: Sequence[Trajectory],
    motions: Mapping[int, Motion],
) -> Image:
    """Every classified track over the dimmed background, coloured by its motion."""
    grey = (np.clip(background, 0, 1) * 255 * DIMMED).astype(np.uint8)
    image = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    for trajectory in trajectories:
        motion = motions.get(trajectory.index, Motion.UNDETERMINED)
        if motion is Motion.UNDETERMINED:
            continue
        points = np.round(np.array(trajectory.positions)).astype(np.int32)
        cv2.polylines(
            image,
            [points.reshape(-1, 1, 2)],
            False,
            MOTION_COLOURS[motion],
            1,
            cv2.LINE_AA,
        )
    if image.shape[1] < SMALL_WIDTH:
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    _legend(image, motions)
    return image


def _legend(image: Image, motions: Mapping[int, Motion]) -> None:
    counts = {motion: list(motions.values()).count(motion) for motion in Motion}
    shown = [motion for motion in Motion if motion is not Motion.UNDETERMINED]
    for row, motion in enumerate(shown):
        y = LEGEND_ROW * (row + 1)
        cv2.rectangle(image, (6, y - 10), (18, y), MOTION_COLOURS[motion], -1)
        label = f"{motion.value} ({counts[motion]})"
        cv2.putText(image, label, (24, y), FONT, 0.45, WHITE, 1, cv2.LINE_AA)
