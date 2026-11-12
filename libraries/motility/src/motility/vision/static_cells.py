from typing import Final

import cv2
import numpy as np

from motility.vision.arrays import Video
from motility.vision.detection import Detection, DetectionParameters, detect
from motility.vision.preprocessing import (
    MID_GREY,
    PreprocessingParameters,
    foreground_of,
    static_objects,
)

FLATTENING: Final = 10.0


def static_cells(
    video: Video,
    preprocessing: PreprocessingParameters,
    detection: DetectionParameters,
) -> tuple[Detection, ...]:
    """The cells present at the same place in most frames, found in the median.

    The median is freed of its own slow background, then read with the same
    window, polarity and areas as the moving cells; the large static objects
    stay excluded, so beads and host cells do not count as cells.
    """
    median = np.median(video, axis=0)
    flattened = median - cv2.GaussianBlur(median, (0, 0), FLATTENING) + MID_GREY
    foreground = foreground_of(flattened[np.newaxis].astype(np.float32), preprocessing)[
        0
    ]
    if preprocessing.exclude_static is not None:
        foreground = foreground * ~static_objects(
            video, preprocessing.exclude_static, preprocessing.exclusion_margin
        )
    return detect(foreground, detection)
