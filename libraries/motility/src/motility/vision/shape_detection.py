import cv2
import numpy as np

from motility.tracking.bounding_box import BoundingBox
from motility.tracking.rectangle import Rectangle


def detect_shapes(
    frame: np.ndarray, min_area: int = 10, max_area: int = 1000
) -> tuple[BoundingBox, ...]:
    """The bounding boxes of the white regions of a grayscale frame in [0, 1]."""
    frame_uint8 = (frame * 255).astype(np.uint8)
    _, threshold = cv2.threshold(frame_uint8, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if min_area <= width * height <= max_area:
            boxes.append(BoundingBox(Rectangle(x, y, width, height)))
    return tuple(boxes)
