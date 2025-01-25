from typing import List
import cv2
import numpy as np
from logic.structure.bounding_box import BoundingBox
from logic.structure.rectangle import Rectangle


def extract_high_optical_flow_areas(frame: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    optical_flow_magnitude = np.sqrt(
        frame[..., 1, 0] ** 2 + frame[..., 1, 1] ** 2)
    high_flow_mask = optical_flow_magnitude > threshold
    return high_flow_mask


def detect_shapes(frame: np.ndarray,
                  min_size: int = 10,
                  max_size: int = 1000) -> List[BoundingBox]:
    """
    Detect bounding boxes of white regions in a grayscale image.

    Args:
        frame: 2D numpy array of shape (height, width) with values in [0, 1]
        min_size: Minimum area of bounding boxes to include
        max_size: Maximum area of bounding boxes to include

    Returns:
        List of bounding boxes as (x, y, width, height) tuples
    """
    # Convert to 8-bit and apply threshold
    frame_uint8 = (frame * 255).astype(np.uint8)
    _, thresh = cv2.threshold(frame_uint8, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract and filter bounding boxes
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if min_size <= area <= max_size:
            bounding_boxes.append(BoundingBox(Rectangle(x, y, w, h)))

    return bounding_boxes
