from dataclasses import dataclass
from enum import StrEnum
from typing import Self

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from motility.vision.arrays import Frame
from motility.vision.geometry import Position, Rectangle

BYTE_MAX = 255
CONNECTIVITY = 8
ELONGATED = 10.0


class ThresholdMethod(StrEnum):
    FIXED = "fixed"
    OTSU = "otsu"


class AreaMeasure(StrEnum):
    BOX = "box"
    PIXELS = "pixels"


class DetectionParameters(BaseModel):
    """Which bright regions of the foreground map count as cells."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: ThresholdMethod = Field(
        default=ThresholdMethod.FIXED,
        description=(
            "fixed cuts at `threshold`; otsu picks the cut of each frame that best"
            " separates two intensity classes, and ignores `threshold`."
        ),
    )
    threshold: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description="Foreground level above which a pixel belongs to a cell.",
    )
    opening: int = Field(
        default=0,
        ge=0,
        le=10,
        description=(
            "Radius in pixels of a morphological opening of the mask: it erases"
            " specks and thin bridges smaller than itself. 0 turns it off."
        ),
    )
    area: AreaMeasure = Field(
        default=AreaMeasure.BOX,
        description=(
            "What min_area and max_area measure: box, the bounding box as in 2025,"
            " which grows when a rod lies diagonally and drops it; pixels, the"
            " cell itself, whatever its orientation."
        ),
    )
    min_area: int = Field(
        default=10,
        ge=1,
        description="Smallest region kept, in pixels, as `area` measures it.",
    )
    max_area: int = Field(
        default=800,
        ge=1,
        description="Largest region kept, in pixels: above, cells touch or debris.",
    )

    min_elongation: float = Field(
        default=1.0,
        ge=1.0,
        description=(
            "Smallest ratio of a region's long axis to its short one, from its"
            " second moments: rods are 1.5 and more, round specks and diffuse"
            " patches near 1. 1 keeps every shape."
        ),
    )
    static_cells: bool = Field(
        default=False,
        description=(
            "Also find the cells that never move, in the clip's median: a"
            " temporal-median background erases dead, stuck or sleeping cells"
            " from every frame. They join as stationary tracks."
        ),
    )

    @model_validator(mode="after")
    def _areas_are_ordered(self) -> Self:
        if self.min_area > self.max_area:
            raise ValueError("min_area must not exceed max_area")
        return self


@dataclass(frozen=True, slots=True)
class Detection:
    """One connected region: its centre, box, pixel count, and its long axis.

    `angle` is the long axis in radians, modulo pi: a rod has no head in one
    frame. `elongation` is the long axis over the short one, 1 for a disc.
    """

    centroid: Position
    rectangle: Rectangle
    pixels: int
    angle: float = 0.0
    elongation: float = 1.0


def mask(foreground: Frame, parameters: DetectionParameters) -> np.ndarray:
    """The binary cell mask of one foreground frame, as bytes 0 or 255."""
    image = (foreground * BYTE_MAX).astype(np.uint8)
    if parameters.method is ThresholdMethod.OTSU:
        _, binary = cv2.threshold(
            image, 0, BYTE_MAX, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
    else:
        level = int(parameters.threshold * BYTE_MAX)
        _, binary = cv2.threshold(image, level, BYTE_MAX, cv2.THRESH_BINARY)
    if parameters.opening == 0:
        return binary
    side = 2 * parameters.opening + 1
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, element)


def detect(foreground: Frame, parameters: DetectionParameters) -> tuple[Detection, ...]:
    """The cells of one foreground frame, kept by their size and their shape."""
    count, labels, statistics, centroids = cv2.connectedComponentsWithStats(
        mask(foreground, parameters), connectivity=CONNECTIVITY
    )
    detections = []
    for label in range(1, count):
        x, y, width, height, pixels = (int(value) for value in statistics[label])
        size = pixels if parameters.area is AreaMeasure.PIXELS else width * height
        if not parameters.min_area <= size <= parameters.max_area:
            continue
        window = labels[y : y + height, x : x + width] == label
        angle, elongation = axis_of(window)
        if elongation < parameters.min_elongation:
            continue
        centre = (float(centroids[label][0]), float(centroids[label][1]))
        detections.append(
            Detection(centre, Rectangle(x, y, width, height), pixels, angle, elongation)
        )
    return tuple(detections)


def axis_of(region: np.ndarray) -> tuple[float, float]:
    """The long axis of a binary region, its angle and its elongation, by moments."""
    moments = cv2.moments(region.astype(np.uint8), binaryImage=True)
    if moments["m00"] == 0:
        return 0.0, 1.0
    xx, yy, xy = moments["mu20"], moments["mu02"], moments["mu11"]
    spread = np.sqrt(((xx - yy) / 2) ** 2 + xy**2)
    long, short = (xx + yy) / 2 + spread, (xx + yy) / 2 - spread
    angle = 0.5 * float(np.arctan2(2 * xy, xx - yy))
    return angle, float(np.sqrt(long / short)) if short > 0 else ELONGATED
