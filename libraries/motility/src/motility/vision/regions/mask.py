from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from motility.vision.arrays import Video

FROZEN: Final = ConfigDict(extra="forbid", frozen=True)


class Disc(BaseModel):
    """A disc of the frame, in pixels: where to mask, or where to give back."""

    model_config = FROZEN

    x: float = Field(description="Centre, column, px.")
    y: float = Field(description="Centre, row, px.")
    radius: float = Field(gt=0.0, description="Radius, px.")


class MaskParameters(BaseModel):
    """What the pipeline ignores: large static objects, and discs an agent draws.

    The ignored region holds no detection, no still cell, no miss; the looks
    show it veiled, so a reader sees what was not looked at.
    """

    model_config = FROZEN

    static: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Ignore where the clip's median departs from its own median level by"
            " more than this: beads, host cells, debris, burnt-in text. Unset"
            " ignores nothing automatically."
        ),
    )
    margin: int = Field(
        default=3, ge=0, le=30, description="Pixels added around each static object."
    )
    ignore: tuple[Disc, ...] = Field(
        default=(), description="Discs masked by hand: an object the level missed."
    )
    keep: tuple[Disc, ...] = Field(
        default=(), description="Discs given back: cells the level masked wrongly."
    )

    @property
    def active(self) -> bool:
        return self.static is not None or bool(self.ignore)


def ignored_region(video: Video, mask: MaskParameters) -> NDArray[np.bool_]:
    """True where the pipeline must not look: static objects, then the discs."""
    shape = video.shape[1:]
    region = np.zeros(shape, dtype=np.uint8)
    if mask.static is not None:
        region = static_objects(video, mask.static, mask.margin).astype(np.uint8)
    for disc in mask.ignore:
        cv2.circle(region, (round(disc.x), round(disc.y)), round(disc.radius), 1, -1)
    for disc in mask.keep:
        cv2.circle(region, (round(disc.x), round(disc.y)), round(disc.radius), 0, -1)
    return region.astype(bool)


def static_objects(video: Video, level: float, margin: int) -> NDArray[np.bool_]:
    """Where the median frame departs from its median level, grown by `margin` px."""
    median = np.median(video, axis=0)
    objects = (np.abs(median - np.median(median)) > level).astype(np.uint8)
    if margin > 0:
        side = 2 * margin + 1
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
        objects = cv2.dilate(objects, element)
    return objects.astype(bool)
