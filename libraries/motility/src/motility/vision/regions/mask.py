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


class Polygon(BaseModel):
    """A region of the frame outlined by its corners, in pixels, in order."""

    model_config = FROZEN

    points: tuple[tuple[float, float], ...] = Field(
        min_length=3, description="Corners (x, y), px, in order around the region."
    )


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
    inside: tuple[Polygon, ...] = Field(
        default=(),
        description=(
            "Where to look, when only part of the frame matters: the inside of a"
            " channel or a chamber. Everything outside these polygons is ignored."
        ),
    )
    ignore_regions: tuple[Polygon, ...] = Field(
        default=(),
        description=(
            "Polygons masked by hand: a wall, a channel's outside, an uneven"
            " light the level would take for objects."
        ),
    )

    @property
    def active(self) -> bool:
        return any((
            self.static is not None,
            self.ignore,
            self.ignore_regions,
            self.inside,
        ))


def ignored_region(video: Video, mask: MaskParameters) -> NDArray[np.bool_]:
    """True where the pipeline must not look: static objects, then the discs."""
    shape = video.shape[1:]
    region = np.zeros(shape, dtype=np.uint8)
    if mask.static is not None:
        region = static_objects(video, mask.static, mask.margin).astype(np.uint8)
    if mask.inside:
        looked = np.zeros(shape, dtype=np.uint8)
        for polygon in mask.inside:
            corners = np.round(np.array(polygon.points)).astype(np.int32)
            cv2.fillPoly(looked, [corners], 1)
        region = np.maximum(region, 1 - looked)
    for polygon in mask.ignore_regions:
        corners = np.round(np.array(polygon.points)).astype(np.int32)
        cv2.fillPoly(region, [corners], 1)
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
