from enum import StrEnum
from typing import Self

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from motility.vision.arrays import Video

MID_GREY = 0.5


class Background(StrEnum):
    NONE = "none"
    TEMPORAL_MEDIAN = "temporal-median"


class Polarity(StrEnum):
    DARK = "dark"
    BRIGHT = "bright"


class PreprocessingParameters(BaseModel):
    """How a raw frame becomes a foreground map, where a cell is bright on black."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    background: Background = Field(
        default=Background.NONE,
        description=(
            "temporal-median subtracts the per-pixel median of the clip, then adds"
            " mid grey back: what never moves (dust, stuck cells, uneven light)"
            " vanishes, so the window below reads a contrast around 0.5."
            " none keeps the raw intensities."
        ),
    )
    polarity: Polarity = Field(
        default=Polarity.DARK,
        description=(
            "dark when the cells are darker than the background (phase contrast,"
            " bright field), bright when they shine (fluorescence, dark field)."
        ),
    )
    window_low: float = Field(
        default=0.388,
        ge=0.0,
        le=1.0,
        description=(
            "Intensity mapped to 0 before the polarity; darker is clipped. 0.388 is"
            " what the 2025 code's renormalisation amounted to on its reference video."
        ),
    )
    window_high: float = Field(
        default=0.49,
        ge=0.0,
        le=1.0,
        description="Intensity mapped to 1 before the polarity; brighter is clipped.",
    )
    gain: float = Field(
        default=4.0,
        gt=0.0,
        description=(
            "Multiplies the foreground after the polarity, then clips to 1:"
            " a high gain saturates faint cells into the detection threshold."
        ),
    )
    smoothing: int = Field(
        default=3,
        ge=1,
        le=31,
        description=(
            "Side in pixels of the box filter applied last, the border averaged"
            " over the pixels inside the frame; 1 turns it off. Odd values only."
        ),
    )

    exclude_static: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Zero the foreground wherever the clip's median departs from its own"
            " median level by more than this: large static objects (beads, host"
            " cells, debris, a scale bar) whose edges jitter into false cells."
            " Unset keeps everything."
        ),
    )
    exclusion_margin: int = Field(
        default=3,
        ge=0,
        le=30,
        description="Pixels added around each excluded object, to cover its halo.",
    )

    @model_validator(mode="after")
    def _window_is_ordered(self) -> Self:
        if self.window_low >= self.window_high:
            raise ValueError("window_low must be below window_high")
        if self.smoothing % 2 == 0:
            raise ValueError("smoothing must be odd")
        return self


def preprocess(video: Video, parameters: PreprocessingParameters) -> Video:
    """The foreground map of every frame: cells bright, background black, in [0, 1]."""
    contrast = _subtract_background(video, parameters.background)
    span = parameters.window_high - parameters.window_low
    windowed = (np.clip(contrast, parameters.window_low, parameters.window_high)) - (
        parameters.window_low
    )
    foreground = windowed / span
    if parameters.polarity is Polarity.DARK:
        foreground = 1.0 - foreground
    foreground = np.clip(foreground * parameters.gain, 0.0, 1.0).astype(np.float32)
    smoothed = np.stack([smooth(frame, parameters.smoothing) for frame in foreground])
    if parameters.exclude_static is None:
        return smoothed
    return smoothed * ~static_objects(
        video, parameters.exclude_static, parameters.exclusion_margin
    )


def _subtract_background(video: Video, background: Background) -> Video:
    if background is Background.NONE:
        return video
    return (video - np.median(video, axis=0) + MID_GREY).astype(np.float32)


def static_objects(video: Video, level: float, margin: int) -> np.ndarray:
    """Where the median frame departs from its median level, grown by `margin` px."""
    median = np.median(video, axis=0)
    objects = (np.abs(median - np.median(median)) > level).astype(np.uint8)
    if margin > 0:
        side = 2 * margin + 1
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
        objects = cv2.dilate(objects, element)
    return objects.astype(bool)


def smooth(frame: np.ndarray, side: int) -> np.ndarray:
    """A box average whose border divides by the pixels it actually covers."""
    if side == 1:
        return frame
    kernel = (side, side)
    total = cv2.boxFilter(
        frame, -1, kernel, normalize=False, borderType=cv2.BORDER_CONSTANT
    )
    covered = cv2.boxFilter(
        np.ones_like(frame), -1, kernel, normalize=False, borderType=cv2.BORDER_CONSTANT
    )
    return total / covered
