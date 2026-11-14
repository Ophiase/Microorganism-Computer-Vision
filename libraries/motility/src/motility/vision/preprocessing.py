from enum import StrEnum
from typing import Self

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from motility.vision.arrays import Video
from motility.vision.regions.mask import MaskParameters, ignored_region

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

    even_exposure: bool = Field(
        default=False,
        description=(
            "Scale every frame to the clip's median brightness first: a camera's"
            " automatic exposure jumps between frames, and the background"
            " subtraction would turn each jump into thousands of false cells."
        ),
    )
    mask: MaskParameters = Field(
        default_factory=MaskParameters,
        description="What the pipeline ignores; see the mask's own fields.",
    )

    @model_validator(mode="after")
    def _window_is_ordered(self) -> Self:
        if self.window_low >= self.window_high:
            raise ValueError("window_low must be below window_high")
        if self.smoothing % 2 == 0:
            raise ValueError("smoothing must be odd")
        return self


def preprocess(
    video: Video,
    parameters: PreprocessingParameters,
    ignored: np.ndarray | None = None,
) -> Video:
    """The foreground map of every frame: cells bright, background black, in [0, 1].

    `ignored` is the region the mask hides, when the caller computed it already.
    """
    if parameters.even_exposure:
        video = even_exposure(video)
    smoothed = foreground_of(
        _subtract_background(video, parameters.background), parameters
    )
    if not parameters.mask.active:
        return smoothed
    if ignored is None:
        ignored = ignored_region(video, parameters.mask)
    return smoothed * ~ignored


def foreground_of(contrast: Video, parameters: PreprocessingParameters) -> Video:
    """Window, polarity, gain and smoothing of intensities freed of their background."""
    span = parameters.window_high - parameters.window_low
    windowed = (np.clip(contrast, parameters.window_low, parameters.window_high)) - (
        parameters.window_low
    )
    foreground = windowed / span
    if parameters.polarity is Polarity.DARK:
        foreground = 1.0 - foreground
    foreground = np.clip(foreground * parameters.gain, 0.0, 1.0).astype(np.float32)
    return np.stack([smooth(frame, parameters.smoothing) for frame in foreground])


def even_exposure(video: Video) -> Video:
    """Every frame scaled so its median brightness is the clip's."""
    medians = np.median(video.reshape(len(video), -1), axis=1)
    scale = float(np.median(medians)) / np.maximum(medians, 1e-6)
    return np.clip(video * scale[:, None, None], 0.0, 1.0).astype(np.float32)


def _subtract_background(video: Video, background: Background) -> Video:
    if background is Background.NONE:
        return video
    return (video - np.median(video, axis=0) + MID_GREY).astype(np.float32)


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
