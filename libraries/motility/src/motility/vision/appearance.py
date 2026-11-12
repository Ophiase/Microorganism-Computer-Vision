import cv2
import numpy as np
from pydantic import BaseModel, Field

from motility.vision.arrays import Video
from motility.vision.preprocessing import Polarity

MAD_TO_SIGMA = 1.4826
TAIL = 0.05
OBJECT_SIGMAS = 5.0
CONNECTIVITY = 8
FRAMES_SAMPLED = 12
MIN_PIXELS = 3


class AppearanceReport(BaseModel):
    """What a video looks like, measured without any parameter from the user."""

    frames: int
    width: int
    height: int
    background: float = Field(description="Median intensity of the static background.")
    gradient: float = Field(
        description="Difference of the background between its left and right tenths."
    )
    noise: float = Field(
        description="Deviation of the pixel noise around the background."
    )
    polarity: Polarity = Field(
        description="Whether what moves is darker or brighter than the background."
    )
    contrast: float = Field(
        description="Typical depth of the moving objects below or above the background."
    )
    objects_per_frame: float = Field(
        description="Moving objects found per frame, above five noise deviations."
    )
    object_length: float = Field(description="Median long side of their boxes, px.")
    object_width: float = Field(description="Median short side of their boxes, px.")
    static_share: float = Field(
        description=(
            "Share of the image's structure that never moves: dust and stuck cells."
            " High, a temporal-median background is worth subtracting."
        )
    )


def measure_appearance(video: Video) -> AppearanceReport:
    """Polarity, noise, contrast and size of the moving objects of a video."""
    background = np.median(video, axis=0)
    residual = video - background
    noise = float(MAD_TO_SIGMA * np.median(np.abs(residual)))
    low, high = np.quantile(residual, (TAIL / 100, 1 - TAIL / 100))
    polarity = Polarity.DARK if abs(low) >= abs(high) else Polarity.BRIGHT
    signed = -residual if polarity is Polarity.DARK else residual
    boxes = _object_boxes(signed, OBJECT_SIGMAS * max(noise, 1e-4))
    tenth = max(video.shape[2] // 10, 1)
    return AppearanceReport(
        frames=video.shape[0],
        width=video.shape[2],
        height=video.shape[1],
        background=float(np.median(background)),
        gradient=float(
            np.median(background[:, :tenth]) - np.median(background[:, -tenth:])
        ),
        noise=noise,
        polarity=polarity,
        contrast=float(max(abs(low), abs(high))),
        objects_per_frame=len(boxes) / min(FRAMES_SAMPLED, video.shape[0]),
        object_length=float(np.median(boxes.max(axis=1))) if len(boxes) else 0.0,
        object_width=float(np.median(boxes.min(axis=1))) if len(boxes) else 0.0,
        static_share=_static_share(background, residual),
    )


def _object_boxes(signed: Video, level: float) -> np.ndarray:
    """Width and height of the regions above `level` in a sample of frames."""
    sampled = np.linspace(0, len(signed) - 1, min(FRAMES_SAMPLED, len(signed))).astype(
        int
    )
    sizes = []
    for frame in signed[sampled]:
        binary = (cv2.GaussianBlur(frame, (3, 3), 0) > level).astype(np.uint8)
        count, _, statistics, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=CONNECTIVITY
        )
        sizes.extend(
            (int(statistics[label][2]), int(statistics[label][3]))
            for label in range(1, count)
            if statistics[label][4] >= MIN_PIXELS
        )
    return np.array(sizes).reshape(-1, 2)


def _static_share(background: np.ndarray, residual: Video) -> float:
    """Variance of the detailed background against that of what moves."""
    detail = background - cv2.GaussianBlur(background, (0, 0), 10)
    static = float(np.var(detail))
    moving = float(np.mean(np.var(residual, axis=0)))
    return static / (static + moving) if static + moving > 0 else 0.0
