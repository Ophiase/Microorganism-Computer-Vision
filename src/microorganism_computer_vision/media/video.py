from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from motility.vision import Video, as_video
from pydantic import BaseModel, ConfigDict, Field

IMAGE_SUFFIXES = frozenset({".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"})
STACK_SUFFIXES = frozenset({".tif", ".tiff"})


class UsageError(ValueError):
    """An input `mcv` refuses: said in one line, never a traceback."""


class Clip(BaseModel):
    """Which frames of the source are read: the half-open range [start, stop)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: int = Field(default=0, ge=0, description="First frame read.")
    stop: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Frame after the last one read; unset reads to the end. A short clip"
            " makes a fast loop: calibrate on 50 to 100 frames, then run it all."
        ),
    )


@dataclass(frozen=True, slots=True)
class Recording:
    """The grayscale frames of a source, and its frame rate when it says one."""

    video: Video
    frame_rate: float | None


def read_recording(source: Path, clip: Clip) -> Recording:
    """A video file, a multi-page TIFF, or a folder of images, as grayscale floats."""
    if source.is_dir():
        frames = [_read_image(path) for path in _clip(_images(source), clip)]
        return Recording(as_video(np.stack(frames)), None)
    if source.suffix.lower() in STACK_SUFFIXES:
        read, pages = cv2.imreadmulti(str(source), flags=cv2.IMREAD_ANYDEPTH)
        if not read:
            raise UsageError(f"{source} is not a readable TIFF stack")
        return Recording(as_video(np.stack(_clip(list(pages), clip))), None)
    return _read_video(source, clip)


def _images(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )


def _clip[Item](items: list[Item], clip: Clip) -> list[Item]:
    selected = items[clip.start : clip.stop]
    if not selected:
        raise UsageError("the clip selects no frame")
    return selected


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise UsageError(f"{path} is not a readable image")
    return image


def _read_video(source: Path, clip: Clip) -> Recording:
    """Decode frame by frame: codecs give no cheap seek, so early frames are read."""
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise UsageError(f"{source} is not a readable video")
    rate = capture.get(cv2.CAP_PROP_FPS) or None
    frames = []
    index = 0
    while clip.stop is None or index < clip.stop:
        read, frame = capture.read()
        if not read:
            break
        if index >= clip.start:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        index += 1
    capture.release()
    if not frames:
        raise UsageError("the clip selects no frame")
    return Recording(as_video(np.stack(frames)), rate)
