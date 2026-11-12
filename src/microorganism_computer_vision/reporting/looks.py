from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np
from cv2.typing import MatLike
from motility.diffusion import Motion
from motility.pipeline import PipelineResult
from motility.tracking import TrackTable, Trajectory
from motility.vision import DetectionParameters, Frame, Video, mask
from numpy.typing import NDArray

from microorganism_computer_vision.reporting.palette import (
    DETECTION,
    MOTION_COLOURS,
)

type Image = MatLike
"""A colour picture, rows by columns by blue, green, red bytes."""

ZOOM_SCALE: Final = 3


@dataclass(frozen=True, slots=True)
class Zoom:
    """A square of the frame to enlarge in every view: its centre and its side, px."""

    x: int
    y: int
    size: int

    def apply(self, image: Image) -> Image:
        half = self.size // 2
        top, left = max(self.y - half, 0), max(self.x - half, 0)
        window = image[top : top + self.size, left : left + self.size]
        return cv2.resize(
            window, None, fx=ZOOM_SCALE, fy=ZOOM_SCALE, interpolation=cv2.INTER_NEAREST
        )


TAIL = 15
BANNER = 22
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
LEGEND_ROW = 18
SMALL_WIDTH = 640
DIMMED = 0.6


def look(
    video: Video,
    result: PipelineResult,
    detection: DetectionParameters,
    motions: Mapping[int, Motion],
    frame: int,
    zoom: Zoom | None = None,
) -> Image:
    """Four views of one frame, what an expert checks: raw, foreground, mask, tracks.

    The tracks are drawn as their last positions, coloured by their verdict,
    over the raw frame; red boxes are this frame's detections.
    """
    views = {
        f"raw, frame {frame}": _grey(video[frame]),
        "foreground": _grey(result.foreground[frame]),
        "mask": _grey(mask(result.foreground[frame], detection) / 255),
        "detections and tracks": _tracks(video[frame], result, motions, frame),
    }
    tiles = [
        _titled(zoom.apply(image) if zoom else image, title)
        for title, image in views.items()
    ]
    return np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))


def _grey(frame: Frame) -> Image:
    bytes_frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(bytes_frame, cv2.COLOR_GRAY2BGR)


def _titled(image: Image, title: str) -> Image:
    banner = np.zeros((BANNER, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, title, (6, 16), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    return np.vstack((banner, image))


def _tracks(
    frame_image: Frame,
    result: PipelineResult,
    motions: Mapping[int, Motion],
    frame: int,
) -> Image:
    image = _grey(frame_image)
    for detection in result.detections[frame]:
        box = detection.rectangle
        corner = (box.x + box.width, box.y + box.height)
        cv2.rectangle(image, (box.x, box.y), corner, DETECTION, 1)
    for track, points in _tails(result.tracks, frame).items():
        colour = MOTION_COLOURS[motions.get(track, Motion.UNDETERMINED)]
        cv2.polylines(image, [points], False, colour, 1, cv2.LINE_AA)
    return image


def _tails(tracks: TrackTable, frame: int) -> dict[int, NDArray[np.int32]]:
    """The positions of each track over the last frames, up to this one."""
    recent = (tracks.frame <= frame) & (tracks.frame > frame - TAIL)
    tails = {}
    for track in np.unique(tracks.track[recent]):
        rows = recent & (tracks.track == track)
        points = np.column_stack((tracks.x[rows], tracks.y[rows]))
        tails[int(track)] = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    return tails


def trajectory_map(
    background: Frame,
    trajectories: Sequence[Trajectory],
    motions: Mapping[int, Motion],
) -> Image:
    """Every classified track over the dimmed background, coloured by its motion."""
    grey = (np.clip(background, 0, 1) * 255 * DIMMED).astype(np.uint8)
    image = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    for trajectory in trajectories:
        motion = motions.get(trajectory.index, Motion.UNDETERMINED)
        if motion is Motion.UNDETERMINED:
            continue
        points = np.round(np.array(trajectory.positions)).astype(np.int32)
        cv2.polylines(
            image,
            [points.reshape(-1, 1, 2)],
            False,
            MOTION_COLOURS[motion],
            1,
            cv2.LINE_AA,
        )
    if image.shape[1] < SMALL_WIDTH:
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    _legend(image, motions)
    return image


def _legend(image: Image, motions: Mapping[int, Motion]) -> None:
    counts = {motion: list(motions.values()).count(motion) for motion in Motion}
    shown = [motion for motion in Motion if motion is not Motion.UNDETERMINED]
    for row, motion in enumerate(shown):
        y = LEGEND_ROW * (row + 1)
        cv2.rectangle(image, (6, y - 10), (18, y), MOTION_COLOURS[motion], -1)
        label = f"{motion.value} ({counts[motion]})"
        cv2.putText(image, label, (24, y), FONT, 0.45, WHITE, 1, cv2.LINE_AA)


def probe_look(video: Video) -> Image:
    """What a video holds before any parameter: a frame, what stays, what moves.

    The motion view is the largest departure of each pixel from its median over
    the clip, stretched to the full range: swimmers draw their paths, dust
    stays black.
    """
    background = np.median(video, axis=0)
    departure = np.max(np.abs(video - background), axis=0)
    views = {
        "first frame": _grey(video[0]),
        "median background": _grey(background),
        "motion: largest departure from the median": _grey(_stretched(departure)),
        "spread over time": _grey(_stretched(np.std(video, axis=0))),
    }
    tiles = [_titled(image, title) for title, image in views.items()]
    return np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))


def _stretched(frame: np.ndarray) -> np.ndarray:
    high = float(np.percentile(frame, 99.5))
    return np.clip(frame / high, 0, 1) if high > 0 else frame
