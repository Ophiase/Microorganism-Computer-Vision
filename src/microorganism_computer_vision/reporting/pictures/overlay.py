import math
from collections.abc import Mapping
from typing import Final

import cv2
import numpy as np
from motility.diffusion import Motion
from motility.pipeline import PipelineResult
from motility.vision import Frame

from microorganism_computer_vision.reporting.pictures.palette import (
    LOST,
    OVERLAY_COLOURS,
    Colour,
)

TAIL: Final = 15
TAIL_OPACITY: Final = 0.45
VEIL: Final = 0.45
LOST_FRAMES: Final = 4
CORNER: Final = 3


class TrackOverlay:
    """Draws the cells of a frame over it: a box and a tail per track, by verdict.

    Drawing happens at the output scale, after the frame is resized, so a one
    pixel line stays one sharp pixel however small the picture. The tails are
    translucent, so the cell under them stays visible; the boxes are thin and
    opaque, a little larger than the cell. A track missed in this frame but
    seen in the last few keeps red corners where it was last seen, so a box
    does not blink: it turns red, then goes.
    """

    def __init__(
        self,
        result: PipelineResult,
        motions: Mapping[int, Motion],
        scale: float = 1.0,
        thickness: int = 1,
        antialiased: bool = True,
        labels: bool = False,
    ) -> None:
        self._tracks = result.tracks
        self._ignored = result.ignored
        self._motions = motions
        self._scale = scale
        self._thickness = thickness
        self._line = cv2.LINE_AA if antialiased else cv2.LINE_8
        self._labels = labels

    def draw(self, frame_image: Frame, frame: int) -> np.ndarray:
        image = self._background(frame_image)
        here = np.flatnonzero(self._tracks.frame == frame).tolist()
        tails = image.copy()
        for track in set(self._tracks.track[here].tolist()):
            self._tail(tails, track, frame)
        image = cv2.addWeighted(tails, TAIL_OPACITY, image, 1 - TAIL_OPACITY, 0)
        for row in here:
            self._box(image, row)
        for row in self._lost(frame, here):
            self._corners(image, row)
        return image

    def _lost(self, frame: int, here: list[int]) -> list[int]:
        """The last observation of each track seen recently but not now."""
        tracks = self._tracks
        seen_now = set(tracks.track[here].tolist())
        recent = np.flatnonzero(
            (tracks.frame < frame) & (tracks.frame >= frame - LOST_FRAMES)
        )
        last: dict[int, int] = {}
        for row in recent.tolist():
            track = int(tracks.track[row])
            if track not in seen_now and (
                track not in last or tracks.frame[row] > tracks.frame[last[track]]
            ):
                last[track] = row
        return list(last.values())

    def _corners(self, image: np.ndarray, row: int) -> None:
        (left, top), (right, bottom) = (
            tuple(point) for point in self._pixels(self._extent(row))
        )
        for x, y, dx, dy in (
            (left, top, 1, 1),
            (right, top, -1, 1),
            (left, bottom, 1, -1),
            (right, bottom, -1, -1),
        ):
            cv2.line(image, (x, y), (x + dx * CORNER, y), LOST, 1, self._line)
            cv2.line(image, (x, y), (x, y + dy * CORNER), LOST, 1, self._line)

    def _extent(self, row: int) -> np.ndarray:
        tracks = self._tracks
        half_width = max(tracks.width[row], 4) / 2 + 2
        half_height = max(tracks.height[row], 4) / 2 + 2
        return np.array([
            (tracks.x[row] - half_width, tracks.y[row] - half_height),
            (tracks.x[row] + half_width, tracks.y[row] + half_height),
        ])

    def _background(self, frame_image: Frame) -> np.ndarray:
        """The frame in grey, the masked region darkened: seen, not looked at."""
        grey = (np.clip(frame_image, 0, 1) * 255).astype(np.uint8)
        if self._ignored is not None:
            grey = np.where(self._ignored, (grey * VEIL).astype(np.uint8), grey)
        if not math.isclose(self._scale, 1.0):
            grey = cv2.resize(
                grey, None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_AREA
            )
        return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

    def _colour(self, track: int) -> Colour:
        return OVERLAY_COLOURS[self._motions.get(track, Motion.UNDETERMINED)]

    def _box(self, image: np.ndarray, row: int) -> None:
        tracks = self._tracks
        first, second = (tuple(point) for point in self._pixels(self._extent(row)))
        colour = self._colour(int(tracks.track[row]))
        cv2.rectangle(image, first, second, colour, 1, self._line)
        if self._labels:
            corner = (second[0] + 2, first[1] + 8)
            text = str(tracks.track[row])
            cv2.putText(image, text, corner, cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)

    def _tail(self, image: np.ndarray, track: int, frame: int) -> None:
        tracks = self._tracks
        rows = (
            (tracks.track == track)
            & (tracks.frame <= frame)
            & (tracks.frame > frame - TAIL)
        )
        if rows.sum() < 2:
            return
        points = self._pixels(np.column_stack((tracks.x[rows], tracks.y[rows])))
        cv2.polylines(
            image,
            [points.reshape(-1, 1, 2)],
            False,
            self._colour(track),
            self._thickness,
            self._line,
        )

    def _pixels(self, points: np.ndarray) -> np.ndarray:
        return np.round(points * self._scale).astype(np.int32)
