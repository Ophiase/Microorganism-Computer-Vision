from collections.abc import Mapping, Sequence
from typing import Final

import cv2
import numpy as np
from motility.diffusion import Motion
from motility.pipeline import PipelineResult
from motility.review import Suspect
from motility.vision import Video

from microorganism_computer_vision.reporting.pictures.overlay import TrackOverlay

TILE: Final = 200
STRIP: Final = 8
AROUND: Final = 4
MARGIN: Final = 24
SMALLEST: Final = 48
LARGEST: Final = 200
SHEET_ROWS: Final = 12
CAPTION: Final = 16
FONT: Final = cv2.FONT_HERSHEY_SIMPLEX


class Inspector:
    """Close views of chosen tracks: a camera that follows them, enlarged.

    Every track in view keeps its box, tail, colour and number, so a swap, a
    merge or a false cell can be judged by eye before it is corrected.
    """

    def __init__(
        self, video: Video, result: PipelineResult, motions: Mapping[int, Motion]
    ) -> None:
        self._video = video
        self._tracks = result.tracks
        self._overlay = TrackOverlay(result, motions, labels=True)

    def film(self, tracks: Sequence[int], frames: int = STRIP) -> np.ndarray:
        """Evenly spaced frames over the tracks' lives, a few before and after."""
        rows = np.isin(self._tracks.track, list(tracks))
        if not rows.any():
            raise ValueError(f"no track {list(tracks)} in this run")
        first = max(int(self._tracks.frame[rows].min()) - AROUND, 0)
        last = min(int(self._tracks.frame[rows].max()) + AROUND, len(self._video) - 1)
        chosen = np.unique(np.linspace(first, last, frames).round().astype(int))
        return np.hstack([self._tile(tracks, int(frame)) for frame in chosen])

    def sheet(self, suspects: Sequence[Suspect]) -> np.ndarray:
        """One row per suspect: four frames around where it is suspected."""
        rows = [self._suspect_row(suspect) for suspect in suspects[:SHEET_ROWS]]
        return np.vstack(rows)

    def _suspect_row(self, suspect: Suspect) -> np.ndarray:
        tracks = [suspect.track] + (
            [suspect.other] if suspect.other is not None else []
        )
        frame = (
            suspect.frame if suspect.frame is not None else self._middle(suspect.track)
        )
        frames = [
            min(max(frame + offset, 0), len(self._video) - 1)
            for offset in (-3, -1, 0, 2)
        ]
        tiles = np.hstack([self._tile(tracks, item) for item in frames])
        caption = np.zeros((CAPTION, tiles.shape[1], 3), dtype=np.uint8)
        text = f"{suspect.kind.value} {' + '.join(map(str, tracks))}: {suspect.note}"
        cv2.putText(caption, text, (4, 12), FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack((caption, tiles))

    def _tile(self, tracks: Sequence[int], frame: int) -> np.ndarray:
        x, y = self._centre(tracks, frame)
        size = self._size(tracks)
        image = self._overlay.draw(self._video[frame], frame)
        half = size // 2
        padded = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_CONSTANT)
        window = padded[int(y) : int(y) + size, int(x) : int(x) + size]
        tile = cv2.resize(window, (TILE, TILE), interpolation=cv2.INTER_NEAREST)
        cv2.putText(
            tile, f"frame {frame}", (4, TILE - 6), FONT, 0.4, (255, 255, 255), 1
        )
        return tile

    def _centre(self, tracks: Sequence[int], frame: int) -> tuple[float, float]:
        """Where the first track is at `frame`, the nearest observation standing in."""
        rows = np.flatnonzero(self._tracks.track == tracks[0])
        nearest = rows[np.argmin(np.abs(self._tracks.frame[rows] - frame))]
        return float(self._tracks.x[nearest]), float(self._tracks.y[nearest])

    def _size(self, tracks: Sequence[int]) -> int:
        """A window that holds the tracks' whole extent, within bounds."""
        rows = np.isin(self._tracks.track, list(tracks))
        extent = max(np.ptp(self._tracks.x[rows]), np.ptp(self._tracks.y[rows]))
        return int(min(max(extent / 2 + MARGIN, SMALLEST), LARGEST))

    def _middle(self, track: int) -> int:
        frames = self._tracks.frame[self._tracks.track == track]
        return int(np.median(frames))
