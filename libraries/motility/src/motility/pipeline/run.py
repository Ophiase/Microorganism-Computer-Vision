from dataclasses import dataclass

from motility.diffusion.models.motion import Motion
from motility.diffusion.models.selection import Verdict, classify
from motility.diffusion.models.turning import Turning
from motility.pipeline.parameters import PipelineParameters
from motility.tracking.kalman_filter import KalmanTracker
from motility.tracking.observation import Observation
from motility.tracking.table import TrackTable
from motility.vision.arrays import Video
from motility.vision.detection import Detection, detect
from motility.vision.preprocessing import preprocess
from motility.vision.static_cells import static_cells


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """What each stage produced: the foreground maps, the detections, the tracks."""

    foreground: Video
    detections: tuple[tuple[Detection, ...], ...]
    tracks: TrackTable
    verdicts: tuple[Verdict, ...]


def run_pipeline(video: Video, parameters: PipelineParameters) -> PipelineResult:
    """Preprocess, detect, track and classify a grayscale video in [0, 1]."""
    foreground = preprocess(video, parameters.preprocessing)
    detections = tuple(detect(frame, parameters.detection) for frame in foreground)
    tracker = KalmanTracker(parameters.tracking)
    observations = [
        observation
        for frame, frame_detections in enumerate(detections)
        for observation in tracker.step(frame_detections, frame)
    ]
    still = _still_observations(video, parameters, tracker.next_index)
    tracks = TrackTable.from_observations([*observations, *still])
    moving = TrackTable.from_observations(observations)
    verdicts = (
        *(classify(item, parameters.classification) for item in moving.trajectories()),
        *(
            _stationary(index, len(video))
            for index in sorted({item.track for item in still})
        ),
    )
    return PipelineResult(foreground, detections, tracks, verdicts)


def _still_observations(
    video: Video, parameters: PipelineParameters, first_index: int
) -> list[Observation]:
    """Each cell of the median, observed at its place in every frame."""
    if not parameters.detection.static_cells:
        return []
    cells = static_cells(video, parameters.preprocessing, parameters.detection)
    return [
        Observation(first_index + index, frame, cell)
        for index, cell in enumerate(cells)
        for frame in range(len(video))
    ]


def _stationary(track: int, frames: int) -> Verdict:
    """A cell of the median does not move by construction: certain, at no speed."""
    return Verdict(
        track, frames, Motion.STATIONARY, 1.0, {}, None, None, 0.0, Turning(0.0, 0.0)
    )
