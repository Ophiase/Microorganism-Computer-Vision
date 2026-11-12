from dataclasses import dataclass

from motility.diffusion.models.selection import Verdict, classify
from motility.pipeline.parameters import PipelineParameters
from motility.tracking.kalman_filter import KalmanTracker
from motility.tracking.table import TrackTable
from motility.vision.arrays import Video
from motility.vision.detection import Detection, detect
from motility.vision.preprocessing import preprocess


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
    tracks = TrackTable.from_observations(observations)
    verdicts = tuple(
        classify(trajectory, parameters.classification)
        for trajectory in tracks.trajectories()
    )
    return PipelineResult(foreground, detections, tracks, verdicts)
