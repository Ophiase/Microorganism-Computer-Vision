from dataclasses import dataclass

from motility.vision.detection import Detection


@dataclass(frozen=True, slots=True)
class Observation:
    """A detection the tracker linked to a track, at one frame."""

    track: int
    frame: int
    detection: Detection
