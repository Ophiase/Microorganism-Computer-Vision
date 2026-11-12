from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from motility.tracking.observation import Observation
from motility.vision.detection import Detection

TRANSITION = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
OBSERVATION = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
POSITION = slice(0, 2)
VELOCITY = slice(2, 4)
INITIAL_COVARIANCE = 0.1
UNREACHABLE_COST = 1e9


class TrackingParameters(BaseModel):
    """How detections are linked into tracks: a constant-velocity Kalman filter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_assignment_distance: float = Field(
        default=30.0,
        gt=0.0,
        description=(
            "Gate in pixels between a track's predicted position and a detection:"
            " farther, they are never linked. About the largest step a cell makes"
            " between two frames, plus its own size."
        ),
    )
    max_missed_frames: int = Field(
        default=15,
        ge=0,
        description=(
            "Frames a track survives without a detection, coasting on its velocity:"
            " high bridges blinks and crossings, but also links two different cells."
        ),
    )
    max_velocity: float = Field(
        default=50.0,
        gt=0.0,
        description="Speed in pixels per frame the filter's velocity is clipped to.",
    )
    process_noise: float = Field(
        default=1e-5,
        gt=0.0,
        description=(
            "Variance the motion model gains per frame: high lets a track turn"
            " quickly, low trusts its velocity."
        ),
    )
    measurement_noise: float = Field(
        default=1e-5,
        gt=0.0,
        description="Variance of a detection's position, in pixels squared.",
    )


@dataclass(slots=True)
class Track:
    """A Kalman state `[x, y, vx, vy]`, its covariance, and the frames it missed."""

    index: int
    state: np.ndarray
    covariance: np.ndarray
    missed: int = 0


class KalmanTracker:
    """Links the detections of successive frames, one call per frame."""

    def __init__(self, parameters: TrackingParameters) -> None:
        self.parameters = parameters
        self.tracks: list[Track] = []
        self.next_index = 0
        self.process_covariance = np.eye(4) * parameters.process_noise
        self.measurement_covariance = np.eye(2) * parameters.measurement_noise

    def step(
        self, detections: Sequence[Detection], frame: int
    ) -> tuple[Observation, ...]:
        """Predict every track, match it to this frame's detections, then age the rest.

        Only the detections are reported, each with the track it joined: a track
        that coasts through a frame reports nothing for it.
        """
        for track in self.tracks:
            self._predict(track)

        observations = []
        unmatched = set(range(len(detections)))
        matched_tracks = set()
        for track_position, detection_position in self._match(detections):
            track = self.tracks[track_position]
            self._update(track, detections[detection_position].centroid)
            track.missed = 0
            matched_tracks.add(track_position)
            unmatched.discard(detection_position)
            observations.append(
                Observation(track.index, frame, detections[detection_position])
            )

        for position, track in enumerate(self.tracks):
            if position not in matched_tracks:
                track.missed += 1
        self.tracks = [
            track
            for track in self.tracks
            if track.missed <= self.parameters.max_missed_frames
        ]

        for detection_position in sorted(unmatched):
            track = self._start(detections[detection_position])
            observations.append(
                Observation(track.index, frame, detections[detection_position])
            )
        return tuple(observations)

    def _start(self, detection: Detection) -> Track:
        x, y = detection.centroid
        track = Track(
            index=self.next_index,
            state=np.array([x, y, 0.0, 0.0]),
            covariance=np.eye(4) * INITIAL_COVARIANCE,
        )
        self.next_index += 1
        self.tracks.append(track)
        return track

    def _predict(self, track: Track) -> None:
        track.state = TRANSITION @ track.state
        self._clip_velocity(track)
        track.covariance = (
            TRANSITION @ track.covariance @ TRANSITION.T + self.process_covariance
        )

    def _update(self, track: Track, measurement: tuple[float, float]) -> None:
        innovation = np.array(measurement) - OBSERVATION @ track.state
        innovation_covariance = (
            OBSERVATION @ track.covariance @ OBSERVATION.T + self.measurement_covariance
        )
        gain = track.covariance @ OBSERVATION.T @ np.linalg.inv(innovation_covariance)
        track.state = track.state + gain @ innovation
        self._clip_velocity(track)
        track.covariance = (np.eye(4) - gain @ OBSERVATION) @ track.covariance

    def _clip_velocity(self, track: Track) -> None:
        limit = self.parameters.max_velocity
        track.state[VELOCITY] = np.clip(track.state[VELOCITY], -limit, limit)

    def _match(self, detections: Sequence[Detection]) -> list[tuple[int, int]]:
        """Pairs of (track position, detection position) within the gate."""
        if not self.tracks or not detections:
            return []
        predicted = np.array([track.state[POSITION] for track in self.tracks])
        measured = np.array([detection.centroid for detection in detections])
        distance = cdist(predicted, measured)
        gate = self.parameters.max_assignment_distance
        cost = np.where(distance <= gate, distance, UNREACHABLE_COST)
        rows, columns = linear_sum_assignment(cost)
        return [
            (int(row), int(column))
            for row, column in zip(rows, columns, strict=True)
            if cost[row, column] <= gate
        ]
