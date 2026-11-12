from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Self

import numpy as np
from numba import njit
from scipy.optimize import linear_sum_assignment

from logic.structure.bounding_box import BoundingBox
from logic.structure.rectangle import Rectangle

TRANSITION = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
OBSERVATION = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
POSITION = slice(0, 2)
VELOCITY = slice(2, 4)
UNREACHABLE_COST = 1e9


@dataclass(slots=True)
class Track:
    """A Kalman state `[x, y, vx, vy]` with its box, its age and its misses."""

    index: int
    rectangle: Rectangle
    state: np.ndarray
    covariance: np.ndarray
    age: int = 0
    missed: int = 0
    history: list[tuple[int, Rectangle]] = field(default_factory=list)

    @classmethod
    def start(cls, index: int, rectangle: Rectangle) -> Self:
        x, y = rectangle.centroid
        return cls(
            index=index,
            rectangle=rectangle,
            state=np.array([x, y, 0.0, 0.0]),
            covariance=np.eye(4) * 0.1,
        )


class BacterialTracker:
    def __init__(
        self,
        max_missed_frames: int = 15,
        process_noise: float = 1e-5,
        measurement_noise: float = 1e-5,
        max_assignment_distance: float = 30.0,
        max_velocity: float = 50.0,
    ) -> None:
        self.tracks: dict[int, Track] = {}
        self.next_index = 0
        self.max_missed = max_missed_frames
        self.process_covariance = np.eye(4) * process_noise
        self.measurement_covariance = np.eye(2) * measurement_noise
        self.max_assignment_distance = max_assignment_distance
        self.max_velocity = max_velocity

    def update_tracks(
        self, frame_boxes: Sequence[BoundingBox], frame_index: int
    ) -> tuple[BoundingBox, ...]:
        """Predict every track, match it to this frame's boxes, then age the rest."""
        for track in self.tracks.values():
            self._predict(track)

        matched = self._match(list(self.tracks.values()), frame_boxes)
        used = set()
        for track_position, box_position in matched:
            track = list(self.tracks.values())[track_position]
            self._update(track, frame_boxes[box_position].centroid)
            track.missed = 0
            track.age += 1
            track.history.append((frame_index, track.rectangle))
            used.add(box_position)

        for box_position in set(range(len(frame_boxes))) - used:
            self.tracks[self.next_index] = Track.start(
                self.next_index, frame_boxes[box_position].rectangle
            )
            self.next_index += 1

        for track in self.tracks.values():
            track.missed += 1
        self.tracks = {
            index: track
            for index, track in self.tracks.items()
            if track.missed <= self.max_missed
        }

        return tuple(
            BoundingBox(track.rectangle, track.index, tracked=track.missed > 0)
            for track in self.tracks.values()
        )

    def _predict(self, track: Track) -> None:
        track.state = TRANSITION @ track.state
        track.state[VELOCITY] = np.clip(
            track.state[VELOCITY], -self.max_velocity, self.max_velocity
        )
        track.covariance = (
            TRANSITION @ track.covariance @ TRANSITION.T + self.process_covariance
        )
        track.rectangle = self._rectangle(track)

    def _update(self, track: Track, measurement: tuple[float, float]) -> None:
        innovation = np.array(measurement) - OBSERVATION @ track.state
        innovation_covariance = (
            OBSERVATION @ track.covariance @ OBSERVATION.T + self.measurement_covariance
        )
        gain = track.covariance @ OBSERVATION.T @ np.linalg.inv(innovation_covariance)
        track.state += gain @ innovation
        track.state[VELOCITY] = np.clip(
            track.state[VELOCITY], -self.max_velocity, self.max_velocity
        )
        track.covariance = (np.eye(4) - gain @ OBSERVATION) @ track.covariance
        track.rectangle = self._rectangle(track)

    def _rectangle(self, track: Track) -> Rectangle:
        x, y = track.state[POSITION]
        width, height = track.rectangle.width, track.rectangle.height
        return Rectangle(int(x - width / 2), int(y - height / 2), width, height)

    def _match(
        self, tracks: Sequence[Track], detections: Sequence[BoundingBox]
    ) -> list[tuple[int, int]]:
        predicted = np.array([track.state[POSITION] for track in tracks]).reshape(-1, 2)
        measured = np.array([box.centroid for box in detections]).reshape(-1, 2)
        cost = self._cost_matrix(predicted, measured, self.max_assignment_distance)
        rows, columns = linear_sum_assignment(cost)
        return [
            (row, column)
            for row, column in zip(rows, columns, strict=True)
            if cost[row, column] <= self.max_assignment_distance
        ]

    @staticmethod
    @njit
    def _cost_matrix(
        predicted: np.ndarray, measured: np.ndarray, max_distance: float
    ) -> np.ndarray:
        cost = np.full((predicted.shape[0], measured.shape[0]), UNREACHABLE_COST)
        for i in range(predicted.shape[0]):
            for j in range(measured.shape[0]):
                dx = predicted[i, 0] - measured[j, 0]
                dy = predicted[i, 1] - measured[j, 1]
                distance = (dx * dx + dy * dy) ** 0.5
                if distance <= max_distance:
                    cost[i, j] = distance
        return cost
