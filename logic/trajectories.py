from typing import List, Optional, Tuple
from .structure.bounding_box import BoundingBox
import numpy as np


class Trajectory:
    def __init__(self, bboxes: List[Optional[BoundingBox]]):
        self.bboxes = bboxes
        self.valid_entries = [(i, bbox)
                              for i, bbox in enumerate(bboxes) if bbox is not None]

    def positions(self) -> List[Tuple[float, float]]:
        return [bbox.centroid() for _, bbox in self.valid_entries]

    def frames(self) -> List[int]:
        return [i for i, _ in self.valid_entries]

    def displacements(self) -> List[Tuple[float, float]]:
        positions = self.positions()
        return [
            (pos2[0]-pos1[0], pos2[1]-pos1[1])
            for pos1, pos2 in zip(positions[:-1], positions[1:])
        ]

    def time_deltas(self) -> List[int]:
        frames = self.frames()
        return [t2 - t1 for t1, t2 in zip(frames[:-1], frames[1:])]

    def average_speed(self) -> float:
        """Calculate the average speed of the trajectory in pixels/frame"""
        displacements = self.displacements()
        time_deltas = self.time_deltas()

        if len(displacements) == 0 or len(time_deltas) == 0:
            return 0.0

        speeds = [
            np.sqrt(dx**2 + dy**2) / dt
            for (dx, dy), dt in zip(displacements, time_deltas)
        ]
        return float(np.mean(speeds)) if speeds else 0.0


def restructure_data(tracked_data: List[List[BoundingBox]]) -> List[List[Optional[BoundingBox]]]:
    max_index = max(
        bbox.index for frame in tracked_data for bbox in frame) if tracked_data else 0
    total_frames = len(tracked_data)

    restructured = [[] for _ in range(max_index + 1)]

    for idx in range(max_index + 1):
        restructured[idx] = [None] * total_frames

    for frame_idx, frame in enumerate(tracked_data):
        for bbox in frame:
            if bbox.visible:
                restructured[bbox.index][frame_idx] = bbox
    return restructured