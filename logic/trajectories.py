from typing import List, Optional, Tuple
from .bounding_box import BoundingBox
import numpy as np

class Trajectory:
    def __init__(self, bboxes: List[Optional[BoundingBox]]):
        self.bboxes = bboxes
        self.valid_entries = [(i, bbox) for i, bbox in enumerate(bboxes) if bbox is not None]
    
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