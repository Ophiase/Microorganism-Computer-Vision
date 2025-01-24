from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class BoundingBox:
    index: int
    x: int
    y: int
    w: int
    h: int
    visible: bool

    #########################################

    def __init__(self, index=0, x=0, y=0, w=0, h=0, visible=False):
        self.index = index
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.visible = visible

    def getBbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def toTupple(self) -> Tuple[int, int, int, int, int, bool]:
        return (self.index, self.x, self.y, self.w, self.h, self.visible)

    def __getitem__(self, item):
        return self.toTupple()[item]

    #########################################

    def centroid(self) -> tuple[float, float]:
        return (self.x + self.w/2, self.y + self.h/2)


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
