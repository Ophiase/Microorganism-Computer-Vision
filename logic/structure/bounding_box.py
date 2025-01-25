from dataclasses import dataclass
from typing import List, Optional, Tuple
from logic.structure.rectangle import Rectangle

DEFAULT_INDEX: int = 0


@dataclass
class BoundingBox:

    index: int
    rectangle: Rectangle
    visible: bool

    #########################################

    def __init__(
            self,
            rectangle: Rectangle | Tuple[int, int, int, int],
            index: int = DEFAULT_INDEX,
            visible: bool = False):
        if isinstance(rectangle, Rectangle):
            self.rectangle = rectangle
        else:
            self.rectangle = Rectangle(*rectangle)

        self.index = index
        self.visible = visible

    def getBbox(self) -> Tuple[int, int, int, int]:
        return self.rectangle.to_tuple()

    def toTupple(self) -> Tuple[int, int, int, int, int, bool]:
        return (self.index, self.x, self.y, self.w, self.h, self.visible)

    def __getitem__(self, item):
        return self.toTupple()[item]

    @property
    def x(self) -> int:
        return self.rectangle.x

    @property
    def y(self) -> int:
        return self.rectangle.y

    @property
    def w(self) -> int:
        return self.rectangle.w

    @property
    def h(self) -> int:
        return self.rectangle.h

    #########################################

    def centroid(self) -> tuple[float, float]:
        return (self.x + self.w/2, self.y + self.h/2)
    
    def icentroid(self) -> tuple[int, int]:
        return (int(self.x + self.w/2), int(self.y + self.h/2))


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
