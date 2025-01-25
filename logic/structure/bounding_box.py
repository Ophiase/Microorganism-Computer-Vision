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
