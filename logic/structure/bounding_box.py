from dataclasses import dataclass
from typing import Tuple
from logic.structure.rectangle import Rectangle
from numba import jit

DEFAULT_INDEX: int = 0

@dataclass
class BoundingBox:
    index: int
    rectangle: Rectangle
    visible: bool

    def __init__(
        self, rectangle: Rectangle | Tuple[int, int, int, int], index: int = DEFAULT_INDEX, visible: bool = False
    ):
        self.rectangle = rectangle if isinstance(rectangle, Rectangle) else Rectangle(*rectangle)
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

    @staticmethod
    @jit(nopython=True)
    def _centroid(x: int, y: int, w: int, h: int) -> Tuple[float, float]:
        return x + w / 2, y + h / 2

    @staticmethod
    @jit(nopython=True)
    def _icentroid(x: int, y: int, w: int, h: int) -> Tuple[int, int]:
        return int(x + w / 2), int(y + h / 2)

    def centroid(self) -> Tuple[float, float]:
        return self._centroid(self.x, self.y, self.w, self.h)

    def icentroid(self) -> Tuple[int, int]:
        return self._icentroid(self.x, self.y, self.w, self.h)
