from dataclasses import dataclass
from typing import Tuple
from numba import jit


@dataclass
class Rectangle:
    x: int
    y: int
    w: int
    h: int

    @staticmethod
    @jit(nopython=True)
    def _centroid(x: int, y: int, w: int, h: int) -> Tuple[float, float]:
        return x + w / 2, y + h / 2

    def centroid(self) -> tuple[float, float]:
        return self._centroid(self.x, self.y, self.w, self.h)

    def area(self) -> int:
        return self.w * self.h

    #########################################

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)
