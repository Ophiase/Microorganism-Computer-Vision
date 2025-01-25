from dataclasses import dataclass
from typing import Tuple


@dataclass
class Rectangle:
    x: int
    y: int
    w: int
    h: int

    def centroid(self) -> tuple[float, float]:
        return (self.x + self.w/2, self.y + self.h/2)

    def area(self) -> int:
        return self.w * self.h

    #########################################

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)
