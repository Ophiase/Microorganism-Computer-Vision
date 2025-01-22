from dataclasses import dataclass
from typing import Tuple


@dataclass
class BoundingBox:
    index: int
    x: int
    y: int
    w: int
    h: int
    visible: bool

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
