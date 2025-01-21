from dataclasses import dataclass


@dataclass
class BoundingBox:
    index: int
    x: int
    y: int
    w: int
    h: int
    visible: bool

    def __getitem__(self, item):
        return (self.index, self.x, self.y, self.w, self.h, self.visible)[item]
