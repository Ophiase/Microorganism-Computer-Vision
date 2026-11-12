from dataclasses import dataclass
from typing import Self

from motility.tracking.rectangle import Rectangle

DEFAULT_INDEX = 0


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """A rectangle carrying the identity the tracker gave it.

    `tracked` tells a box the tracker produced, which enters a trajectory,
    from a raw detection, which does not: a detection is what the tracker
    reads, not what it reports.
    """

    rectangle: Rectangle
    index: int = DEFAULT_INDEX
    tracked: bool = False

    @classmethod
    def from_tuple(
        cls,
        x: int,
        y: int,
        width: int,
        height: int,
        index: int = DEFAULT_INDEX,
        tracked: bool = False,
    ) -> Self:
        return cls(Rectangle(x, y, width, height), index, tracked)

    @property
    def x(self) -> int:
        return self.rectangle.x

    @property
    def y(self) -> int:
        return self.rectangle.y

    @property
    def width(self) -> int:
        return self.rectangle.width

    @property
    def height(self) -> int:
        return self.rectangle.height

    @property
    def centroid(self) -> tuple[float, float]:
        return self.rectangle.centroid
