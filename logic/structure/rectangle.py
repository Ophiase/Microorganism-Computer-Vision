from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Rectangle:
    """An axis-aligned box in pixels, its origin at the top left of the frame."""

    x: int
    y: int
    width: int
    height: int

    @property
    def centroid(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)
