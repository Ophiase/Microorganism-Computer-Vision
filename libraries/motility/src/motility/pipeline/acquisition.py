from dataclasses import dataclass
from typing import Self

from pydantic import BaseModel, ConfigDict, Field


class Acquisition(BaseModel):
    """What the microscope recorded: the clock and the scale, when they are known."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    frame_rate: float | None = Field(
        default=None,
        gt=0.0,
        description="Frames per second; unset, the video's own rate is used.",
    )
    pixel_size: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Micrometres per pixel, from the objective and the camera; unset,"
            " lengths stay in pixels and say so."
        ),
    )


@dataclass(frozen=True, slots=True)
class Scale:
    """Converts pixels and frames into the units of the report."""

    length: float
    time: float
    length_unit: str
    time_unit: str

    @classmethod
    def from_acquisition(
        cls, acquisition: Acquisition, video_frame_rate: float | None
    ) -> Self:
        rate = acquisition.frame_rate or video_frame_rate
        return cls(
            length=acquisition.pixel_size or 1.0,
            time=1 / rate if rate else 1.0,
            length_unit="µm" if acquisition.pixel_size else "px",
            time_unit="s" if rate else "frame",
        )

    def speed(self, pixels_per_frame: float) -> float:
        return pixels_per_frame * self.length / self.time

    def diffusion(self, square_pixels_per_frame: float) -> float:
        return square_pixels_per_frame * self.length**2 / self.time

    @property
    def speed_unit(self) -> str:
        return f"{self.length_unit}/{self.time_unit}"

    @property
    def diffusion_unit(self) -> str:
        return f"{self.length_unit}²/{self.time_unit}"
