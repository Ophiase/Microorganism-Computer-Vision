from pydantic import BaseModel, ConfigDict, Field

from motility.vision.preprocessing import Polarity

FROZEN = ConfigDict(extra="forbid", frozen=True)


class Population(BaseModel):
    """How many cells of each motion the scene holds."""

    model_config = FROZEN

    stationary: int = Field(default=8, ge=0, description="Cells that only jitter.")
    diffusive: int = Field(default=8, ge=0, description="Brownian walkers.")
    directed: int = Field(
        default=12, ge=0, description="Swimmers with a slowly wandering heading."
    )
    confined: int = Field(default=4, ge=0, description="Cells held in a small well.")
    circular: int = Field(
        default=4, ge=0, description="Swimmers circling, as near a surface."
    )


class Kinetics(BaseModel):
    """The motion parameters, in pixels and frames."""

    model_config = FROZEN

    jitter: float = Field(
        default=0.3, ge=0.0, description="Deviation of the stationary jitter, px."
    )
    diffusion: float = Field(
        default=0.6, gt=0.0, description="Diffusion coefficient, px^2 per frame."
    )
    speed: float = Field(
        default=2.5, gt=0.0, description="Swimming speed, px per frame."
    )
    speed_spread: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Relative spread of the speeds."
    )
    rotational_diffusion: float = Field(
        default=0.005, ge=0.0, description="Heading variance per frame, rad^2."
    )
    confinement: float = Field(
        default=3.0,
        gt=0.0,
        description="Radius of the well a confined cell stays in, px.",
    )
    angular_speed: float = Field(
        default=0.12,
        gt=0.0,
        description="Turning rate of a circling cell, rad per frame.",
    )


class Appearance(BaseModel):
    """What the microscope shows: intensities in [0, 1], sizes in pixels."""

    model_config = FROZEN

    polarity: Polarity = Field(
        default=Polarity.DARK,
        description="Cells darker or brighter than the background.",
    )
    background: float = Field(
        default=0.55, ge=0.0, le=1.0, description="Mean background level."
    )
    contrast: float = Field(
        default=0.25, gt=0.0, le=1.0, description="Peak intensity difference of a cell."
    )
    contrast_spread: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Relative spread of the cells' contrast, as focus varies.",
    )
    length: float = Field(default=7.0, gt=0.0, description="Cell length, px.")
    width: float = Field(default=3.0, gt=0.0, description="Cell width, px.")
    noise: float = Field(
        default=0.03, ge=0.0, description="Deviation of the pixel noise."
    )
    blur: float = Field(
        default=0.7, ge=0.0, description="Gaussian blur of the optics, px."
    )
    gradient: float = Field(
        default=0.1, ge=0.0, description="Illumination drop across the frame."
    )
    texture: float = Field(
        default=0.03, ge=0.0, description="Amplitude of the static background texture."
    )
    dust: int = Field(default=15, ge=0, description="Static specks that never move.")
    dust_contrast: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Peak intensity difference of a speck."
    )


class SceneParameters(BaseModel):
    """A synthetic video with a known truth: who is where, and how each one moves."""

    model_config = FROZEN

    width: int = Field(default=384, ge=32, description="Frame width, px.")
    height: int = Field(default=288, ge=32, description="Frame height, px.")
    frames: int = Field(default=80, ge=4, description="Number of frames.")
    seed: int = Field(default=0, ge=0, description="Seed of every random draw.")
    population: Population = Field(default_factory=Population)
    kinetics: Kinetics = Field(default_factory=Kinetics)
    appearance: Appearance = Field(default_factory=Appearance)
