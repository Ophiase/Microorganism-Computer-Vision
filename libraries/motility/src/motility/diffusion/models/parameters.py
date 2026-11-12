from pydantic import BaseModel, ConfigDict, Field


class ClassificationParameters(BaseModel):
    """How a trajectory is given a motion, from its MSD and its turning angles."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_track_length: int = Field(
        default=20,
        ge=6,
        description=(
            "Observations a track needs to be classified; shorter ones are"
            " undetermined, and left out of the distributions."
        ),
    )
    lag_fraction: float = Field(
        default=0.33,
        gt=0.0,
        le=0.5,
        description=(
            "Longest MSD lag as a share of the track's duration: the long lags"
            " average few pairs and are noisy."
        ),
    )
    min_lags: int = Field(
        default=5,
        ge=4,
        description="Fewest MSD lags the models are fitted on; fewer is undetermined.",
    )
    circular_turns: float = Field(
        default=0.5,
        gt=0.0,
        description="Net turns, signed, a track must make to be called circular.",
    )
    circular_significance: float = Field(
        default=0.999,
        gt=0.0,
        lt=1.0,
        description=(
            "How far from chance the mean turning angle must be, one minus its"
            " p-value, for a track to be called circular."
        ),
    )
    min_step: float = Field(
        default=0.5,
        ge=0.0,
        description=(
            "Step in pixels below which a heading is jitter and not counted in the"
            " turning angles."
        ),
    )
    turning_stride: int = Field(
        default=3,
        ge=1,
        description=(
            "Observations between the steps whose headings are compared: longer"
            " steps lift the heading above the localisation noise."
        ),
    )
