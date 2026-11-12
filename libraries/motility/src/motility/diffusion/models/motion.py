from enum import StrEnum


class Motion(StrEnum):
    """The ways a cell is said to move, shared by the truth and the verdicts."""

    STATIONARY = "stationary"
    DIFFUSIVE = "diffusive"
    DIRECTED = "directed"
    CONFINED = "confined"
    CIRCULAR = "circular"
    UNDETERMINED = "undetermined"
