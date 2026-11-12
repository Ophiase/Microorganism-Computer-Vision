from collections.abc import Sequence

from pydantic import BaseModel, Field

from motility.diffusion.models.motion import Motion
from motility.diffusion.models.selection import Verdict


class ClassificationScore(BaseModel):
    """How often the verdicts name the motion the true cell had."""

    accuracy: float = Field(
        description="Share of the determined, matched tracks given their true motion."
    )
    determined: int = Field(description="Tracks given a motion and matched to a cell.")
    undetermined: int = Field(description="Tracks too short or too sparse to classify.")
    confusion: dict[Motion, dict[Motion, int]] = Field(
        description="Counts of verdicts, by true motion then by verdict."
    )


def score_classification(
    verdicts: Sequence[Verdict],
    truth_of_track: dict[int, int],
    motions: dict[int, Motion],
) -> ClassificationScore:
    """Compare each verdict with the motion of the cell its track followed most."""
    confusion: dict[Motion, dict[Motion, int]] = {}
    undetermined = 0
    for verdict in verdicts:
        if verdict.motion is Motion.UNDETERMINED:
            undetermined += 1
            continue
        if verdict.track not in truth_of_track:
            continue
        true_motion = motions[truth_of_track[verdict.track]]
        row = confusion.setdefault(true_motion, {})
        row[verdict.motion] = row.get(verdict.motion, 0) + 1
    determined = sum(sum(row.values()) for row in confusion.values())
    correct = sum(row.get(motion, 0) for motion, row in confusion.items())
    return ClassificationScore(
        accuracy=correct / determined if determined else 0.0,
        determined=determined,
        undetermined=undetermined,
        confusion=confusion,
    )
