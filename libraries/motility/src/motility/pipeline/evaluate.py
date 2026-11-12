from dataclasses import dataclass

from pydantic import BaseModel, Field

from motility.diffusion.models.selection import classify
from motility.pipeline.parameters import PipelineParameters
from motility.pipeline.run import PipelineResult, run_pipeline
from motility.truth.evaluation.classification_score import (
    ClassificationScore,
    score_classification,
)
from motility.truth.evaluation.matching import majority_truth, match_frames
from motility.truth.evaluation.tracking_score import TrackingScore, score_tracking
from motility.truth.synthetic import Simulation


class Evaluation(BaseModel):
    """A configuration scored on a synthetic video against its truth."""

    tracking: TrackingScore
    classification: ClassificationScore
    oracle: ClassificationScore = Field(
        description=(
            "The classifier alone, on the true trajectories: what the verdicts"
            " would be with a perfect tracker."
        )
    )


@dataclass(frozen=True, slots=True)
class ScoredRun:
    """A pipeline run on a synthetic video, and its score."""

    result: PipelineResult
    evaluation: Evaluation


def evaluate(
    simulation: Simulation, parameters: PipelineParameters, radius: float
) -> ScoredRun:
    """Run the pipeline on a synthetic video and score it, pairs within `radius` px."""
    result = run_pipeline(simulation.video, parameters)
    truth = simulation.truth
    matches = match_frames(truth.table, result.tracks, radius)
    oracle = tuple(
        classify(trajectory, parameters.classification)
        for trajectory in truth.table.trajectories()
    )
    identity = {cell: cell for cell in truth.motions}
    return ScoredRun(
        result,
        Evaluation(
            tracking=score_tracking(matches, truth.table),
            classification=score_classification(
                result.verdicts, majority_truth(matches), truth.motions
            ),
            oracle=score_classification(oracle, identity, truth.motions),
        ),
    )
