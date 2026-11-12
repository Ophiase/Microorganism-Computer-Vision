from pydantic import BaseModel, ConfigDict, Field

from motility.diffusion.models.parameters import ClassificationParameters
from motility.pipeline.acquisition import Acquisition
from motility.tracking.kalman_filter import TrackingParameters
from motility.vision.detection import DetectionParameters
from motility.vision.preprocessing import PreprocessingParameters


class PipelineParameters(BaseModel):
    """Every knob of the pipeline, stage by stage; the defaults are the 2025 study's."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    acquisition: Acquisition = Field(default_factory=Acquisition)
    preprocessing: PreprocessingParameters = Field(
        default_factory=PreprocessingParameters
    )
    detection: DetectionParameters = Field(default_factory=DetectionParameters)
    tracking: TrackingParameters = Field(default_factory=TrackingParameters)
    classification: ClassificationParameters = Field(
        default_factory=ClassificationParameters
    )
