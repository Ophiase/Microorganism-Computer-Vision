from motility.pipeline.parameters import PipelineParameters
from motility.synthetic.scene import Appearance, Kinetics, Population, SceneParameters
from motility.tracking.kalman_filter import TrackingParameters
from motility.vision.appearance import AppearanceReport
from motility.vision.detection import DetectionParameters
from motility.vision.preprocessing import (
    MID_GREY,
    Background,
    Polarity,
    PreprocessingParameters,
)

STATIC_SHARE_FOR_BACKGROUND = 0.2
SMALLEST_SHARE = 0.3
LARGEST_FACTOR = 4.0
GATE_FACTOR = 2.5
MIN_GATE = 8.0
DEFAULT_LENGTH = 6.0


def suggest(appearance: AppearanceReport) -> PipelineParameters:
    """A first configuration read from the measures: a start, never an answer.

    The window spans the measured contrast on the side of the polarity, so the
    detection threshold at 0.5 cuts at half the depth of a typical cell.
    """
    length = appearance.object_length or DEFAULT_LENGTH
    width = appearance.object_width or length / 2
    box = length * width
    background = (
        Background.TEMPORAL_MEDIAN
        if appearance.static_share >= STATIC_SHARE_FOR_BACKGROUND
        else Background.NONE
    )
    centre = (
        MID_GREY if background is Background.TEMPORAL_MEDIAN else appearance.background
    )
    depth = max(appearance.contrast, 1e-3)
    low, high = (
        (centre - depth, centre)
        if appearance.polarity is Polarity.DARK
        else (centre, centre + depth)
    )
    return PipelineParameters(
        preprocessing=PreprocessingParameters(
            background=background,
            polarity=appearance.polarity,
            window_low=max(low, 0.0),
            window_high=min(high, 1.0),
            gain=1.0,
            smoothing=3,
        ),
        detection=DetectionParameters(
            min_area=max(int(box * SMALLEST_SHARE), 1),
            max_area=max(int(box * LARGEST_FACTOR), 2),
        ),
        tracking=TrackingParameters(
            max_assignment_distance=max(GATE_FACTOR * length, MIN_GATE),
            max_missed_frames=3,
        ),
    )


def twin(
    appearance: AppearanceReport,
    population: Population,
    kinetics: Kinetics,
    seed: int = 0,
) -> SceneParameters:
    """A synthetic scene that looks like the measured video, with a known truth.

    The appearance comes from the measures; the population and the kinetics are
    what the agent believes the cells do, to be checked against the real tracks.
    """
    return SceneParameters(
        width=appearance.width,
        height=appearance.height,
        frames=appearance.frames,
        seed=seed,
        population=population,
        kinetics=kinetics,
        appearance=Appearance(
            polarity=appearance.polarity,
            background=min(max(appearance.background, 0.0), 1.0),
            contrast=min(max(appearance.contrast, 0.01), 1.0),
            length=appearance.object_length or DEFAULT_LENGTH,
            width=appearance.object_width or DEFAULT_LENGTH / 2,
            noise=appearance.noise,
            gradient=abs(appearance.gradient),
            dust=round(appearance.static_share * appearance.objects_per_frame),
        ),
    )
