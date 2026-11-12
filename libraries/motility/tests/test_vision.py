import numpy as np
import pytest
from motility.vision import (
    Background,
    DetectionParameters,
    Polarity,
    PreprocessingParameters,
    detect,
    preprocess,
)
from motility.vision.preprocessing import smooth

SIZE = 48


def dark_cell_video(frames: int = 4) -> np.ndarray:
    video = np.full((frames, SIZE, SIZE), 0.6, dtype=np.float32)
    for frame in range(frames):
        video[frame, 20:24, 10 + 3 * frame : 16 + 3 * frame] = 0.2
    return video


def test_smoothing_keeps_a_flat_field_flat_up_to_the_border() -> None:
    flat = np.full((SIZE, SIZE), 0.5, dtype=np.float32)
    np.testing.assert_allclose(smooth(flat, 5), flat, atol=1e-6)


def test_a_dark_cell_becomes_bright_on_black() -> None:
    parameters = PreprocessingParameters(
        polarity=Polarity.DARK, window_low=0.2, window_high=0.6, gain=1.0, smoothing=1
    )
    foreground = preprocess(dark_cell_video(), parameters)

    assert foreground[0, 21, 12] == pytest.approx(1.0)
    assert foreground[0, 0, 0] == pytest.approx(0.0)


def test_the_median_background_erases_what_never_moves() -> None:
    video = dark_cell_video(frames=9)
    video[:, 40:44, 40:44] = 0.1
    parameters = PreprocessingParameters(
        background=Background.TEMPORAL_MEDIAN,
        window_low=0.2,
        window_high=0.5,
        gain=1.0,
        smoothing=1,
    )
    foreground = preprocess(video, parameters)

    assert foreground[:, 40:44, 40:44].max() == pytest.approx(0.0)
    assert foreground[0, 21, 12] > 0.9


def test_detection_finds_each_cell_at_its_centre_of_mass() -> None:
    foreground = np.zeros((SIZE, SIZE), dtype=np.float32)
    foreground[10:14, 10:16] = 1.0
    foreground[30:33, 30:33] = 1.0

    detections = detect(foreground, DetectionParameters(min_area=4, max_area=100))

    centres = sorted(detection.centroid for detection in detections)
    assert centres == [(12.5, 11.5), (31.0, 31.0)]


def test_the_area_bounds_drop_specks_and_debris() -> None:
    foreground = np.zeros((SIZE, SIZE), dtype=np.float32)
    foreground[5, 5] = 1.0
    foreground[20:40, 20:40] = 1.0

    assert detect(foreground, DetectionParameters(min_area=4, max_area=100)) == ()
