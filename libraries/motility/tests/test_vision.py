import cv2
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
from motility.vision.regions.mask import Disc, MaskParameters

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


def test_a_static_object_and_its_halo_are_excluded() -> None:
    video = dark_cell_video(frames=5)
    video[:, 30:38, 30:38] = 0.95
    parameters = PreprocessingParameters(
        background=Background.TEMPORAL_MEDIAN,
        window_low=0.2,
        window_high=0.5,
        gain=1.0,
        smoothing=1,
        mask=MaskParameters(static=0.2, margin=2),
    )
    video[2, 28, 33] = 0.1
    foreground = preprocess(video, parameters)

    assert foreground[2, 28, 33] == pytest.approx(0.0)
    assert foreground[0, 21, 12] > 0.9


def test_a_white_bar_in_a_corner_is_found_with_its_length() -> None:
    from motility.vision.measures.scale_bar import find_scale_bars

    frame = np.full((200, 300), 0.5, dtype=np.float32)
    frame[180:182, 240:270] = 1.0
    frame[60:100, 100:140] = 1.0

    bars = find_scale_bars(frame)

    assert (bars[0].x, bars[0].y, bars[0].length, bars[0].thickness) == (
        240,
        180,
        30,
        2,
    )
    assert all(bar.x != 100 for bar in bars)


def test_a_diagonal_rod_counts_as_much_as_a_level_one_in_pixels() -> None:
    from motility.vision import AreaMeasure

    level = np.zeros((SIZE, SIZE), dtype=np.float32)
    level[10:12, 5:19] = 1.0
    diagonal = np.zeros((SIZE, SIZE), dtype=np.float32)
    for step in range(14):
        diagonal[20 + step, 20 + step] = diagonal[20 + step, 21 + step] = 1.0
    parameters = DetectionParameters(area=AreaMeasure.PIXELS, min_area=10, max_area=40)

    assert len(detect(level, parameters)) == 1
    assert len(detect(diagonal, parameters)) == 1
    boxed = parameters.model_copy(update={"area": AreaMeasure.BOX})
    assert detect(diagonal, boxed) == ()


def test_a_disc_masks_by_hand_and_a_kept_one_gives_back() -> None:
    video = dark_cell_video(frames=3)
    mask = MaskParameters(ignore=(Disc(x=12, y=21, radius=4),))
    parameters = PreprocessingParameters(
        window_low=0.2, window_high=0.6, gain=1.0, smoothing=1, mask=mask
    )
    assert preprocess(video, parameters)[0, 21, 12] == pytest.approx(0.0)

    kept = mask.model_copy(update={"keep": (Disc(x=12, y=21, radius=2),)})
    parameters = parameters.model_copy(update={"mask": kept})
    assert preprocess(video, parameters)[0, 21, 12] == pytest.approx(1.0)


def test_the_axis_of_a_rod_follows_it_and_a_disc_is_not_elongated() -> None:
    from motility.vision.detection import axis_of

    rod = np.zeros((20, 20), dtype=bool)
    for step in range(12):
        rod[4 + step, 4 + step] = rod[4 + step, 5 + step] = True
    disc = np.zeros((20, 20), dtype=np.uint8)
    cv2.circle(disc, (10, 10), 6, 1, -1)

    angle, elongation = axis_of(rod)
    assert angle == pytest.approx(np.pi / 4, abs=0.1)
    assert elongation > 3
    assert axis_of(disc.astype(bool))[1] == pytest.approx(1.0, abs=0.05)
