import numpy as np
import pytest

from logic.filters import pre_transform, tensor_gradient, transform, transform_video
from logic.kernel import KernelType, average_neighbors
from logic.structure.bounding_box import BoundingBox
from logic.structure.rectangle import Rectangle
from logic.trajectories import Trajectory

RESOLUTION = 32
FRAMES = 3
CHANNELS = 3
TRANSFORMS = 4


@pytest.fixture
def noise_video() -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed=100)
    video = generator.random((FRAMES, RESOLUTION, RESOLUTION))
    flow = generator.random((FRAMES, RESOLUTION, RESOLUTION, 2))
    return video, flow


def test_gradient_is_a_forward_difference() -> None:
    tensor = np.random.default_rng(seed=1).random((RESOLUTION, RESOLUTION))
    gradient = tensor_gradient(tensor)

    assert gradient.shape == (RESOLUTION, RESOLUTION, 2)
    np.testing.assert_allclose(gradient[1:, 1:, 0], tensor[1:, 1:] - tensor[1:, :-1])
    np.testing.assert_allclose(gradient[1:, 1:, 1], tensor[1:, 1:] - tensor[:-1, 1:])


def test_pre_transform_stays_in_the_unit_interval() -> None:
    tensor = np.random.default_rng(seed=2).random((RESOLUTION, RESOLUTION)) * 3 - 1
    result = pre_transform(tensor)

    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_every_frame_gets_a_trend_and_two_gradients(noise_video) -> None:
    video, flow = noise_video
    processed = transform_video(video, flow)

    assert processed.shape == (FRAMES, RESOLUTION, RESOLUTION, CHANNELS, TRANSFORMS)
    np.testing.assert_allclose(processed[1], transform_frame(video[1], flow[1]))


def transform_frame(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
    return np.stack(
        (transform(frame), transform(flow[:, :, 0]), transform(flow[:, :, 1])), axis=-2
    )


def test_averaging_neighbours_preserves_a_flat_field() -> None:
    flat = np.full((RESOLUTION, RESOLUTION), 0.5)
    for kernel_type in KernelType:
        np.testing.assert_allclose(
            average_neighbors(flat, kernel_type), flat, atol=1e-6
        )


def test_a_trajectory_reads_the_ranked_boxes_of_its_track() -> None:
    boxes = [
        [BoundingBox(Rectangle(frame * 10, 0, 20, 20), index=frame, tracked=True)]
        for frame in range(FRAMES)
    ]
    trajectories = Trajectory.from_frames(boxes)

    assert len(trajectories) == FRAMES
    assert trajectories[0].frames == (0,)
    assert trajectories[0].positions == ((10.0, 10.0),)


def test_a_position_ranks_by_its_displacement() -> None:
    boxes = [
        BoundingBox(Rectangle(step * 4, 0, 2, 2), index=0, tracked=True)
        for step in range(FRAMES)
    ]
    trajectory = Trajectory.from_frames([[box] for box in boxes])[0]

    assert trajectory.displacements == ((4.0, 0.0), (4.0, 0.0))
    assert trajectory.average_speed == pytest.approx(4.0)
