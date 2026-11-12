from motility.tracking import KalmanTracker, TrackingParameters, TrackTable
from motility.vision import Detection, Rectangle


def detection_at(x: float, y: float) -> Detection:
    return Detection((x, y), Rectangle(int(x) - 2, int(y) - 2, 4, 4), 16)


def test_a_moving_cell_keeps_one_track() -> None:
    tracker = KalmanTracker(TrackingParameters(max_assignment_distance=6.0))
    observations = [
        observation
        for frame in range(10)
        for observation in tracker.step([detection_at(10 + 3 * frame, 20)], frame)
    ]

    assert {observation.track for observation in observations} == {0}
    assert [observation.frame for observation in observations] == list(range(10))


def test_a_coasting_track_reports_nothing_and_is_found_again() -> None:
    tracker = KalmanTracker(TrackingParameters(max_missed_frames=3))
    frames = [[detection_at(10, 10)], [], [], [detection_at(10, 10)]]
    observations = [
        observation
        for frame, detections in enumerate(frames)
        for observation in tracker.step(detections, frame)
    ]

    assert [(item.track, item.frame) for item in observations] == [(0, 0), (0, 3)]


def test_a_track_lost_too_long_starts_again_under_a_new_index() -> None:
    tracker = KalmanTracker(TrackingParameters(max_missed_frames=1))
    frames = [[detection_at(10, 10)], [], [], [detection_at(10, 10)]]
    tracks = {
        observation.track
        for frame, detections in enumerate(frames)
        for observation in tracker.step(detections, frame)
    }

    assert tracks == {0, 1}


def test_the_table_gives_one_ordered_trajectory_per_track() -> None:
    tracker = KalmanTracker(TrackingParameters())
    observations = [
        observation
        for frame in range(3)
        for observation in tracker.step(
            [detection_at(10 + frame, 10), detection_at(40, 40 + frame)], frame
        )
    ]
    trajectories = TrackTable.from_observations(observations).trajectories()

    assert len(trajectories) == 2
    assert all(trajectory.frames == (0, 1, 2) for trajectory in trajectories)
