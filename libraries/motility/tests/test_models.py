import numpy as np
import pytest
from motility.diffusion import Motion, classify, mean_squared_displacement
from motility.diffusion.models import ClassificationParameters
from motility.tracking import Trajectory
from motility.truth.synthetic.motions import walk
from motility.truth.synthetic.scene import Kinetics

FRAMES = 80
CELLS = 10


def verdicts_of(motion: Motion) -> list[Motion]:
    generator = np.random.default_rng(7)
    paths = [
        walk(motion, Kinetics(), FRAMES, (0.0, 0.0), generator) for _ in range(CELLS)
    ]
    return [
        classify(
            Trajectory(0, tuple(range(FRAMES)), tuple(map(tuple, path.positions))),
            ClassificationParameters(),
        ).motion
        for path in paths
    ]


@pytest.mark.parametrize(
    "motion", [Motion.STATIONARY, Motion.DIRECTED, Motion.CONFINED, Motion.CIRCULAR]
)
def test_the_clear_motions_are_recognised(motion: Motion) -> None:
    assert verdicts_of(motion).count(motion) >= 0.8 * CELLS


def test_a_short_track_stays_undetermined() -> None:
    trajectory = Trajectory(0, (0, 1, 2), ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)))

    assert (
        classify(trajectory, ClassificationParameters()).motion is Motion.UNDETERMINED
    )


def test_the_msd_measures_lags_in_frames_across_a_gap() -> None:
    trajectory = Trajectory(0, (0, 1, 3), ((0.0, 0.0), (1.0, 0.0), (3.0, 0.0)))
    curve = mean_squared_displacement(trajectory, max_lag=3)

    assert curve.lags == (1, 2, 3)
    assert curve.squared == (1.0, 4.0, 9.0)


def test_speed_slope_is_flat_when_no_step_lies_outside_the_decor() -> None:
    from motility.diffusion.evidence.context_tests import speed_slope

    inside = np.zeros((10, 10), dtype=np.float32)
    track = np.array([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    assert not speed_slope([track], inside)


def test_cells_slowed_near_the_decor_give_a_slope_above_the_surrogates() -> None:
    from motility.diffusion.evidence.context_tests import SurrogateTest, speed_slope
    from motility.tracking import Trajectory

    objects = np.zeros((200, 200), dtype=bool)
    objects[:, :20] = True
    generator = np.random.default_rng(0)
    tracks = []
    for index in range(60):
        x, y = generator.uniform(25, 180), generator.uniform(10, 190)
        positions = []
        for _ in range(30):
            step, heading = 0.02 * (x - 20), generator.uniform(0, 2 * np.pi)
            x = float(np.clip(x + step * np.cos(heading), 21, 199))
            y = float(np.clip(y + step * np.sin(heading), 0, 199))
            positions.append((x, y))
        tracks.append(
            Trajectory(index, tuple(range(30)), tuple(positions), (0.0,) * 30, 1.0)
        )
    outcome = SurrogateTest(speed_slope, alternative_greater=True).run(tracks, objects)
    assert outcome is not None
    assert np.mean(outcome.null >= outcome.observed) < 0.05
