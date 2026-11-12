import numpy as np
import pytest
from motility.diffusion import Motion, classify, mean_squared_displacement
from motility.diffusion.models import ClassificationParameters
from motility.synthetic.motions import walk
from motility.synthetic.scene import Kinetics
from motility.tracking import Trajectory

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
