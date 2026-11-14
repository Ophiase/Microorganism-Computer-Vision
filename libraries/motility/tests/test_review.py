import numpy as np
import pytest
from motility.review import (
    Action,
    Correction,
    SuspectKind,
    correct_tracks,
    find_suspects,
)
from motility.tracking import TrackTable


def table(rows: list[tuple[int, int, float, float]]) -> TrackTable:
    track, frame, x, y = (np.array(column) for column in zip(*rows, strict=True))
    return TrackTable.from_columns({"track": track, "frame": frame, "x": x, "y": y})


def straight(track: int, frames: range, start: float, step: float = 2.0) -> list:
    return [(track, frame, start + step * frame, 10.0) for frame in frames]


def test_a_split_gives_the_later_frames_a_new_track() -> None:
    tracks = table(straight(0, range(6), 0.0))
    split = Correction(action=Action.SPLIT, track=0, frame=3, reason="a swap")

    corrected = correct_tracks(tracks, [split])

    assert corrected.track.tolist() == [0, 0, 0, 1, 1, 1]


def test_a_join_keeps_the_first_track_where_both_hold_a_frame() -> None:
    tracks = table(straight(0, range(4), 0.0) + straight(1, range(3, 7), 0.5))
    join = Correction(action=Action.JOIN, track=0, other=1, reason="one cell")

    corrected = correct_tracks(tracks, [join])

    assert set(corrected.track.tolist()) == {0}
    assert sorted(corrected.frame.tolist()) == list(range(7))
    assert corrected.x[corrected.frame == 3][0] == pytest.approx(6.0)


def test_a_swap_is_suspected_where_it_happens() -> None:
    rows = straight(0, range(5), 0.0) + [
        (0, frame, 80.0 + 2 * frame, 40.0) for frame in range(5, 10)
    ]
    suspects = find_suspects(table(rows), [])

    jump = next(item for item in suspects if item.kind is SuspectKind.JUMP)
    assert (jump.track, jump.frame) == (0, 5)


def test_a_broken_track_is_suspected_of_continuing() -> None:
    tracks = table(straight(0, range(5), 0.0) + straight(1, range(7, 12), 0.0))

    gap = next(
        item for item in find_suspects(tracks, []) if item.kind is SuspectKind.GAP
    )
    assert (gap.track, gap.other) == (0, 1)


def test_a_forced_cell_is_adopted_without_duplicating_a_kept_one() -> None:
    from motility.review import adopt

    kept = table(straight(0, range(6), 0.0))
    relaxed = table(straight(5, range(6), 0.0) + straight(6, range(6), 0.0, step=-2.0))

    adopted = adopt(kept, relaxed, [(3, -6.0, 10.0)], radius=2.0)

    assert set(adopted.track.tolist()) == {0, 1}
    assert (adopted.track == 1).sum() == 5


def test_reversals_in_rhythm_are_found_and_random_turns_are_not() -> None:
    from motility.diffusion.evidence.turning_tests import periodic
    from motility.tracking import Trajectory
    from motility.truth.synthetic.behaviours import Behaviour, behave
    from motility.truth.synthetic.scene import Kinetics

    generator = np.random.default_rng(1)

    def p_value(behaviour: Behaviour) -> float:
        path = behave(behaviour, Kinetics(), 120, generator)
        trajectory = Trajectory(0, tuple(range(120)), tuple(map(tuple, path.tolist())))
        evidence = periodic(trajectory)
        return 1.0 if evidence is None or evidence.p_value is None else evidence.p_value

    assert p_value(Behaviour.RUN_AND_REVERSE) < 0.01
    assert p_value(Behaviour.BROWNIAN) > 0.05


def test_a_rod_kicked_along_its_axis_is_found_and_an_isotropic_one_is_not() -> None:
    from motility.diffusion.evidence.body_tests import body_frame
    from motility.tracking import Trajectory
    from motility.truth.synthetic.behaviours import Behaviour, behave_oriented
    from motility.truth.synthetic.scene import Kinetics

    generator = np.random.default_rng(4)

    def p_value(behaviour: Behaviour) -> float:
        path = behave_oriented(behaviour, Kinetics(), 150, generator)
        trajectory = Trajectory(
            0,
            tuple(range(150)),
            tuple(map(tuple, path.positions.tolist())),
            tuple(path.angles.tolist()),
            2.5,
        )
        evidence = body_frame(trajectory)
        return 1.0 if evidence is None or evidence.p_value is None else evidence.p_value

    assert p_value(Behaviour.BODY_FRAME) < 0.01
    assert p_value(Behaviour.BROWNIAN) > 0.05
