import pytest
from motility.diffusion import Motion
from motility.evaluation import majority_truth, match_frames, score_tracking
from motility.synthetic import Population, SceneParameters, simulate


def scene() -> SceneParameters:
    return SceneParameters(frames=20, population=Population(directed=5, stationary=3))


def test_the_truth_scores_itself_perfectly() -> None:
    truth = simulate(scene()).truth.table
    score = score_tracking(match_frames(truth, truth, radius=1.0), truth)

    assert score.mota == pytest.approx(1.0)
    assert score.identity_switches == 0
    assert score.mostly_tracked == pytest.approx(1.0)


def test_every_track_is_owned_by_the_cell_it_follows() -> None:
    truth = simulate(scene()).truth.table

    assert majority_truth(match_frames(truth, truth, radius=1.0)) == {
        cell: cell for cell in set(truth.track.tolist())
    }


def test_a_seed_draws_the_same_scene_and_the_population_it_asks() -> None:
    first, second = simulate(scene()), simulate(scene())

    assert (first.video == second.video).all()
    assert sorted(first.truth.motions.values()).count(Motion.DIRECTED) == 5
