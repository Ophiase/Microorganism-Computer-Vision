import pytest
from motility.vision import Background
from pydantic import ValidationError

from microorganism_computer_vision.configuration import Configuration
from microorganism_computer_vision.media.video import UsageError


def test_an_override_reads_its_value_as_toml() -> None:
    configuration = Configuration().with_overrides([
        "detection.threshold=0.35",
        "preprocessing.background=temporal-median",
        "clip.stop=40",
    ])

    assert configuration.detection.threshold == pytest.approx(0.35)
    assert configuration.preprocessing.background is Background.TEMPORAL_MEDIAN
    assert configuration.clip.stop == 40


def test_an_unknown_parameter_is_refused_with_its_name() -> None:
    with pytest.raises(UsageError, match="treshold"):
        Configuration().with_overrides(["detection.treshold=0.3"])


def test_a_value_out_of_bounds_is_refused() -> None:
    with pytest.raises(ValidationError):
        Configuration().with_overrides(["preprocessing.smoothing=4"])


def test_a_saved_configuration_loads_back_the_same(tmp_path) -> None:
    configuration = Configuration(reason="a test").with_overrides([
        "tracking.max_missed_frames=2"
    ])

    assert (
        Configuration.load(configuration.save(tmp_path / "configuration.toml"))
        == configuration
    )
