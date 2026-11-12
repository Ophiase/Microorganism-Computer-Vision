import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

import tomli_w
from motility.pipeline import PipelineParameters
from pydantic import BaseModel, Field

from microorganism_computer_vision.media.video import Clip, UsageError


class Configuration(PipelineParameters):
    """A pipeline configuration as a researcher or an agent writes it, in TOML."""

    reason: str = Field(
        default="",
        description=(
            "Why this configuration: what was seen, what was changed. Every run"
            " keeps it, so the calibration reads as a journal."
        ),
    )
    clip: Clip = Field(default_factory=Clip)

    @classmethod
    def load(cls, path: Path | None) -> Self:
        if path is None:
            return cls()
        with path.open("rb") as file:
            return cls.model_validate(tomllib.load(file))

    def save(self, path: Path) -> Path:
        path.write_text(tomli_w.dumps(self.model_dump(mode="json", exclude_none=True)))
        return path

    def with_overrides(self, assignments: Sequence[str]) -> Self:
        return override(self, assignments)


def override[Model: BaseModel](model: Model, assignments: Sequence[str]) -> Model:
    """Apply `section.key=value` assignments, each value read as TOML or as text."""
    values = model.model_dump(mode="json")
    for assignment in assignments:
        key, separator, raw = assignment.partition("=")
        if not separator:
            raise UsageError(f"{assignment!r} is not section.key=value")
        *sections, name = key.strip().split(".")
        target = values
        for section in sections:
            target = target.get(section)
            if not isinstance(target, dict):
                raise UsageError(f"{key!r}: no section {section!r}")
        if name not in target:
            raise UsageError(f"{key!r}: no parameter {name!r}, see `mcv schema`")
        target[name] = _scalar(raw.strip())
    return type(model).model_validate(values)


def _scalar(raw: str) -> Any:
    try:
        return tomllib.loads(f"value = {raw}")["value"]
    except tomllib.TOMLDecodeError:
        return raw
