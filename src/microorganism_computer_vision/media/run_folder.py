import json
import tomllib
from pathlib import Path
from typing import Final

import tomli_w
from motility.review import Correction

from microorganism_computer_vision.configuration import Configuration
from microorganism_computer_vision.media.video import UsageError

CORRECTIONS: Final = "corrections.toml"


class RunFolder:
    """A written run, read back: what it read, how, and the judgements made on it."""

    def __init__(self, path: Path) -> None:
        if not (path / "summary.json").exists():
            raise UsageError(f"{path} is not a run folder: no summary.json")
        self.path = path

    @property
    def source(self) -> Path:
        source = json.loads((self.path / "summary.json").read_text())["source"]
        if source.startswith("scene:"):
            raise UsageError(f"{self.path} ran on a bench scene, not a file")
        return Path(source)

    @property
    def configuration(self) -> Configuration:
        return Configuration.load(self.path / "configuration.toml")

    def corrections(self) -> tuple[Correction, ...]:
        path = self.path / CORRECTIONS
        if not path.exists():
            return ()
        with path.open("rb") as file:
            entries = tomllib.load(file).get("correction", [])
        return tuple(Correction.model_validate(entry) for entry in entries)

    def append(self, correction: Correction) -> tuple[Correction, ...]:
        corrections = (*self.corrections(), correction)
        entries = [
            item.model_dump(mode="json", exclude_none=True) for item in corrections
        ]
        (self.path / CORRECTIONS).write_text(tomli_w.dumps({"correction": entries}))
        return corrections
