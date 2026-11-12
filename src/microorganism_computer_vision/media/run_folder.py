import json
import tomllib
from pathlib import Path
from typing import Final

import tomli_w
from motility.review import Correction
from pydantic import BaseModel

from microorganism_computer_vision.configuration import Configuration
from microorganism_computer_vision.media.video import UsageError

CORRECTIONS: Final = "corrections.toml"
DIGITS: Final = 3


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


class HistoryEntry(BaseModel):
    """One run of a source, as the calibration journal shows it."""

    run: str
    reason: str
    tracks: int
    classified: int
    hints: int


def source_key(source: Path) -> str:
    """The name a source's runs are filed under: a file's stem, a folder's parent too.

    A folder of frames is often called `frames` or `images`; its parent names it.
    """
    if source.is_dir():
        return f"{source.resolve().parent.name}-{source.resolve().name}"
    return source.stem


class Workspace:
    """The runs of one source, numbered in the order they were made."""

    def __init__(self, root: Path, source: str) -> None:
        self._folder = root / source

    def next_folder(self) -> Path:
        numbers = [int(path.name) for path in self._runs()]
        return self._folder / str(max(numbers, default=0) + 1).zfill(DIGITS)

    def history(self) -> tuple[HistoryEntry, ...]:
        return tuple(self._entry(folder) for folder in self._runs())

    def _runs(self) -> list[Path]:
        if not self._folder.is_dir():
            return []
        return sorted(
            path
            for path in self._folder.iterdir()
            if path.name.isdigit() and (path / "summary.json").exists()
        )

    @staticmethod
    def _entry(folder: Path) -> HistoryEntry:
        summary = json.loads((folder / "summary.json").read_text())
        configuration = Configuration.load(folder / "configuration.toml")
        return HistoryEntry(
            run=str(folder),
            reason=configuration.reason,
            tracks=summary["tracks"],
            classified=summary["classified"],
            hints=len(summary["hints"]),
        )
