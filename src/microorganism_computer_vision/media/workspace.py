import json
from pathlib import Path
from typing import Final

from pydantic import BaseModel

from microorganism_computer_vision.configuration import Configuration

DIGITS: Final = 3


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
