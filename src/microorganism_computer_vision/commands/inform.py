import argparse
import json
from pathlib import Path
from typing import Final

import yaml
from motility.truth.synthetic import SceneParameters
from pydantic import BaseModel

from microorganism_computer_vision.configuration import Configuration
from microorganism_computer_vision.media.extract import process as download_study
from microorganism_computer_vision.media.folders import CATALOGUE, RUNS
from microorganism_computer_vision.media.run_folder import (
    HistoryEntry,
    Workspace,
    source_key,
)

GUIDE: Final = Path("docs/guide.md")


class GuideCommand:
    name = "guide"
    summary = "how an agent calibrates the toolkit: read this first"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """The guide takes no argument."""

    def execute(self, _arguments: argparse.Namespace, /) -> str:
        return GUIDE.read_text()


class SchemaCommand:
    name = "schema"
    summary = "every parameter with its description and bounds, as JSON Schema"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--scene", action="store_true", help="the synthetic scene's instead"
        )

    def execute(self, arguments: argparse.Namespace) -> str:
        model = SceneParameters if arguments.scene else Configuration
        return json.dumps(model.model_json_schema(), indent=2, ensure_ascii=False)


class Sources(BaseModel):
    count: int
    sources: tuple[dict[str, object], ...]


class SourcesCommand:
    name = "sources"
    summary = "the catalogue of motility videos, their truth, licence and trust"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--with-truth", action="store_true", help="only those with a truth"
        )

    def execute(self, arguments: argparse.Namespace) -> Sources:
        entries = yaml.safe_load(CATALOGUE.read_text())
        if arguments.with_truth:
            entries = [entry for entry in entries if _has_truth(entry)]
        return Sources(count=len(entries), sources=tuple(entries))


class History(BaseModel):
    source: str
    runs: tuple[HistoryEntry, ...]


class HistoryCommand:
    name = "history"
    summary = "the runs of a source, each with the reason it was made"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "source", type=Path, help="the video, as given to `mcv run`"
        )

    def execute(self, arguments: argparse.Namespace) -> History:
        stem = source_key(arguments.source)
        return History(source=stem, runs=Workspace(RUNS, stem).history())


class Fetched(BaseModel):
    videos: tuple[Path, ...]


class FetchCommand:
    name = "fetch"
    summary = "download the nine videos of the 2025 study into data/ (asks the network)"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        """The study's addresses are in settings/dataset.toml."""

    def execute(self, _arguments: argparse.Namespace, /) -> Fetched:
        return Fetched(videos=download_study())


def _has_truth(entry: dict[str, object]) -> bool:
    return not str(entry.get("truth", "none")).startswith("none")
