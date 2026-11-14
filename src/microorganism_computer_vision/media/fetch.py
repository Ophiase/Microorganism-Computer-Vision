from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import requests
import yaml
from pydantic import BaseModel, ConfigDict, Field

from microorganism_computer_vision.media.video import UsageError

TIMEOUT: Final = 120
CHUNK: Final = 1 << 16


class Download(BaseModel):
    """One file a source offers, where to get it, and how much disk it takes."""

    model_config = ConfigDict(frozen=True)

    file: str
    url: str
    megabytes: float


class Source(BaseModel):
    """An entry of the catalogue; the fields fetch needs, the rest kept as they are."""

    model_config = ConfigDict(extra="allow", frozen=True)

    id: str
    name: str
    truth: str = "none"
    licence: str = "to check"
    tags: tuple[str, ...] = ()
    folder: str | None = None
    downloads: tuple[Download, ...] = Field(default=())

    @property
    def has_truth(self) -> bool:
        return not self.truth.startswith("none")


@dataclass(frozen=True, slots=True)
class Planned:
    """One file to bring, and where it lands."""

    source: Source
    download: Download
    destination: Path


def load_catalogue(path: Path) -> tuple[Source, ...]:
    return tuple(
        Source.model_validate(entry) for entry in yaml.safe_load(path.read_text())
    )


def plan(
    sources: Sequence[Source],
    data: Path,
    names: Sequence[str] = (),
    tag: str | None = None,
    with_truth: bool = False,
    max_megabytes: float | None = None,
) -> tuple[Planned, ...]:
    """The files the filters select, smallest first, within the disk budget given."""
    unknown = set(names) - {source.id for source in sources}
    if unknown:
        raise UsageError(
            f"no source {sorted(unknown)} in the catalogue; see mcv sources"
        )
    chosen = [
        source
        for source in sources
        if source.downloads
        and (not names or source.id in names)
        and (tag is None or tag in source.tags)
        and (not with_truth or source.has_truth)
    ]
    files = sorted(
        (
            Planned(source, item, data / (source.folder or source.id) / item.file)
            for source in chosen
            for item in source.downloads
        ),
        key=lambda planned: planned.download.megabytes,
    )
    if max_megabytes is None:
        return tuple(files)
    kept, total = [], 0.0
    for planned in files:
        if total + planned.download.megabytes <= max_megabytes:
            kept.append(planned)
            total += planned.download.megabytes
    return tuple(kept)


def bring(planned: Planned) -> Path:
    """Download one file whole, or not at all: a partial or empty one is removed.

    A host that answers a script with a challenge page sends nothing and a 2xx;
    an empty file is that, never a video.
    """
    destination = planned.destination
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    headers = {"User-Agent": "motility-toolkit/0.8"}
    with requests.get(
        planned.download.url, stream=True, timeout=TIMEOUT, headers=headers
    ) as response:
        response.raise_for_status()
        with partial.open("wb") as file:
            for chunk in response.iter_content(chunk_size=CHUNK):
                file.write(chunk)
    if partial.stat().st_size == 0:
        partial.unlink()
        raise UsageError(
            f"{planned.download.url} sent an empty file: the host refused it"
        )
    return partial.replace(destination)
