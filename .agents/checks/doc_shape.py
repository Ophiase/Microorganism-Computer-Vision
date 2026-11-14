"""Whether each document of ideas/ and docs/ is one linked, drawn module."""

import re
import subprocess
import sys
from pathlib import Path
from typing import Final

ROOTS: Final = ("ideas/", "docs/")
GENERATED: Final = ("docs/study/", "docs/scenes/")
MODULES: Final = "ideas/model/"
BARE_LINES: Final = 60
DENSE_SECTIONS: Final = 7
DENSE_LINES: Final = 150
NAMES_IN_TABLE: Final = 3
LINK: Final = re.compile(r"\]\(([^)#]+)(#[^)]*)?\)")
PLAIN_NAME: Final = re.compile(r"^[A-Za-z][A-Za-z ,'-]{0,40}$")


def tracked_markdown() -> list[Path]:
    listing = subprocess.run(
        ["git", "ls-files", "*.md"], capture_output=True, text=True, check=True
    )
    return [Path(line) for line in listing.stdout.splitlines()]


def linked_targets(files: list[Path]) -> set[Path]:
    targets = set()
    for source in files:
        for match in LINK.finditer(source.read_text()):
            target = (source.parent / match.group(1)).resolve()
            targets.add(target)
            if target.is_dir():
                targets.add(target / "README.md")
    return targets


def in_scope(path: Path) -> bool:
    text = str(path)
    return text.startswith(ROOTS) and not text.startswith(GENERATED)


def prose_lines(text: str) -> list[str]:
    """The lines a reader reads as text: no code, no table, no blank."""
    lines, fenced = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
            continue
        if not fenced and line.strip() and not line.startswith("|"):
            lines.append(line)
    return lines


def unlinked_names(text: str) -> list[str]:
    """First cells of table rows that name something in plain text, headers out."""
    lines = text.splitlines()
    names = []
    for index, line in enumerate(lines):
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        following = lines[index + 1] if index + 1 < len(lines) else ""
        if following.startswith("|") and set(following) <= set("|-: "):
            continue
        first = line.strip("|").split("|")[0].strip().strip("*")
        if PLAIN_NAME.match(first):
            names.append(first)
    return names


def notes_for(path: Path, text: str) -> list[str]:
    notes = []
    drawn = "![" in text or "```mermaid" in text
    if str(path).startswith(MODULES) and not drawn and path.name != "README.md":
        notes.append(f"BARE {path}: a concept of the model without a picture: draw it")
    elif len(prose_lines(text)) > BARE_LINES and not drawn:
        notes.append(f"BARE {path}: {len(prose_lines(text))} lines of text, no picture: draw what it says")
    sections = len(re.findall(r"^## ", text, flags=re.MULTILINE))
    if sections > DENSE_SECTIONS and len(text.splitlines()) > DENSE_LINES:
        notes.append(f"DENSE {path}: {sections} sections: several concepts, make it a folder")
    names = unlinked_names(text)
    if len(names) >= NAMES_IN_TABLE:
        shown = ", ".join(names[:4])
        notes.append(f"UNLINKED {path}: table names in plain text ({shown}): link each to its file or section")
    return notes


def main(paths: list[str]) -> int:
    files = tracked_markdown()
    targets = linked_targets(files)
    checked = [Path(item) for item in paths] if paths else files
    orphans = [
        path for path in checked
        if in_scope(path) and path.exists() and path.resolve() not in targets
    ]
    for path in orphans:
        print(f"ORPHAN {path}: no document links it; link it from the one above first")
    for path in checked:
        if in_scope(path) and path.exists():
            for note in notes_for(path, path.read_text()):
                print(f"note: {note}")
    return 1 if orphans else 0


if __name__ == "__main__":
    sys.exit(main([item for item in sys.argv[1:] if item.endswith(".md")]))
