# CHANGELOG

Milestones of the project; the version is in `VERSION`.

## Unreleased

- **The README says what the project is**: a student project of 2025,
  revisited with agents, in a warning at the top.
- **The library apart from the application**: `libraries/motility/` holds
  the algorithms and reads no file; the application in `src/` runs the study.
  The source videos are `settings/dataset.toml`, no longer a list in the code.
- **A src layout**: the code is the package
  `src/microorganism_computer_vision/`, built by hatchling, with a
  `microorganism-computer-vision` command; no folder holds more than seven
  entries, so `folder-size.ignore` lists only the tooling.
- **The modernization is planned in `ideas/`**: a measured baseline, the
  dependency audit, the case for staying in Python, and the ordered steps.

## 0.6.0 — 2026-11-12

The tags `v0.1` to `v0.5` are the original project; this release
continues their numbering.

- **The toolchain is the author's standard**: `justfile` plus `uv`, Python
  3.14 pinned in `.python-version`, dependencies in `pyproject.toml` and
  `uv.lock`; `make` and `requirements.txt` are gone.
- **Ruff and ty gate the code**: `just check` runs the lint, the types and the
  tests; the module imports, the dead code and the untyped signatures they
  reported are fixed.
- **The agent layer arrives**: `AGENTS.md`, `.agents/` with the portable
  rules and checks, the Kanon and its `<>` sign, `just agent gate`; the
  guard settings hold no local path, and `.claude/CLAUDE.md` points Claude
  Code at `AGENTS.md`.
- **The gate also checks the format and the markdown**: `just check` adds
  `ruff format --check` and `rumdl`.
- **The tests learn pytest**: the old `unittest` file becomes property tests
  on the filters and the trajectories, run by `just test`.
- **The `unittest` demos leave the test folder**: they drew a window, so they
  were screenshots of one run, not tests.
