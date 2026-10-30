# CHANGELOG

Milestones of the project; the version is in `VERSION`.

## Unreleased

## 0.1.0 — 2026-10-30

- **The toolchain is the author's standard**: `justfile` plus `uv`, Python
  3.14 pinned in `.python-version`, dependencies in `pyproject.toml` and
  `uv.lock`; `make` and `requirements.txt` are gone.
- **Ruff and ty gate the code**: `just check` runs the lint, the types and the
  tests; the module imports, the dead code and the untyped signatures they
  reported are fixed.
- **The agent layer arrives**: `AGENTS.md`, `.agents/` with the portable
  rules and checks, the Kanon and its `<>` sign, `just agent gate`.
- **The tests learn pytest**: the old `unittest` file becomes property tests
  on the filters and the trajectories, run by `just test`.
- **The `unittest` demos leave the test folder**: they drew a window, so they
  were screenshots of one run, not tests.
