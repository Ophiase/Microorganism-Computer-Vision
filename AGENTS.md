# AGENTS.md

Instructions for any coding agent in this repository. Claude Code and Cline
read this file; every other tool reads it too, it is the one source.

## What this project is

A research pipeline that reads bacterial motility from raw video: extract the
frames, compute the optical flow, transform each frame into a channel tensor,
track the cells with a Kalman filter, then test each trajectory against six
diffusion hypotheses and draw the figures.

**The direction** ([ideas/vision.md](ideas/vision.md)): a toolkit an agent
calibrates on a researcher's videos, instead of a network trained on labelled
data. A parameter is exposed and documented, never buried; the test cases are
the sources of [catalogue/sources.yaml](catalogue/sources.yaml), which an agent
keeps current: add what you learn, say who judged and when.

`README.md` says how to run it; the tests and the code say how it works.

## Language

English everywhere: code, comments, markdown, commits. The author may write to
you in French; answer in French, write files in English.

## Ideas

Where the project goes is in `ideas/`: the measured baseline, the dependency
audit, the language question and the ordered steps in
[ideas/roadmap.md](ideas/roadmap.md). A question for the author waits in
[ideas/open.md](ideas/open.md).

## Before working

1. Read [.agents/portable/rules/python.md](.agents/portable/rules/python.md)
   and [.agents/portable/rules/comments.md](.agents/portable/rules/comments.md):
   they decide what a file, a function and a comment may look like here.
2. Read [.agents/rules/layout.md](.agents/rules/layout.md) before adding a
   file: one concept per file, seven entries per folder, two hundred lines.
3. `.agents/local/projects.yaml` (gitignored) names the reference projects.
   Take their conventions, never their code.

## Toolchain

- **just**: the entry point. `just --list` shows the recipes; never call a
  tool directly when a recipe wraps it.
  - `just install`: `uv sync --all-groups`, Python 3.14 from `.python-version`.
  - `just check`: lint, types, tests. `just guards`: the file checks.
  - `just agent gate`: `check` then `guards`: the one command that declares
    work done.
- **uv**: packages and virtualenv. Never `pip install`, never create a venv
  by hand, never `uv add` without a lockfile in the commit.
- **ruff**: lint and format; the rules are in `ruff.toml`.
- **ty**: the type checker. A `# type: ignore` is a debt: fix the signature,
  or say why it cannot be fixed.
- **pytest**: `just test`. A test checks a property of the algorithm, never a
  number copied from one run; a demo that draws a window is not a test.

Read third-party sources through the in-repo venv
(`.venv/lib/python3.14/site-packages/...`), never through `~/.cache/uv/`.

## Layout

A uv workspace, built by hatchling, with **the library apart from the
application**:

- `libraries/motility/`: the library, imported as `motility`: what can be
  measured from a video. No file, no print, no figure, no setting; it takes
  arrays and returns values. Its tests are in `libraries/motility/tests/`.
  - `vision/`: the filters, the optical flow, the shapes.
  - `tracking/`: the boxes, the Kalman tracker, the trajectories.
  - `diffusion/`: the MSD, `models/` (the verdicts) and `hypotheses/`, the
    six tests of 2025.
  - `synthetic/`: scenes rendered into videos with their truth.
  - `evaluation/`: the tracking and classification scores against a truth.
  - `pipeline/`: the stages composed, the probe's suggestion, the twin.
- `src/microorganism_computer_vision/`: the application, the `mcv` command:
  it reads the videos and the settings, writes the run folders and draws.
  - `commands/`: one class per subcommand, entered through `__main__.py`.
  - `media/`: the reads, the tables, the run folders.
  - `reporting/`: the looks, the figures, the summary and the report.
  - `benchmark/`: the scenes of `settings/bench/` and the blind search.
- `settings/`: the data the application reads, never written in code: the
  source videos are `settings/dataset.toml`.
- `tests/`: the tests of the application.

A value that names the world (an address, a file id, a frame rate) is a
setting. A function that touches a file is the application's.

## Rules and checks

`.agents/portable/` is copied as is from the author's other repositories and
stays project-agnostic (`.agents/checks/portable.sh` guards it). Project rules
and checks go to `.agents/rules/` and `.agents/checks/`.
`.agents/checks/run.sh <check> <file>...` runs one; `just guards` runs them all
over the changed files.

`just agent install-hooks` once per clone, then every commit runs `just
guards`. The Kanon is [.agents/kanon.md](.agents/kanon.md); the `<>` in the
README is its sign.

## For agents: the command line

`mcv` (or `just mcv`, `just agent::<command>`) is the agent's surface: every
command prints JSON, every run is a folder of PNG looks, CSV tables and a
report. `mcv guide` is the method, printed from [docs/guide.md](docs/guide.md);
`mcv schema` lists every parameter with its description and bounds. The skill
[calibrate-motility](.claude/skills/calibrate-motility/SKILL.md) says when to
use it. A parameter added to the library is a pydantic `Field` with a
description: the schema, the guide's table and the agent all read it there.

## Done

A topic is done when `just agent gate` passes and the documentation it makes
wrong is updated in the same change: `README.md` for what a user runs,
`CHANGELOG.md` for what they notice, `VERSION` and `pyproject.toml` for the
number; `just version` shows the three side by side.

## Branches and commits

Work happens on `dev/<topic>`, one topic per branch, merged on `main` when the
gate passes. The agent commits as soon as the gate is green, and pushes only
when asked. The policy is in [.agents/project.yaml](.agents/project.yaml).

## Never

- A network call the person did not ask for: no telemetry, no font, script or
  style fetched from elsewhere. `just extract` is the one download, it asks.
- A hardcoded URL in code: a URL stays only with a `# hardcoded-url: <why>`
  directive on its line, and the author decides.
- A name of a company, client or professional project in a tracked file or a
  commit.
- Anything but the principles of the projects in `.agents/local/`: they are
  private. No name, no quotation, no code, no wording copied.
