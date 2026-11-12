# AGENTS.md

Instructions for any coding agent in this repository. Claude Code and Cline
read this file; every other tool reads it too, it is the one source.

## What this project is

A research pipeline that reads bacterial motility from raw video: extract the
frames, compute the optical flow, transform each frame into a fourteen-channel
tensor, track the cells with a Kalman filter, then test each trajectory against
six diffusion hypotheses and draw the figures.

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

- **just** — the entry point. `just --list` shows the recipes; never call a
  tool directly when a recipe wraps it.
  - `just install` — `uv sync --all-groups`, Python 3.14 from `.python-version`.
  - `just check` — lint, types, tests. `just guards` — the file checks.
  - `just agent gate` — `check` then `guards`: the one command that declares
    work done.
- **uv** — packages and virtualenv. Never `pip install`, never create a venv
  by hand, never `uv add` without a lockfile in the commit.
- **ruff** — lint and format; the rules are in `ruff.toml`.
- **ty** — the type checker. A `# type: ignore` is a debt: fix the signature,
  or say why it cannot be fixed.
- **pytest** — `just test`. A test checks a property of the algorithm, never a
  number copied from one run; a demo that draws a window is not a test.

Read third-party sources through the in-repo venv
(`.venv/lib/python3.14/site-packages/...`), never through `~/.cache/uv/`.

## Layout

Four concepts, one folder each; the modules import from the repository root,
which `pyproject.toml` puts on `sys.path`:

- `logic/` — the algorithms: filters, kernels, optical flow, shapes, the
  Kalman tracker, the trajectories and `logic/diffusion/` with the six tests.
- `script/` — the pipeline tasks, one file each, entered through
  `script/main.py`.
- `visualization/` — the Plotly and matplotlib figures.
- `tests/` — the unit tests.

`common.py` holds the paths and the defaults every task reads. It moves to a
package in the next pass: until then, import it, do not copy its values.

## Rules and checks

`.agents/portable/` is copied as is from the author's other repositories and
stays project-agnostic (`.agents/checks/portable.sh` guards it). Project rules
and checks go to `.agents/rules/` and `.agents/checks/`.
`.agents/checks/run.sh <check> <file>...` runs one; `just guards` runs them all
over the changed files.

`just agent install-hooks` once per clone, then every commit runs `just
guards`. The Kanon is [.agents/kanon.md](.agents/kanon.md); the `<>` in the
README is its sign.

## Done

A topic is done when `just agent gate` passes and the documentation it makes
wrong is updated in the same change: `README.md` for what a user runs,
`CHANGELOG.md` for what they notice, `VERSION` and `pyproject.toml` for the
number — `just version` shows the three side by side.

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
