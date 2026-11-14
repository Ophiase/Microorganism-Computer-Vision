# CHANGELOG

Milestones of the project; the version is in `VERSION`.

## Unreleased

## 0.9.0 (2026-11-14)

- **Three cases on the home page**: Salmonella near a surface, Pseudomonas and
  E. coli under antibiotics, simulated swimmers whose tumbles follow their
  known run times.
- **`mcv validate` draws its figure**: how often each test says yes on cells
  of known behaviour, the false alarms in a red row, the test made for each
  behaviour boxed.
- **A background queue** for slow runs, and long videos read as windows.
- **Fixed**: `mcv interactions` tested the wrong tail for *slowed*, and failed
  on a group with no step outside the decor.

## 0.8.0 (2026-11-12)

- **An agent's toolkit**: `mcv` answers in JSON; `probe` measures a video and
  suggests a configuration, `run` writes a folder of looks, maps, a GIF, CSV
  tables and a report, `history` is the journal of the reasons, `synth`
  renders a twin with a known truth, `bench` and `search` score against it.
  The skill `calibrate-motility` and `mcv guide` tell an agent the method.
- **Measured**: on three scenes unseen while tuning, the agent scores 0.80,
  0.86 and 0.59 against 0.76, 0.81 and 0.22 for the best search with the
  truth; the study's video read in µm/s, 107 swimmers at 18.3 µm/s.
- **One verdict per track**: four MSD laws compared by AICc, a turning test,
  α with its interval, in place of six overlapping booleans.
- **Every parameter documented** in the schema, with its bounds; the 2025
  defaults are the 2025 code's, its foreground reproduced to 3e-7, a run in
  seconds instead of a minute, without numba, plotly or a browser.
- **Fixed**: the tracker reported every coasting prediction as observed; the
  study's download wrote nine empty files and called it a success.
- **The catalogue** lists twenty sources with their truth and licence.

## 0.7.0 (2026-11-12)

- **A change of direction, planned**: in a future PR, the project will become
  a toolkit an agent calibrates on a researcher's videos, instead of a
  pipeline tuned by hand or a network to train; the README and
  `ideas/vision.md` say why. The code is still the student version.
- **A catalogue of sources**: `catalogue/sources.yaml` lists the motility
  videos found, their truth, licence and trust.
- **The README says what the project is**: a student project of 2025, to be
  modernized with agents, in a warning at the top; the features it claimed
  without a measure give way to what it does today and its known limits.
- **The library apart from the application**: `libraries/motility/` holds
  the algorithms and reads no file; the application in `src/` runs the study.
  The source videos are `settings/dataset.toml`, no longer a list in the code.
- **A src layout**: the code is the package
  `src/microorganism_computer_vision/`, built by hatchling, with a
  `microorganism-computer-vision` command; no folder holds more than seven
  entries, so `folder-size.ignore` lists only the tooling.
- **The modernization is planned in `ideas/`**: a measured baseline, the
  dependency audit, the case for staying in Python, and the ordered steps.

## 0.6.0 (2026-11-12)

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
