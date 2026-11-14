mod agent 'agent.just'
import '.agents/portable/just/harness.just'

set positional-arguments := true

default:
    @just --list # --list-submodules

[group("user")]
[doc("Create the environment: Python 3.14 from .python-version, dependencies from uv.lock.")]
install:
    uv sync --all-groups

[group("user")]
[doc("Download the nine source videos of the 2025 study into data/, out of git.")]
extract:
    uv run mcv fetch

[group("user")]
[doc("The toolkit's command line: just mcv probe|run|bench|search|synth|guide ...")]
mcv *arguments:
    uv run mcv "$@"

[group("user")]
[doc("Score the default configuration on the tuning and evaluation scenes.")]
bench:
    uv run mcv bench --config settings/default.toml --split tuning
    uv run mcv bench --config settings/default.toml --split evaluation

[group("dev")]
[doc("Format the code and apply the safe lint fixes.")]
fmt:
    uv run ruff format .
    uv run ruff check . --fix

[group("dev")]
[doc("Report the lint and format violations without changing the code.")]
lint:
    uv run ruff check .
    uv run ruff format --check .

[group("dev")]
[doc("Type-check the modules and the tests.")]
types:
    uv run ty check .

[group("dev")]
[doc("Lint and format the markdown. Needs rumdl.")]
rumdl:
    rumdl check .

[group("dev")]
rumdl-fmt:
    rumdl fmt .

[group("test")]
[doc("Run the unit tests.")]
test:
    uv run pytest -q

[group("gate")]
[doc("Everything that must pass before declaring work done: lint, types, tests, markdown.")]
check: lint types test rumdl

[group("gate")]
[doc("Run the project's and the portable checks over every changed file.")]
[script("bash")]
guards:
    status=0
    for check in markdown french comment-slop hardcoded-url portable config-format yaml-extension file-length folder-size version; do
        just _guard "$check" || status=1
    done
    exit $status

[group("gate")]
_guard check:
    @.agents/portable/checks/changed-files.sh | xargs -r -d '\n' .agents/checks/run.sh {{ check }}

[group("dev")]
[doc("Lock the dependencies again, newest allowed versions.")]
update:
    uv lock --upgrade

[group("dev")]
[doc("VERSION, pyproject.toml and the newest CHANGELOG section, side by side.")]
version:
    @printf 'VERSION    %s\n' "$(cat VERSION)"
    @printf 'pyproject  %s\n' "$(awk -F'\"' '/^version = / { print $2; exit }' pyproject.toml)"
    @printf 'changelog  %s\n' "$(awk '/^## [0-9]/ { print $2; exit }' CHANGELOG.md)"

[group("planned")]
[doc("Planned: a mask of the static objects the agent sees and edits. docs/tools.md#mask")]
mask source:
    @just _planned "an editable mask of the static objects" docs/tools.md#mask

[group("planned")]
[doc("Planned: the body axis of each cell, from its shape. ideas/model/trajectory.md")]
axis source:
    @just _planned "the body axis of each cell" ideas/model/trajectory.md

[group("planned")]
[doc("Planned: the skeleton of elongated cells, and their gait. ideas/model/models.md")]
skeleton source:
    @just _planned "the skeleton and the gait" ideas/model/models.md

[group("user")]
[doc("Each hypothesis test's false positives and power on ten synthetic behaviours.")]
validate:
    uv run mcv validate

[group("planned")]
[doc("Planned: the tests of the decor and the others, by surrogates. ideas/model/population.md")]
interactions run:
    @just _planned "the hypotheses on the decor and the neighbours" ideas/model/population.md

[group("planned")]
[doc("Planned: the gallery of the study's videos and of the scenes. docs/study/README.md")]
gallery:
    @just _planned "the galleries, written again from the runs" docs/study/README.md

[group("docs")]
[doc("Draw the figures of ideas/model/ from the library, into docs/figures/model/.")]
figures:
    PYTHONPATH=scripts uv run python -m illustrate

_planned what where:
    @printf 'not built yet: %s\nthe sub-problem is stated in %s\n' "$1" "$2" >&2
    @exit 1
