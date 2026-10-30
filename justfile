mod agent 'agent.just'
import '.agents/portable/just/harness.just'

set positional-arguments := true

default:
    @just --list --list-submodules

[group("user")]
[doc("Create the environment: Python 3.14 from .python-version, dependencies from uv.lock.")]
install:
    uv sync --all-groups

[group("user")]
[doc("Download the nine source videos into data/, which stays out of git.")]
extract:
    uv run -m script.main --task extract

[group("user")]
[doc("Run one pipeline task: just run transform|detection|render|analysis|synthetic")]
run task *args:
    uv run -m script.main --task {{ task }} {{ args }}

[group("dev")]
[doc("Format the code and apply the safe lint fixes.")]
fmt:
    uv run ruff format .
    uv run ruff check . --fix

[group("dev")]
[doc("Report the lint violations ruff cannot fix by itself.")]
lint:
    uv run ruff check .

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
[doc("Everything that must pass before declaring work done: lint, types, tests.")]
check: lint types test

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
    @printf 'pyproject  %s\n' "$(grep -m1 '^version' pyproject.toml | cut -d'\"' -f2)"
    @printf 'changelog  %s\n' "$(grep -m1 '^## ' CHANGELOG.md | cut -d' ' -f2)"
