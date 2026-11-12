set positional-arguments := true

import '.agents/portable/just/harness.just'

[group("code")]
[doc("The longest Python files, tests left out: green to 100 lines, yellow to 200.")]
[script("bash")]
longest count="30":
    set -euo pipefail
    [[ $1 =~ ^[0-9]+$ ]] || { echo "a count of files" >&2; exit 1; }
    git ls-files -- '*.py' ':!*tests/*' | while read -r file; do
        wc -l < "$file" | tr -d ' ' | sed "s|\$| $file|"
    done | sort -rn | head -n "$1" | while read -r lines file; do
        if ((lines > 200)); then colour=31; elif ((lines > 100)); then colour=33; else colour=32; fi
        printf '\e[%sm%5d\e[0m  %s\n' "$colour" "$lines" "$file"
    done

[group("motility")]
[doc("How an agent calibrates the toolkit: read it before the first run.")]
guide:
    @uv run mcv guide

[group("motility")]
[doc("Measure a video and suggest a configuration: just agent::probe data/342843.avi")]
probe source *arguments:
    uv run mcv probe "$@"

[group("motility")]
[doc("Run a configuration: just agent::run data/342843.avi --config c.toml --reason '...'")]
run source *arguments:
    uv run mcv run "$@"

[group("motility")]
[doc("The journal of the runs of a source, each with its reason.")]
history source:
    uv run mcv history "$1"

[group("motility")]
[doc("Score a configuration on the tuning scenes: just agent::bench --config c.toml")]
bench *arguments:
    uv run mcv bench "$@"

[group("motility")]
[doc("The blind baseline: just agent::search --budget 30 --seed 0")]
search *arguments:
    uv run mcv search "$@"

[group("motility")]
[doc("A synthetic twin with its truth: just agent::synth twin --like appearance.json")]
synth name *arguments:
    uv run mcv synth "$@"
