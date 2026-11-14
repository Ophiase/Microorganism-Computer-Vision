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

[group("self")]
[doc("The newest run folder of a source: just agent::last data/342843.avi")]
[script("bash")]
last source:
    set -euo pipefail
    key=$(uv run python -c 'import sys; from pathlib import Path; from microorganism_computer_vision.media.run_folder import source_key; print(source_key(Path(sys.argv[1])))' "$1")
    ls -d output/runs/"$key"/[0-9]*/ | sort | tail -1

[group("self")]
[doc("One frame of a GIF as a PNG, to look at it: just agent::frame output/runs/x/001/tracks.gif 20")]
[script("bash")]
frame gif index="0":
    set -euo pipefail
    out="${1%.gif}_frame_$2.png"
    uv run python -c 'import sys; from PIL import Image; g = Image.open(sys.argv[1]); g.seek(int(sys.argv[2])); g.convert("RGB").save(sys.argv[3])' "$1" "$2" "$out"
    echo "$out"

[group("self")]
[doc("A crop of a picture too large to read at once: just agent::crop sheet.png TOP BOTTOM LEFT RIGHT")]
[script("bash")]
crop picture top bottom left right:
    set -euo pipefail
    out="${1%.png}_crop.png"
    uv run python -c 'import sys, cv2; t, b, l, r = map(int, sys.argv[2:6]); cv2.imwrite(sys.argv[6], cv2.imread(sys.argv[1])[t:b, l:r])' "$@" "$out"
    echo "$out"

[group("self")]
[doc("Every review sheet of a run at once: suspects, misses, and a gallery per motion")]
[script("bash")]
review run:
    set -euo pipefail
    uv run mcv suspects "$1" > /dev/null
    uv run mcv misses "$1" > /dev/null
    for motion in stationary diffusive directed confined circular; do
        uv run mcv gallery "$1" "$motion" > /dev/null 2>&1 || true
    done
    ls "$1"/suspects.png "$1"/misses.png "$1"/gallery_*.png 2>/dev/null

[group("self")]
[doc("The shape of every document of ideas/ and docs/: orphans, bare, dense, unlinked")]
docs:
    @python3 .agents/checks/doc_shape.py

[group("self")]
[doc("Replace a text in a file, or fail loudly when it is absent: just agent::edit FILE OLD NEW")]
[script("bash")]
edit file old new:
    set -euo pipefail
    python3 -c 'import sys, pathlib; path, old, new = sys.argv[1:4]; text = pathlib.Path(path).read_text(); count = text.count(old); sys.exit(f"{path}: the anchor is absent, nothing replaced") if count == 0 else pathlib.Path(path).write_text(text.replace(old, new)); print(f"{path}: {count} replaced")' "$1" "$2" "$3"

[group("self")]
[doc("Stack a slow command for the background worker, 0 first: just agent::queue 5 uv run mcv run ...")]
[script("bash")]
queue priority +command:
    set -euo pipefail
    [[ $1 =~ ^[0-9]$ ]] || { echo "a priority from 0 to 9" >&2; exit 1; }
    mkdir -p output/queue/pending
    task="output/queue/pending/$1-$(date +%s%N).sh"
    shift
    printf '%q ' "$@" > "$task"
    echo "$task"

[group("self")]
[doc("Run the stacked commands niced, highest priority first, until none is left")]
[script("bash")]
worker:
    set -uo pipefail
    mkdir -p output/queue/{pending,done,failed}
    mkdir output/queue/lock 2>/dev/null || { echo "a worker already runs"; exit 0; }
    trap 'rmdir output/queue/lock' EXIT
    while task=$(ls output/queue/pending | sort | head -1) && [[ -n $task ]]; do
        mv "output/queue/pending/$task" "output/queue/$task"
        start=$(date +%s)
        if nice -n 19 bash "output/queue/$task" > "output/queue/${task%.sh}.log" 2>&1; then status=done; else status=failed; fi
        mv "output/queue/$task" "output/queue/$status/$task"
        mv "output/queue/${task%.sh}.log" "output/queue/$status/"
        printf '%s\t%s\t%ss\t%s\t%s\n' "$(date +%FT%T)" "$status" "$(($(date +%s) - start))" "$(cat "output/queue/$status/$task")" "$(tail -c 200 "output/queue/$status/${task%.sh}.log" | tr '\n\t' '  ')" >> output/queue/harvest.tsv
    done

[group("self")]
[doc("What the background gathered: the pending count, then the last harvests")]
[script("bash")]
harvest count="10":
    set -euo pipefail
    echo "pending: $(ls output/queue/pending 2>/dev/null | wc -l), running: $(ls output/queue/*.sh 2>/dev/null | wc -l)"
    tail -n "$1" output/queue/harvest.tsv 2>/dev/null || echo "nothing gathered yet"

[group("self")]
[doc("Queue clips of a long video, one configuration: just agent::windows v.mp4 c.toml 150 6 750 (size, count, every)")]
[script("bash")]
windows source config size count every priority="4":
    set -euo pipefail
    for ((window = 0; window < $4; window++)); do
        start=$((window * $5))
        just agent::queue "$6" uv run mcv run "$1" --config "$2" \
            --set clip.start=$start --set clip.stop=$((start + $3)) \
            --reason "window $((window + 1)) of $4, frames $start to $((start + $3))"
    done
