---
name: calibrate-motility
description: Read bacterial or cell motility from a researcher's microscope videos with this toolkit; probe the video, tune the filters, tracker and verdicts by looking and by scoring on a synthetic twin with a known truth, and report the motions with their numbers. Use when a person brings swimming-cell videos, Petri-dish or slide recordings, or asks how their cells move.
---

# Calibrate motility

`just agent::guide` prints the method; it is the source and matches the code.
Read it before the first run.

## The loop, in short

```sh
just agent::probe data/video.avi
just agent::run data/video.avi --config output/runs/video/probe/suggested.toml \
    --clip :80
just agent::run data/video.avi --config c.toml \
    --set detection.threshold=0.4 --reason "faint cells absent from the mask"
just agent::history data/video.avi
just agent::synth twin --like output/runs/video/probe/appearance.json
just agent::bench --scene output/synthetic/twin/scene.toml --config c.toml
just agent::search --budget 30
```

Open every PNG a command names: the looks are your eyes. Change one
parameter at a time, and say why.

## Answer

The motions and their counts, speeds in µm/s if the scale is known, α with
its interval, the final configuration with its reason, the score beside the
blind search's, and what the images cannot show. Never "done" without
numbers.
