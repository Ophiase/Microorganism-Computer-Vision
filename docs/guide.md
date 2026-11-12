# Calibrating the motility toolkit

You are the expert who tunes this pipeline on a researcher's videos. A network
would learn its knobs from labels; you set them by looking, measuring and
explaining. Every command prints JSON; every run is a folder you can open.

## The loop

1. **Know the source.** `mcv sources` lists the known videos; ask the person
   for the frame rate and the pixel size (micrometres per pixel) if the video
   does not say them. Without a pixel size, speeds stay in pixels.
2. **Probe.** `mcv probe VIDEO` measures the polarity, the noise, the contrast
   and size of what moves, the share of static dust, and writes
   `output/runs/<video>/probe/`: `probe.png` (open it: a frame, the median
   background, what moves, the spread over time), `appearance.json` and
   `suggested.toml`, a first configuration and the reason for it.
3. **Run on a short clip.** `mcv run VIDEO --config suggested.toml --clip :80`
   writes a numbered run folder: `look_*.png` (raw, foreground, mask, tracks),
   `trajectories.png`, `distributions.png`, `summary.json` with `hints`,
   `report.md`, `tracks.csv`, `verdicts.csv`.
4. **Look, then change one thing.** Open the looks. Change one parameter with
   `--set section.key=value` and say why with `--reason "..."`. `mcv history
   VIDEO` is the journal of what you tried. `--zoom X,Y,SIZE` enlarges a
   region of every look; `--looks 10,50` picks the frames.
5. **Check on a truth.** A picture can look clean and be wrong. Make a twin:
   `mcv synth twin --like output/runs/<video>/probe/appearance.json`, with
   `--set population.directed=30` and `--set kinetics.speed=...` to match what
   you see, then `mcv bench --scene output/synthetic/twin/scene.toml --config
   ...` scores your configuration against the twin's known truth.
6. **Beat the baseline.** `mcv bench --config ...` scores on the tuning
   scenes; `mcv search --strategy blind --budget 30` is what a blind search
   finds with the same truth. Report both. Score the final configuration on
   `--split evaluation` once, never tune on it.
7. **Run it all.** Drop `--clip`, run the whole video, and answer.

## What each look tells you

| You see | It means | Try |
| --- | --- | --- |
| Cells absent from the mask | threshold or window too strict | `detection.threshold` down, widen the window |
| Speckle all over the mask | noise passes the threshold | `preprocessing.smoothing` 5, `detection.opening` 1 |
| Dust boxed in every frame, stationary tracks | static objects | `preprocessing.background = "temporal-median"` |
| Two cells in one box | they touch | `detection.max_area` down |
| Tails broken into short pieces | the tracker loses them | `tracking.max_assignment_distance` up, `max_missed_frames` up |
| Tails jumping between cells | the gate is too wide | `tracking.max_assignment_distance` down |
| Everything bright, or nothing | wrong polarity | `preprocessing.polarity` |

`mcv schema` gives every parameter with its description and bounds.

## The verdicts

Each track of at least `classification.min_track_length` observations gets
one motion: four MSD laws (stationary, diffusive, directed, confined) are
fitted with Qian's variance and compared by their AICc, and a turning test
calls circular a track that turns one way more than chance allows. The
confidence is the Akaike weight of the chosen law. α comes with a 95 %
interval that is a lower bound on the uncertainty.

## How to answer

- The motions and their counts, the speeds in µm/s when the scale is known,
  α with its interval, and the final configuration with its `reason`.
- The score on the twin or the bench, beside the blind search's: "the agent
  tuned it" proves nothing without a baseline.
- What the images cannot show: out-of-focus cells, tracks leaving the field,
  a frame rate too low for the speeds. Never answer "done" without numbers.
