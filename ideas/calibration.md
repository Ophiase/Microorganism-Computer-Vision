# Agentic calibration

**[sketch]** How an agent tunes the pipeline, and how it is judged.

## The loop

1. The agent reads a source: a few frames, the frame rate, the pixel size.
2. It proposes a configuration: filters, detection thresholds, the
   tracker's noise and gating distance, the hypotheses to test.
3. The toolkit runs it and returns what an expert would look at: overlays of
   the boxes on frames, the number and length of the tracks, the speed
   distribution.
4. The agent judges, changes one thing, and repeats; each step is logged
   with its reason.

The configuration is a settings file, never code: the agent's result is a
file a researcher can read, rerun and change.

## What it is compared with

**[caution]** Without baselines, "the agent tuned it" proves nothing. The
agent runs them itself, on the same sources, as part of the work.

- **The defaults**: the configuration of the original study.
- **A blind search**: random or Optuna search over the same parameters,
  given the same budget of runs and the truth as its score.
- **A tool built for it**: TrackMate or trackpy with their usual settings
  ([prior art](science.md)).

The agent's edge over a blind search is not in tuning numbers: it is that it
can **look** when no truth exists, **change the structure** (add a filter,
drop a hypothesis), and **explain**. The benchmark must show it on the
sources with a truth, before it is trusted on the others.

## Where the cases come from

The agent builds its own test set: it searches the web for motility videos
(species, microscopes, frame rates), reads the papers they come with, and
records each in [the catalogue](../catalogue/sources.yaml). Labelled tracks
are rare in public; where none exist, the agent can annotate a few tracks by
hand on a frame, and the synthetic videos give the rest of the truth.

## The traps

- **[caution]** An agent judging overlays by eye can prefer what looks
  clean over what is true. The score on the truth decides, not the picture.
- **[caution]** Tuning and scoring on the same videos measures memory. The
  sources split into tuning and evaluation sets, as in the point-cloud
  benchmark.
