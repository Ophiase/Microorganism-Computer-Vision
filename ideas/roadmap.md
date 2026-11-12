# Roadmap

**[sketch]** The ordered steps. A step is done when `just agent gate` passes
and its acceptance sentence is true. The optimization steps run on the
desktop, not the laptop: re-measure [baseline](baseline.md) there first.

## How to resume

1. Read [AGENTS.md](../AGENTS.md), [ideas/README.md](README.md), this file.
2. `git switch dev/modernization`, `just install`, `just agent gate`.
3. One step per `dev/<topic>` branch, merged on `main` when it passes.

## The plan of `dev/agentic`

**[defined]** 2026-11-12: the branch that turns the student pipeline into the
agent's toolkit, one commit per box, the gate green at each.

1. [x] The library takes documented parameters: pydantic models for each stage,
   OpenCV and NumPy in place of numba, the tracker's bookkeeping fixed, a track
   table, synthetic scenes rendered with their truth, tracking and
   classification scores, a model selection in place of the six booleans.
2. [x] The agent's command line: `mcv guide`, `schema`, `probe`, `run`,
   `synth`, `bench`, `search`, `history`, every answer in JSON, every run a
   folder of PNG looks, a table of tracks and a report.
3. [x] The bench: scenes in `settings/bench/`, a tuning and an evaluation
   split, `just bench`.
4. [x] The catalogue: every source found, its truth, licence and trust.
5. [x] The skill, `agent.just` recipes, the guide the command prints.
6. [x] The calibration measured: the defaults, a blind search and the agent,
   on the bench and on the study's videos.
7. [x] The release: README, CHANGELOG, 0.8.0.

## The plan of `dev/skeleton`

**[defined]** 2026-11-12, asked by the author: a track is a point today; an
elongated body is a chain of points, and its posture carries what a point
cannot: the wave along a worm, the axis of a rod.

1. [x] The overlays are seen: drawn after the downscale, on an explicit
   palette, a box per cell in its verdict's colour.
2. [ ] `truth/`: `synthetic/` and `evaluation/` under one concept, which
   makes room for `posture/`.
3. [ ] The centreline: the geodesic path through a cell's mask, centred by
   its distance transform, resampled to N points equally spaced, the head
   kept from frame to frame and set by the direction of travel.
4. [ ] Synthetic worms: a travelling wave of curvature, rendered with its
   true skeleton, frequency and direction; a skeleton score.
5. [ ] The posture: tangent angles, the kymograph of curvature, the
   eigenworms, the wave's frequency, wavelength and direction per track.
6. [ ] In `mcv`: a `posture` section, `skeletons.csv`, skeletons in the
   looks and the GIF, a kymograph; measured on synthetic worms, and a real
   worm video if one can be fetched.

## Step 1: remove the waste [done]

**[fact]** 2026-11-12: numba and the channel tensor are gone; the same
foreground as 2025 to 3e-7, a run of 85 frames in 0.3 s for the stages and
4.5 s with every picture written, 0.8 GB of memory.

Drop numba and matplotlib: the blur through `cv2.filter2D`, normalized by the
filtered ones at the border; the gradient through NumPy; the cost matrix
through `scipy.spatial.distance.cdist`. Float32 throughout; the transform
writes only the channels a later stage reads.

*Acceptance*: same boxes on the reference video, `transform` under 2 s,
tensor under 500 MB.

## Step 2: the pipeline runs end to end [done]

**[fact]** 2026-11-12: no browser needed; tracks are `tracks.csv`;
`tests/test_end_to_end.py` runs synth, probe, run and bench.

Fix the analysis without Chrome ([open](open.md)); store the tracks as a
plain table (track, frame, x, y, width, height) instead of a pickle; one
test runs `transform`, `detection` and `analysis` on a few synthetic frames.

*Acceptance*: `just run analysis` passes on a fresh clone without Chrome.

## Step 3: the package [done]

**[fact]** 2026-11-12: `src/microorganism_computer_vision/`, built by
hatchling; `common.py` is `pipeline/paths.py`; `logic/` is split into
`vision/`, `tracking/` and `diffusion/`. Then the library
`libraries/motility/` apart from the application, in a uv workspace, and the
dataset in `settings/`.

*Acceptance*: `folder-size.ignore` lists only `.agents/`.

## Step 4: the truth benchmark [done]

**[fact]** 2026-11-12: `settings/bench/`, three scenes for tuning and their
twins for evaluation; `just bench`.

[science](science.md) step 1: synthetic videos with a known truth, the
tracking and classification scores, the defaults as the first baseline.

*Acceptance*: `just bench` writes the scores of the default configuration.

## Step 5: the agent's surface [done]

**[fact]** 2026-11-12: `mcv`, its guide and its skill.

Every parameter of the pipeline in one settings file; one command runs a
configuration on a source and returns the overlays and the scores as files
an agent reads ([calibration](calibration.md), [open](open.md) question 2).

*Acceptance*: an agent changes a threshold and sees its effect without
touching code.

## Step 6: the agentic calibration, measured [done]

**[fact]** 2026-11-12: [docs/benchmark/calibration.md](../docs/benchmark/calibration.md).

The loop of [calibration](calibration.md) against the defaults and a blind
search, on the tuning and evaluation split of the sources.

## Step 7: model selection [done]

**[fact]** 2026-11-12: four MSD laws by AICc on Qian's variance, a turning
test, α with an interval. Next: test it on the AnDi benchmark.

[science](science.md) step 2.
