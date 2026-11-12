# Roadmap

**[sketch]** The ordered steps. A step is done when `just agent gate` passes
and its acceptance sentence is true. The optimization steps run on the
desktop, not the laptop: re-measure [baseline](baseline.md) there first.

## How to resume

1. Read [AGENTS.md](../AGENTS.md), [ideas/README.md](README.md), this file.
2. `git switch dev/modernization`, `just install`, `just agent gate`.
3. One step per `dev/<topic>` branch, merged on `main` when it passes.

## Step 1 — remove the waste

Drop numba and matplotlib: the blur through `cv2.filter2D`, normalized by the
filtered ones at the border; the gradient through NumPy; the cost matrix
through `scipy.spatial.distance.cdist`. Float32 throughout; the transform
writes only the channels a later stage reads.

*Acceptance*: same boxes on the reference video, `transform` under 2 s,
tensor under 500 MB.

## Step 2 — the pipeline runs end to end

Fix the analysis without Chrome ([open](open.md)); store the tracks as a
plain table — track, frame, x, y, width, height — instead of a pickle; one
test runs `transform`, `detection` and `analysis` on a few synthetic frames.

*Acceptance*: `just run analysis` passes on a fresh clone without Chrome.

## Step 3 — the package [done]

**[fact]** 2026-11-12: `src/microorganism_computer_vision/`, built by
hatchling; `common.py` is `pipeline/paths.py`; `logic/` is split into
`vision/`, `tracking/` and `diffusion/`. Then the library
`libraries/motility/` apart from the application, in a uv workspace, and the
dataset in `settings/`.

*Acceptance*: `folder-size.ignore` lists only `.agents/`.

## Step 4 — the truth benchmark

[science](science.md) step 1: synthetic videos with a known truth, the
tracking and classification scores, the defaults as the first baseline.

*Acceptance*: `just bench` writes the scores of the default configuration.

## Step 5 — the agent's surface

Every parameter of the pipeline in one settings file; one command runs a
configuration on a source and returns the overlays and the scores as files
an agent reads ([calibration](calibration.md), [open](open.md) question 2).

*Acceptance*: an agent changes a threshold and sees its effect without
touching code.

## Step 6 — the agentic calibration, measured

The loop of [calibration](calibration.md) against the defaults and a blind
search, on the tuning and evaluation split of the sources.

## Step 7 — model selection

[science](science.md) step 2.
