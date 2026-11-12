# Dependencies

**[fact]** 2026-11-12: `uv audit` finds no known vulnerability in the 35
packages; only `ruff` and `ty` are behind, by a patch.

| Package | Used for | Verdict |
| --- | --- | --- |
| numpy 2.5 | everything | keep |
| opencv-python-headless 5.0 | decode, Farneback, contours | keep; the headless build is enough, no `cv2` window is opened |
| scipy 1.18 | Hungarian matching, KS test | keep |
| numba 0.68 (removed 2026-11-12) | a 3 x 3 blur, a gradient, a distance matrix | **remove**: 196 MB with llvmlite, 14 s of compile per run, for three kernels that OpenCV, NumPy and `scipy.spatial` already provide |
| matplotlib 3.11 | the distributions, Agg backend | **kept** 2026-11-12: the only PNG path without a browser |
| plotly 7.1 | every figure | **removed** 2026-11-12, with kaleido |
| kaleido 1.5 | plotly to PNG | **removed** 2026-11-12: it needed Chrome |
| pillow 13 | the GIFs | keep |
| requests 2.34 | one download in `just extract` | could be the standard library; low priority |

The two removals take the environment from 772 MB to about 550 MB.

**[fact]** 2026-11-12: pydantic (the documented parameters), tomli-w (the
configurations an agent writes) and PyYAML (the catalogue) are added.
