# Open questions

**[question]** The author decides. An answer the agent gave while working
waits here for the author's word.

1. **Static figures**: **[sketch]** answered by the agent, 2026-11-12: the
   looks and maps are drawn with OpenCV, the distributions with matplotlib's
   Agg backend, the animation with Pillow; no browser, so kaleido and plotly
   are gone. To confirm.
2. **The agent's surface**: **[sketch]** answered by the agent, 2026-11-12: a
   command line, `mcv`, JSON on stdout and a folder per run, wrapped by
   `just` recipes and a skill. No service: an agent with a shell needs
   nothing more, and a run folder is the state. To confirm; a local service
   returns only if a window is wanted beside the agent.
3. **The units**: **[fact]** 2026-11-12, read off the burnt-in scale bars:
   0.323 µm per pixel for `342843` (10 µm in 31 px), 0.37 for `342655` (5 µm
   in 13.5 px); the other seven are to read, video by video.
4. **Downloads**: which sources of the catalogue may be fetched, once their
   licence is checked. VISEM-Tracking and the AnDi benchmark are the two
   with a truth and a CC-BY licence.
5. **The 2025 hypotheses**: the six tests stay in the library, no longer
   used by the report, which gives one verdict per track. Delete them, or
   keep them as the student's record?
