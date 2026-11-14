# Top-down

Work as computational thinking does: state the problem at the top, name each
sub-problem with an empty placeholder, dive into one only when there is the
attention for it.

1. **The top first.** The README, then `ideas/`, say what the work is for. A
   new concept gets its link there before its file exists.
2. **A placeholder for each sub-problem**, made at once:
   - a markdown file with its headings and nothing under them;
   - a `just` recipe in the `planned` group that fails with "not built yet"
     and the path where its sub-problem is stated;
   - a section of `docs/tools.md` tagged **[question]** for a tool to come.
3. **Then dive**, one sub-problem at a time, filling the placeholder.
4. **Formalise before code.** The space of the problem, its models and their
   diagrams come before the implementation: a test that fails has a reason,
   and it shows on paper first.

A placeholder that stays empty is a decision deferred, visible to anyone who
lists the recipes or reads the map. Work written bottom-up is invisible until
it is finished, and lost if it is not.
