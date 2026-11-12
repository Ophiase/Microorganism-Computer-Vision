# Contributing

Read [AGENTS.md](AGENTS.md) first: it says how a file, a function, a comment
and a test are expected to look here, and it applies to humans the same way.

## Setup

```bash
just install       # Python 3.14 and the dependencies of uv.lock
just extract       # the nine source videos, into data/, out of git
```

## Before opening a change

```bash
just agent gate    # lint, types, tests, then the file checks
```

A change is ready when the gate is green and the documentation it makes wrong
is updated in the same commit: [README.md](README.md) for what a user runs,
[CHANGELOG.md](CHANGELOG.md) for what they notice.

## Version

One topic, one section in [CHANGELOG.md](CHANGELOG.md), one number in
[VERSION](VERSION), `pyproject.toml` and [CITATION.cff](CITATION.cff).
`just version` shows the three side by side.
