# Security

This project has no server, no account and no telemetry: it reads local videos
and writes local figures. The one network call is `just extract`, which
downloads the source dataset when the author asks for it.

## Reporting

Open an issue, or write to the author through the repository page
<https://github.com/Ophiase/Microorganism-Computer-Vision>. Do not open a
public issue for a flaw that would let a downloaded video run code: the
pipeline reads untrusted video files through OpenCV.

## Supported versions

The latest release only. The project targets Python 3.14 and the dependency
versions pinned in `uv.lock`.
