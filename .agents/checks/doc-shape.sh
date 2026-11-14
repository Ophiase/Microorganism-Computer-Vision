#!/usr/bin/env bash
# Usage: doc-shape.sh <file>...
# Reports the documents of ideas/ and docs/ nothing links (blocking), and as
# notes those long without a picture, dense, or naming concepts unlinked.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
exec python3 .agents/checks/doc_shape.py "$@"
