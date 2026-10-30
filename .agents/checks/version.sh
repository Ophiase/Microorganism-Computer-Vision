#!/usr/bin/env bash
# Usage: version.sh <file>...
# Reports a VERSION with no section in CHANGELOG.md, or that disagrees with
# CITATION.cff or pyproject.toml. Runs whatever files it is given.
set -uo pipefail

root=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
version=$(tr -d '[:space:]' <"$root/VERSION")

status=0
if ! grep -q "^## $version " "$root/CHANGELOG.md"; then
  printf 'CHANGELOG.md has no section for %s.\n' "$version"
  status=1
fi

if [[ -f $root/CITATION.cff ]]; then
  citation=$(awk '/^version:/ { print $2; exit }' "$root/CITATION.cff")
  if [[ $version != "$citation" ]]; then
    printf 'VERSION says %s, CITATION.cff says %s: bump both together.\n' "$version" "$citation"
    status=1
  fi
fi

if [[ -f $root/pyproject.toml ]]; then
  project=$(awk -F'"' '/^version = / { print $2; exit }' "$root/pyproject.toml")
  if [[ $version != "$project" ]]; then
    printf 'VERSION says %s, pyproject.toml says %s: bump both together.\n' "$version" "$project"
    status=1
  fi
fi

exit $status
