#!/usr/bin/env bash
# Usage: file-length.sh <file>...
# Reports each file of code longer than 200 lines before its test module:
# past that, it holds more than one concept. See .agents/rules/layout.md.
set -uo pipefail

checks=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
limit=200
status=0

for file in "$@"; do
  case $file in
    *.rs | *.js | *.py) ;;
    *) continue ;;
  esac
  [[ -f $file ]] || continue
  "$checks/../portable/checks/is-ignored.sh" file-length "$file" && continue
  count=$(awk '/^#\[cfg\(test\)\]/ { exit } { lines++ } END { print lines + 0 }' "$file")
  if ((count > limit)); then
    printf '%s has %d lines of code, more than %d: split it by concept.\n' "$file" "$count" "$limit"
    status=1
  fi
done

exit $status
