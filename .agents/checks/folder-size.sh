#!/usr/bin/env bash
# Usage: folder-size.sh <file>...
# Reports each folder of code holding more than seven entries, among the
# folders of the files given. See .agents/rules/layout.md.
set -uo pipefail

checks=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
limit=7
status=0
declare -A seen=()

for file in "$@"; do
  case $file in
    *.rs | *.js | *.py) ;;
    *) continue ;;
  esac
  folder=$(dirname "$file")
  [[ -n ${seen[$folder]:-} || ! -d $folder ]] && continue
  seen[$folder]=1
  "$checks/../portable/checks/is-ignored.sh" folder-size "$(realpath -m "$folder")" && continue
  count=$(find "$folder" -mindepth 1 -maxdepth 1 ! -name '.*' | wc -l)
  if ((count > limit)); then
    printf '%s holds %d entries, more than %d: split it into named concepts.\n' "$folder" "$count" "$limit"
    status=1
  fi
done

exit $status
