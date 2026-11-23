#!/usr/bin/env bash
# Fetch safely by default. --apply performs a guarded, fast-forward-only update.
set -euo pipefail

usage() {
  printf 'Usage: bash scripts/safe_update.sh [--fetch|--apply]\n' >&2
  printf '  --fetch  Download remote commits without changing local files (default).\n' >&2
  printf '  --apply  Update tracked code only if no existing local files are at risk.\n' >&2
}

mode="${1:---fetch}"
if [[ $# -gt 1 || ( "$mode" != "--fetch" && "$mode" != "--apply" ) ]]; then
  usage
  exit 2
fi

if ! root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  echo 'Run this script inside a Git clone.' >&2
  exit 1
fi
cd "$root"

git fetch --tags origin main
target="$(git rev-parse --verify FETCH_HEAD)"

if [[ "$mode" == "--fetch" ]]; then
  echo
  echo 'Fetch completed. No working files were changed.'
  echo 'Review the incoming updates with: git diff --stat HEAD FETCH_HEAD'
  echo 'To attempt a guarded fast-forward update: bash scripts/safe_update.sh --apply'
  exit 0
fi

if ! git merge-base --is-ancestor HEAD "$target"; then
  echo 'Update refused: local history cannot fast-forward to origin/main.' >&2
  exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo 'Update refused: tracked local files have uncommitted changes.' >&2
  echo 'Commit, or back them up and handle them explicitly before updating.' >&2
  exit 1
fi

unsafe=0
while IFS= read -r -d '' path; do
  if [[ -e "$path" || -L "$path" ]]; then
    printf 'Would remove existing local file: %q\n' "$path" >&2
    unsafe=1
  fi
done < <(git diff --no-renames --name-only --diff-filter=D -z HEAD "$target")

while IFS= read -r -d '' path; do
  if [[ -e "$path" || -L "$path" ]] &&
     ! git ls-files --error-unmatch -- "$path" >/dev/null 2>&1; then
    printf 'Would conflict with existing untracked/ignored file: %q\n' "$path" >&2
    unsafe=1
  fi
done < <(git diff --no-renames --name-only --diff-filter=ACMRT -z HEAD "$target")

if [[ "$unsafe" -ne 0 ]]; then
  echo 'Update refused: back up the named files outside this repository first.' >&2
  echo 'A .gitignore cannot protect files previously tracked by Git.' >&2
  exit 1
fi

git merge --ff-only "$target"
echo 'Safe fast-forward complete; existing ignored and untracked files were preserved.'
