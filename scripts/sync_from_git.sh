#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "[$(date --iso-8601=seconds)] tracked files have local changes; skip automatic update" >&2
  exit 2
fi

git fetch --prune origin main
local_commit="$(git rev-parse HEAD)"
remote_commit="$(git rev-parse origin/main)"

if [[ "$local_commit" == "$remote_commit" ]]; then
  exit 0
fi

git merge --ff-only origin/main
echo "[$(date --iso-8601=seconds)] updated $local_commit -> $remote_commit"

