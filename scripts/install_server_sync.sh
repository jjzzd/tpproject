#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sync_script="$repo_dir/scripts/sync_from_git.sh"
log_file="$repo_dir/logs/git-sync.log"
mkdir -p "$repo_dir/logs"
chmod +x "$sync_script"

cron_line="* * * * * /usr/bin/flock -n '$repo_dir/.git/sync.lock' /bin/bash '$sync_script' >> '$log_file' 2>&1"
existing="$(crontab -l 2>/dev/null || true)"

if printf '%s\n' "$existing" | grep -Fq "$sync_script"; then
  echo "Automatic Git sync is already installed."
  exit 0
fi

{
  printf '%s\n' "$existing"
  printf '%s\n' "$cron_line"
} | awk 'NF && !seen[$0]++' | crontab -

echo "Installed one-minute automatic sync for: $repo_dir"
echo "Log file: $log_file"

