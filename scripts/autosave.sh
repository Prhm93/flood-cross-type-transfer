#!/usr/bin/env bash
# autosave.sh - every 10 minutes, commit and push new code, logs, results, figures
# and new exp15 checkpoints. Never adds data/, the cache or the paper.
cd ~/projects/floodtransfer || exit 1
while true; do
  git add scripts/*.py scripts/*.sh reports/ results/*.json results/figures/ .gitignore 2>/dev/null
  git add -f results/exp15_*.pt 2>/dev/null
  if ! git diff --cached --quiet; then
    git commit -q -m "autosave $(date '+%Y-%m-%d %H:%M')"
    if git push -q origin main; then echo "$(date '+%H:%M') pushed"; else echo "$(date '+%H:%M') PUSH FAILED"; fi
  else
    echo "$(date '+%H:%M') nothing new"
  fi
  sleep 600
done
