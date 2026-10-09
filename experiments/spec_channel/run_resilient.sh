#!/usr/bin/env bash
# Run runner.py until it finishes, restarting after outages or crashes; resume comes from the run's JSONL logs.
# Usage: ./run_resilient.sh <runner args...>   Env: RETRY_SLEEP (s, default 900), MAX_RESTARTS (default 30).
set -u
cd "$(dirname "$0")"
sleep_s=${RETRY_SLEEP:-900}
max=${MAX_RESTARTS:-30}
for ((i = 0; i <= max; i++)); do
  echo "[$(date -Is)] start attempt $i: runner.py $*"
  python -I runner.py "$@"
  rc=$?
  echo "[$(date -Is)] runner exited $rc"
  case $rc in
    0) exit 0 ;;
    3) echo "config mismatch, not retrying"; exit 3 ;;
  esac
  sleep "$sleep_s"
done
echo "[$(date -Is)] giving up after $max restarts"
exit 1
