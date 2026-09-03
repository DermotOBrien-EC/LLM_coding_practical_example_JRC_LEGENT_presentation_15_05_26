#!/bin/bash
# Keep trying to launch a wave until its usage gate passes, then run it.
# Meant to be started detached (nohup) after the previous wave has finished,
# so the next wave starts by itself when the five-hour window allows.
#
# Usage:  bash scripts/wait_and_launch.sh <wave-name> [poll-seconds]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WAVE="${1:?wave name}"
POLL="${2:-300}"
LOGS="$ROOT/runs_2026_09/_logs"
mkdir -p "$LOGS"
while true; do
  bash "$ROOT/scripts/launch_wave.sh" "$WAVE" > "$LOGS/$WAVE.launcher.log" 2>&1
  rc=$?
  if [ "$rc" != 5 ]; then
    echo "WAIT_AND_LAUNCH $WAVE finished rc=$rc at $(date -u +%H:%M:%SZ)"
    exit "$rc"
  fi
  echo "gate closed at $(date -u +%H:%M:%SZ): $(tail -1 "$LOGS/$WAVE.launcher.log")"
  sleep "$POLL"
done
