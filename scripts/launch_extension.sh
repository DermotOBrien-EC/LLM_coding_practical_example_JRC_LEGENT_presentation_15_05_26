#!/bin/bash
# Extension driver (DESIGN.md sections 11 to 13): run wave A, then wave B,
# each through scripts/launch_wave.sh. Wave B follows wave A unless wave A
# completed no run at all (a route failure); the exit status is non-zero if
# any run failed. When launch_wave refuses on the Claude usage gate (exit
# 5) the driver waits GATE_WAIT seconds (default 900) and tries again, up
# to GATE_TRIES times (default 40), so a Claude-routed extension paces
# itself through the five-hour windows. Start it detached (nohup);
# progress and the terminal marker EXTENSION_DONE go to the log.
#
# Usage:  nohup bash scripts/launch_extension.sh [waveA] [waveB] > runs_2026_09/_logs/<name>.log 2>&1 &
#         (defaults: ext_waveA ext_waveB)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$ROOT/runs_2026_09/_logs"
MANIFEST="$ROOT/runs_2026_09/waves.json"
WAVE_A="${1:-ext_waveA}"
WAVE_B="${2:-ext_waveB}"
GATE_WAIT="${GATE_WAIT:-900}"
GATE_TRIES="${GATE_TRIES:-40}"
mkdir -p "$LOGS"

run_wave() {
  wave="$1"; tries=0
  while :; do
    bash "$ROOT/scripts/launch_wave.sh" "$wave" > "$LOGS/$wave.launcher.log" 2>&1
    rc=$?
    if [ "$rc" != 5 ]; then return "$rc"; fi
    tries=$((tries + 1))
    if [ "$tries" -ge "$GATE_TRIES" ]; then
      echo "$wave: usage gate still closed after $tries tries; giving up at $(date -u +%H:%M:%SZ)"
      return 5
    fi
    echo "$wave: usage gate closed ($(tail -1 "$LOGS/$wave.launcher.log" | cut -c1-120)); retry $tries in ${GATE_WAIT}s at $(date -u +%H:%M:%SZ)"
    sleep "$GATE_WAIT"
  done
}

completed_in() {
  # count the wave's runs whose own log carries a DONE line with exit=0
  python3 - "$MANIFEST" "$1" "$LOGS" <<'PY'
import json, re, sys, os
m, wave, logs = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
n = 0
for spec in m[wave]["runs"]:
    tag, level, rep = spec.split()
    p = os.path.join(logs, f"{tag}_{level}_r{rep}.log")
    if os.path.exists(p) and re.search(r"DONE .* exit=0", open(p, errors="replace").read()):
        n += 1
print(n)
PY
}

echo "$WAVE_A start at $(date -u +%H:%M:%SZ)"
run_wave "$WAVE_A"; RC_A=$?
COMPLETED_A="$(completed_in "$WAVE_A")"
echo "$WAVE_A rc=$RC_A completed=$COMPLETED_A at $(date -u +%H:%M:%SZ)"
if [ "$COMPLETED_A" = 0 ]; then
  echo "EXTENSION_DONE waveA=$RC_A waveB=not_started (no wave A run completed; route failure) at $(date -u +%H:%M:%SZ)"
  exit 1
fi
echo "$WAVE_B start at $(date -u +%H:%M:%SZ)"
run_wave "$WAVE_B"; RC_B=$?
echo "$WAVE_B rc=$RC_B at $(date -u +%H:%M:%SZ)"
echo "EXTENSION_DONE waveA=$RC_A waveB=$RC_B at $(date -u +%H:%M:%SZ)"
[ "$RC_A" = 0 ] && [ "$RC_B" = 0 ]
