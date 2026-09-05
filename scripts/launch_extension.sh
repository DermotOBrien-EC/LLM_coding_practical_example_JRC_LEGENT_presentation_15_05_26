#!/bin/bash
# Extension A driver (DESIGN.md section 11): after the canary has been
# inspected by hand, run ext_waveA (the seventeen remaining L1 and L2 runs,
# six at a time) and then ext_waveB (the nine L3 runs, two at a time).
# Start it detached (nohup); progress and the terminal marker
# EXTENSION_DONE go to runs_2026_09/_logs/extension.log.
#
# Usage:  nohup bash scripts/launch_extension.sh > runs_2026_09/_logs/extension.log 2>&1 &
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$ROOT/runs_2026_09/_logs"
mkdir -p "$LOGS"
echo "ext_waveA start at $(date -u +%H:%M:%SZ)"
bash "$ROOT/scripts/launch_wave.sh" ext_waveA > "$LOGS/ext_waveA.launcher.log" 2>&1
RC_A=$?
echo "ext_waveA rc=$RC_A at $(date -u +%H:%M:%SZ)"
echo "ext_waveB start at $(date -u +%H:%M:%SZ)"
bash "$ROOT/scripts/launch_wave.sh" ext_waveB > "$LOGS/ext_waveB.launcher.log" 2>&1
RC_B=$?
echo "ext_waveB rc=$RC_B at $(date -u +%H:%M:%SZ)"
echo "EXTENSION_DONE waveA=$RC_A waveB=$RC_B at $(date -u +%H:%M:%SZ)"
