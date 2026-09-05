#!/bin/bash
# Relaunch of wave 2 on 2026-09-05 (DESIGN.md section 9). Pair A: the single
# permitted resume of opus5_L3_r1 (scripts/resume_run.sh) beside a fresh
# fable51 L3 r1 (wave2_pairA_fable). Pair B, once both have ended: fable51 L3
# r2 + r3 (wave2_pairB) through the usage gate, polled every five minutes.
# Start it detached (nohup) from a shell whose TMPDIR is the real per-user
# temp directory, because the preserved seatbelt profile of opus5_L3_r1
# allows only that staging path. Progress and the terminal marker
# WAVE2_RELAUNCH_DONE go to runs_2026_09/_logs/wave2_relaunch.log.
#
# Usage:  nohup bash scripts/relaunch_wave2.sh > runs_2026_09/_logs/wave2_relaunch.log 2>&1 &
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$ROOT/runs_2026_09/_logs"
mkdir -p "$LOGS"
REASON="first attempt cut off by print mode's 600 s background-task wait ceiling while N-BEATS and PatchTST were training (harness limitation, not the model)"

echo "pair A start at $(date -u +%H:%M:%SZ)"
bash "$ROOT/scripts/resume_run.sh" opus5_L3_r1 "$REASON" > "$LOGS/opus5_L3_r1.resume.log" 2>&1 &
P_RESUME=$!
echo "resume opus5_L3_r1 pid $P_RESUME"
bash "$ROOT/scripts/wait_and_launch.sh" wave2_pairA_fable 300 > "$LOGS/wave2_pairA_fable.wait.log" 2>&1 &
P_A=$!
echo "wave2_pairA_fable waiter pid $P_A"
wait "$P_RESUME"; RC_R=$?
echo "resume opus5_L3_r1 rc=$RC_R at $(date -u +%H:%M:%SZ)"
wait "$P_A"; RC_A=$?
echo "wave2_pairA_fable rc=$RC_A at $(date -u +%H:%M:%SZ)"

echo "pair B start at $(date -u +%H:%M:%SZ)"
bash "$ROOT/scripts/wait_and_launch.sh" wave2_pairB 300 > "$LOGS/wave2_pairB.wait.log" 2>&1
RC_B=$?
echo "wave2_pairB rc=$RC_B at $(date -u +%H:%M:%SZ)"
echo "WAVE2_RELAUNCH_DONE resume=$RC_R pairA=$RC_A pairB=$RC_B at $(date -u +%H:%M:%SZ)"
