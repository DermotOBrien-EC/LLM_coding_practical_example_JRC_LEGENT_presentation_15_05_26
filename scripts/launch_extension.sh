#!/bin/bash
# Extension driver (DESIGN.md sections 11 to 13): run the named waves in
# order, each through scripts/launch_wave.sh. Every wave name is validated
# against the manifest before anything starts. The second and later waves
# run only if the first wave completed at least one run (otherwise the
# route failed and the driver stops); the exit status is non-zero if any
# wave failed. When launch_wave refuses on the Claude usage gate (exit 5)
# the driver waits GATE_WAIT seconds (default 900) and tries again, up to
# GATE_TRIES times (default 40): a bounded pre-wave retry, not a guarantee
# against exhaustion during a wave. Start it detached (nohup); progress and
# the terminal marker EXTENSION_DONE go to the log.
#
# Usage:  nohup bash scripts/launch_extension.sh <wave> [<wave> ...] > runs_2026_09/_logs/<name>.log 2>&1 &
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$ROOT/runs_2026_09/_logs"
MANIFEST="$ROOT/runs_2026_09/waves.json"
GATE_WAIT="${GATE_WAIT:-900}"
GATE_TRIES="${GATE_TRIES:-40}"
mkdir -p "$LOGS"
if [ "$#" -lt 1 ]; then echo "usage: $0 <wave> [<wave> ...]" >&2; exit 2; fi

for wave in "$@"; do
  python3 - "$MANIFEST" "$wave" <<'PY' || { echo "unknown wave '$wave' in $MANIFEST" >&2; exit 2; }
import json, sys
m = json.load(open(sys.argv[1]))
w = m.get(sys.argv[2])
sys.exit(0 if isinstance(w, dict) and isinstance(w.get("runs"), list) and w["runs"] else 1)
PY
done

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
  # written after this driver started (older logs of the same name are not
  # evidence for this launch); prints nothing on error, and the caller
  # fails closed
  python3 - "$MANIFEST" "$1" "$LOGS" "$START_EPOCH" <<'PY'
import json, os, re, sys
m, wave, logs, start = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3], float(sys.argv[4])
n = 0
for spec in m[wave]["runs"]:
    tag, level, rep = spec.split()
    p = os.path.join(logs, f"{tag}_{level}_r{rep}.log")
    if os.path.exists(p) and os.path.getmtime(p) >= start and re.search(r"DONE .* exit=0", open(p, errors="replace").read()):
        n += 1
print(n)
PY
}

START_EPOCH="$(date +%s)"
status=""
first=1
for wave in "$@"; do
  echo "$wave start at $(date -u +%H:%M:%SZ)"
  run_wave "$wave"; rc=$?
  status="$status $wave=$rc"
  if [ "$first" = 1 ]; then
    first=0
    completed="$(completed_in "$wave")"
    echo "$wave rc=$rc completed=${completed:-ERROR} at $(date -u +%H:%M:%SZ)"
    case "$completed" in
      ''|*[!0-9]*) echo "EXTENSION_DONE$status (completion count failed; stopping) at $(date -u +%H:%M:%SZ)"; exit 1 ;;
      0) echo "EXTENSION_DONE$status (no run of the first wave completed; route failure; later waves not started) at $(date -u +%H:%M:%SZ)"; exit 1 ;;
    esac
  else
    echo "$wave rc=$rc at $(date -u +%H:%M:%SZ)"
  fi
done
echo "EXTENSION_DONE$status at $(date -u +%H:%M:%SZ)"
for s in $status; do [ "${s#*=}" = 0 ] || exit 1; done
exit 0
