#!/bin/bash
# Launch one pre-registered wave from runs_2026_09/waves.json: its runs, in
# its order, at most <concurrency> at a time (batches), one log per run, and
# a terminal marker when every run has finished. Refuses to start unless the
# usage gate in waves.json passes, judged from the latest rate_limit_event in
# any session log on disk. Exits non-zero if any run failed. Ctrl-C or
# SIGTERM stops the running batch (each run's supervisor kills its group).
#
# Usage:  bash scripts/launch_wave.sh <wave-name>
# Works under macOS bash 3.2.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WAVE="${1:?wave name (a key of runs_2026_09/waves.json)}"
MANIFEST="$ROOT/runs_2026_09/waves.json"
LOGS="$ROOT/runs_2026_09/_logs"
mkdir -p "$LOGS"

# --- membership and concurrency from the manifest ---------------------------
SPEC_LINES="$(python3 - "$MANIFEST" "$WAVE" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
w = m.get(sys.argv[2])
if not isinstance(w, dict) or "runs" not in w:
    sys.exit(f"no wave '{sys.argv[2]}' in {sys.argv[1]}")
print(int(w.get("concurrency", 1)))
for r in w["runs"]:
    print(r)
PY
)" || { echo "$SPEC_LINES" >&2; exit 2; }
CONC="$(printf '%s\n' "$SPEC_LINES" | head -1)"
SPECS="$(printf '%s\n' "$SPEC_LINES" | tail -n +2)"

# --- usage gate ------------------------------------------------------------
# The gate reads Claude rate-limit events, so it applies only to waves that
# contain a Claude-routed run; a wave made entirely of gateway-routed runs
# (Extension A, DESIGN.md section 11) skips it.
ALL_GATEWAY=1
while IFS= read -r spec; do
  [ -z "$spec" ] && continue
  case "${spec%% *}" in astra|sol|gpt55|astraos|solos|gpt55os) ;; *) ALL_GATEWAY=0 ;; esac
done <<< "$SPECS"
if [ "$ALL_GATEWAY" = 1 ]; then
GATE="SKIPPED: every run in this wave is gateway-routed, so the Claude five-hour gate does not apply (DESIGN.md section 11)"
GATE_RC=0
else
GATE="$(python3 - "$ROOT/runs_2026_09" "$MANIFEST" <<'PY'
import glob, json, os, sys, time
runs, manifest = sys.argv[1], sys.argv[2]
limit = float(json.load(open(manifest))["usage_gate"]["max_five_hour_utilization"])
latest = None  # (mtime, line_no, util, resets_at, file)
for path in glob.glob(os.path.join(runs, "**", "session*.jsonl"), recursive=True):
    mtime = os.path.getmtime(path)
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if '"rate_limit_event"' not in line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            info = ev.get("rate_limit_info") or {}
            fh_ = (info.get("unifiedWindows") or {}).get("five_hour") or {}
            util, resets = fh_.get("utilization"), fh_.get("resetsAt")
            if util is None:
                continue
            key = (mtime, i)
            if latest is None or key > latest[:2]:
                latest = (mtime, i, float(util), resets, path)
now = time.time()
if latest is None:
    print("PASS no rate_limit_event on disk; nothing to judge from")
    sys.exit(0)
_, _, util, resets, path = latest
if resets and now >= float(resets):
    print(f"PASS fresh window: last observation util={util:.2f} resetsAt={time.strftime('%H:%M', time.localtime(float(resets)))} (past) from {os.path.relpath(path, runs)}")
    sys.exit(0)
if util <= limit:
    print(f"PASS util={util:.2f} <= {limit} (resets {time.strftime('%H:%M', time.localtime(float(resets or 0)))}) from {os.path.relpath(path, runs)}")
    sys.exit(0)
print(f"FAIL util={util:.2f} > {limit}, window resets {time.strftime('%H:%M', time.localtime(float(resets or 0)))} (from {os.path.relpath(path, runs)})")
sys.exit(5)
PY
)"; GATE_RC=$?
fi
echo "usage gate: $GATE"
if [ "$GATE_RC" != 0 ]; then
  echo "refusing to launch wave $WAVE" >&2
  exit 5
fi

# --- validate every run before launching anything --------------------------
set -f
names=()
while IFS= read -r spec; do
  [ -z "$spec" ] && continue
  tag=""; level=""; rep=""; extra=""
  read -r tag level rep extra <<< "$spec"
  if [ -z "$tag" ] || [ -z "$level" ] || [ -z "$rep" ] || [ -n "$extra" ]; then
    echo "bad spec in manifest: '$spec'" >&2; exit 2
  fi
  case "$tag" in fable51|opus5|opus48|opus47|astra|sol|gpt55|astraos|solos|gpt55os) ;; *) echo "bad model tag in '$spec'" >&2; exit 2 ;; esac
  case "$level" in L1|L2|L3) ;; *) echo "bad level in '$spec'" >&2; exit 2 ;; esac
  case "$rep" in ''|*[!0-9]*) echo "bad rep in '$spec'" >&2; exit 2 ;; esac
  name="${tag}_${level}_r${rep}"
  for seen in ${names[@]+"${names[@]}"}; do
    if [ "$seen" = "$name" ]; then echo "duplicate run in wave: $name" >&2; exit 2; fi
  done
  if [ -e "$ROOT/runs_2026_09/$name" ]; then echo "run dir already exists: $name" >&2; exit 2; fi
  names+=("$name")
done <<< "$SPECS"

pids=()
cleanup() {
  echo "wave $WAVE interrupted; stopping runs" >&2
  for p in ${pids[@]+"${pids[@]}"}; do kill -TERM "$p" 2>/dev/null || true; done
  wait
  exit 143
}
trap cleanup INT TERM

fail=0
i=0
batch=()
launch_batch() {
  pids=()
  for spec in "${batch[@]}"; do
    read -r tag level rep extra <<< "$spec"
    name="${tag}_${level}_r${rep}"
    bash "$ROOT/scripts/run_headless.sh" "$tag" "$level" "$rep" > "$LOGS/$name.log" 2>&1 &
    pids+=("$!")
    echo "launched $name pid $! at $(date -u +%H:%M:%SZ)"
  done
  for p in "${pids[@]}"; do
    wait "$p" || fail=$((fail + 1))
  done
  batch=()
}
while IFS= read -r spec; do
  [ -z "$spec" ] && continue
  batch+=("$spec")
  if [ "${#batch[@]}" -ge "$CONC" ]; then launch_batch; fi
done <<< "$SPECS"
if [ "${#batch[@]}" -gt 0 ]; then launch_batch; fi
echo "ALL_DONE $WAVE failures=$fail at $(date -u +%H:%M:%SZ)"
[ "$fail" -eq 0 ]
