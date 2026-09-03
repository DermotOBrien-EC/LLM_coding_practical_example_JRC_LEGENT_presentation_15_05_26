#!/bin/bash
# Resume an L3 run once after an infrastructure failure (DESIGN.md section 3,
# attempt policy). Refuses for L1/L2, for a run that completed, for a second
# resume, and unless the first attempt ended in a timeout, a stop, or a
# rate-limit / API error visible in its log. The resumed session gets the
# single message "Continue." and is logged to session_resume1.jsonl.
#
# Usage:  bash scripts/resume_run.sh <run-name> "<reason>"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="/Users/doob/.local/share/claude/versions/2.1.259"
CAP_SECONDS=10800
RUNNAME="${1:?run name, e.g. fable51_L3_r1}"
REASON="${2:?one-line reason (infrastructure failure only)}"
RUN="$ROOT/runs_2026_09/$RUNNAME"
[ -d "$RUN" ] || { echo "no such run: $RUN" >&2; exit 2; }

CHECK="$(python3 - "$RUN" <<'PY'
import json, os, sys
run = sys.argv[1]
meta = json.load(open(os.path.join(run, "run_meta.json")))
if meta.get("level") != "L3":
    sys.exit("only L3 may be resumed (design section 3)")
if int(meta.get("attempts", 1)) != 1 or meta.get("resumed"):
    sys.exit("already resumed once; a second resume is not allowed")
if meta.get("status") == "completed":
    sys.exit("first attempt completed; nothing to resume")
sid = None
infra = meta.get("status") in ("timeout", "stopped")
for line in open(os.path.join(run, "session.jsonl"), encoding="utf-8", errors="replace"):
    if '"init"' in line and sid is None:
        try:
            ev = json.loads(line)
            if ev.get("type") == "system" and ev.get("subtype") == "init":
                sid = ev.get("session_id")
        except json.JSONDecodeError:
            pass
    if '"rate_limit_event"' in line and '"status": "allowed' not in line and '"status":"allowed' not in line:
        infra = True
    if '"result"' in line and ('rate_limit' in line or 'api_error' in line or 'overloaded' in line or '"is_error": true' in line or '"is_error":true' in line):
        infra = True
if not sid:
    sys.exit("no session id found in session.jsonl")
if not infra:
    sys.exit("first attempt did not end in an infrastructure failure; not resumable under the design")
print(sid)
print(meta["work_dir"])
print(meta["sandbox_token"])
PY
)" || { echo "$CHECK" >&2; exit 5; }
SID="$(printf '%s\n' "$CHECK" | sed -n 1p)"
WORK="$(printf '%s\n' "$CHECK" | sed -n 2p)"
TOKEN="$(printf '%s\n' "$CHECK" | sed -n 3p)"
[ -d "$WORK" ] || { echo "sandbox work dir gone: $WORK" >&2; exit 5; }
SB="$RUN/seatbelt.sb"
STAGE="${TMPDIR:-/tmp}/hs/$TOKEN"
mkdir -p "$STAGE"
printf 'Continue.\n' > "$STAGE/input.txt"
cp "$ROOT/scripts/supervise.py" "$STAGE/sv.py"

python3 - "$RUN/run_meta.json" "$REASON" <<'PY'
import json, sys, datetime
p, reason = sys.argv[1], sys.argv[2]
meta = json.load(open(p))
meta.update({"attempts": 2, "resumed": True, "resume_reason": reason,
             "resume_started_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "resume_message": "Continue.", "status": "resuming"})
json.dump(meta, open(p, "w"), indent=2)
PY

# the supervisor writes session.jsonl in its cwd; use a scratch dir and move
RES="$RUN/_resume"
mkdir -p "$RES"
cd "$RES"
set +e
python3 "$STAGE/sv.py" "$CAP_SECONDS" "$STAGE/input.txt" "$WORK" -- \
  sandbox-exec -f "$SB" \
  env -i HOME="$HOME" USER="$USER" LOGNAME="$USER" SHELL=/bin/zsh LANG=en_US.UTF-8 \
      PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Library/TeX/texbin \
      "$BIN" -p --resume "$SID" \
      --setting-sources project,local \
      --permission-mode auto --permission-prompts none \
      --output-format stream-json --verbose
EXIT=$?
set -e
cd "$ROOT"
mv "$RES/session.jsonl" "$RUN/session_resume1.jsonl"
mv "$RES/stderr.log" "$RUN/stderr_resume1.log"
mv "$RES/supervise_status.json" "$RUN/supervise_status_resume1.json"
rmdir "$RES" 2>/dev/null || true
rm -rf "$STAGE"
WS="$(dirname "$(dirname "$WORK")")"
python3 "$ROOT/scripts/inventory.py" snapshot "$WS" "$RUN/inventory_after_resume1.json" --stat-only .venv > /dev/null
python3 "$ROOT/scripts/inventory.py" diff "$RUN/inventory_before.json" "$RUN/inventory_after_resume1.json" "$RUN/changes.json" > /dev/null
rm -rf "$RUN/output" "$RUN/output_outside"
python3 "$ROOT/scripts/inventory.py" harvest "$RUN/changes.json" "$WS" "runs/$TOKEN" "$RUN" opsd_de_load.csv AGENTS.md > /dev/null
python3 - "$RUN/run_meta.json" "$EXIT" <<'PY'
import json, sys, datetime
p, code = sys.argv[1], int(sys.argv[2])
meta = json.load(open(p))
meta.update({"resume_exit_code": code,
             "status": {0: "completed", 124: "timeout", 143: "stopped"}.get(code, "exited_nonzero") + " (after resume)",
             "resume_ended_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
json.dump(meta, open(p, "w"), indent=2)
PY
echo "RESUMED $RUNNAME exit=$EXIT"
exit "$EXIT"
