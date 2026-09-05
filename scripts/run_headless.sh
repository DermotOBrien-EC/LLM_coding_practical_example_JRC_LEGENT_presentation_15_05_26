#!/bin/bash
# Run one level of the prompt experiment headlessly in a fresh, isolated
# Claude Code session inside a neutral, seatbelt-confined sandbox, and keep
# the full session log. See runs_2026_09/DESIGN.md section 3.
#
# Usage:  bash scripts/run_headless.sh <fable51|opus5> <L1|L2|L3> <rep>
#
# Fails closed unless the pinned inputs (runs_2026_09/pins.sha256) and the
# pinned Claude Code build verify, and unless the built sandbox passes a
# scan for study identifiers.
#
# Sandbox: ~/dev/energy_forecast_ws/<token>/project/ holds an APFS clone of
# the repository venv (.venv, with pyvenv.cfg and entry-point shebangs
# rewritten to the clone), a generic pyproject.toml and uv.lock, and the
# working directory runs/<token>/ with the data file and, for L2/L3, the
# frozen AGENTS.md; for L3 also prompts/L3.md, which its AGENTS.md cites.
# The agent process runs under a macOS seatbelt profile that denies the
# repository, every other sandbox, every other ~/.claude/projects entry and
# ~/.claude/history.jsonl, and process listing. The prompt file, the
# supervisor and the logs are staged or piped so that nothing in the
# agent's argv, fds or cwd names the run, the level or the model tag.
#
# Harness files live in runs_2026_09/<model>_<level>_r<rep>/ (run_meta.json,
# session.jsonl, stderr.log, inventories); the agent's files are harvested
# afterwards from a before/after inventory diff into output/ (working
# directory) and output_outside/ (anywhere else in the sandbox). The prompt
# is fed on stdin, byte for byte, as the first and only user message.
# Permissions run in auto mode with prompts routed to nobody. The session
# runs in its own process group under a wall-clock cap.
#
# Wave 2 relaunch (2026-09-05, DESIGN.md section 9): the agent's environment
# also carries CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0. Without it, print mode
# ends the session 600 s after a turn that leaves background jobs running
# ("Background tasks still running after 600s; terminating"), which is what
# cut the first opus5 L3 attempt off mid-training. With it, the session waits
# for the agent's own background jobs and re-enters the model when one
# finishes, as an interactive session would; the wall-clock cap still binds.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="/Users/doob/.local/share/claude/versions/2.1.259"
EXPECT_VERSION="2.1.259 (Claude Code)"
CAP_SECONDS=10800   # the same cap for every level (design section 3)
WS_BASE="$HOME/dev/energy_forecast_ws"
STAGE_BASE="${TMPDIR:-/tmp}"

MODEL_TAG="${1:?model tag (fable51|opus5|opus48|opus47|astra|sol|gpt55|astraos|solos|gpt55os)}"
LEVEL="${2:?level (L1|L2|L3)}"
REP="${3:?rep number}"

# Extension A (DESIGN.md section 11): the OpenAI models run on this same
# harness through the operator's local CLIProxyAPI gateway, which serves
# them to Claude Code under their Codex ids. ROUTE=gateway adds exactly the
# environment the operator's `poly` launcher uses; ROUTE=direct is the
# Claude arm, unchanged.
# Extension B (section 12): the same OpenAI models with a one-shot note
# appended to the prompt on stdin (ONESHOT=1; the frozen prompt files are
# untouched). Extension C (section 13): the Opus family on the direct route.
ROUTE="direct"
ONESHOT=0
case "$MODEL_TAG" in
  fable51) MODEL="claude-fable-5-1" ;;
  opus5)   MODEL="claude-opus-5" ;;
  opus48)  MODEL="claude-opus-4-8" ;;
  opus47)  MODEL="claude-opus-4-7" ;;
  astra)   MODEL="gpt-6-astra";  ROUTE="gateway" ;;
  sol)     MODEL="gpt-5.6-sol";  ROUTE="gateway" ;;
  gpt55)   MODEL="gpt-5.5";      ROUTE="gateway" ;;
  astraos) MODEL="gpt-6-astra";  ROUTE="gateway"; ONESHOT=1 ;;
  solos)   MODEL="gpt-5.6-sol";  ROUTE="gateway"; ONESHOT=1 ;;
  gpt55os) MODEL="gpt-5.5";      ROUTE="gateway"; ONESHOT=1 ;;
  *) echo "unknown model tag: $MODEL_TAG" >&2; exit 2 ;;
esac
GATEWAY_ENV=()
if [ "$ROUTE" = "gateway" ]; then
  SP="/Users/doob/dev/Default_project_skills_n_setup/scripts/sol-proxy.sh"
  bash "$SP" ensure --with-claude > /dev/null || { echo "gateway not available" >&2; exit 4; }
  bash "$SP" has-model "$MODEL" > /dev/null 2>&1 || { echo "gateway catalog lacks $MODEL" >&2; exit 4; }
  GW_BASE="$(bash "$SP" print-base-url)" || exit 4
  GW_KEY="$(bash "$SP" print-key)" || exit 4
  GATEWAY_ENV=(ANTHROPIC_BASE_URL="$GW_BASE" ANTHROPIC_AUTH_TOKEN="$GW_KEY"
               CLAUDE_CODE_SUBAGENT_MODEL="$MODEL" CLAUDE_CODE_ALWAYS_ENABLE_EFFORT=1
               CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY=3 CLAUDE_CODE_MAX_CONTEXT_TOKENS=272000
               ENABLE_TOOL_SEARCH=false)
fi
case "$LEVEL" in
  L1|L2|L3) ;;
  *) echo "unknown level: $LEVEL" >&2; exit 2 ;;
esac
case "$REP" in
  ''|*[!0-9]*) echo "rep must be an integer: $REP" >&2; exit 2 ;;
esac

# --- fail closed on the pinned inputs and the pinned build -----------------
GOT_VERSION="$("$BIN" --version 2>/dev/null || true)"
if [ "$GOT_VERSION" != "$EXPECT_VERSION" ]; then
  echo "claude build mismatch: got '$GOT_VERSION', expected '$EXPECT_VERSION'" >&2
  exit 4
fi
if ! (cd "$ROOT" && shasum -a 256 -c --status runs_2026_09/pins.sha256); then
  echo "pinned inputs do not verify (runs_2026_09/pins.sha256)" >&2
  (cd "$ROOT" && shasum -a 256 -c runs_2026_09/pins.sha256 >&2) || true
  exit 4
fi

RUN="$ROOT/runs_2026_09/${MODEL_TAG}_${LEVEL}_r${REP}"
if ! mkdir "$RUN" 2>/dev/null; then
  echo "refusing to overwrite existing run dir: $RUN" >&2
  exit 3
fi

set_status() {
  python3 - "$RUN/run_meta.json" "$1" <<'PY'
import json, sys
p, status = sys.argv[1], sys.argv[2]
try:
    meta = json.load(open(p))
except (OSError, ValueError):
    meta = {}
meta["status"] = status
json.dump(meta, open(p, "w"), indent=2)
PY
}
FINAL=0
on_exit() {
  if [ "$FINAL" != 1 ]; then
    set_status "harness_failed_before_finalise"
    echo "DONE ${MODEL_TAG}_${LEVEL}_r${REP} exit=99 (harness failed before finalising)"
  fi
}
trap on_exit EXIT

# --- build the neutral sandbox --------------------------------------------
TOKEN="$(python3 -c 'import secrets; print(secrets.token_hex(3))')"
WS="$WS_BASE/$TOKEN/project"
WORK_REL="runs/$TOKEN"
WORK="$WS/$WORK_REL"
STAGE="$STAGE_BASE/hs/$TOKEN"
mkdir -p "$WORK" "$STAGE"
cp -Rc "$ROOT/.venv" "$WS/.venv"
# neutralise the clone's metadata: prompt name, entry-point shebangs, and the
# venv path and project name embedded in the activate scripts
/usr/bin/sed -i '' -E "s|^prompt = .*|prompt = energy-load-forecast|" "$WS/.venv/pyvenv.cfg"
for f in "$WS"/.venv/bin/*; do
  if [ -f "$f" ] && [ ! -L "$f" ] && /usr/bin/file -b "$f" | /usr/bin/grep -q -i 'text'; then
    /usr/bin/sed -i '' -E "s#/Users/doob/dev/[A-Za-z0-9_.-]+/\.venv#$WS/.venv#g; s#llm-coding-example-jrc-andres#energy-load-forecast#g" "$f"
  fi
done
cat > "$WS/pyproject.toml" <<'EOF'
[project]
name = "energy-load-forecast"
version = "0.1.0"
description = "Hourly electricity load forecasting."
requires-python = ">=3.11,<3.13"
dependencies = [
    "pandas>=2.2,<3",
    "numpy>=1.26,<3",
    "matplotlib>=3.8,<4",
    "statsmodels>=0.14,<1",
    "pmdarima>=2.0,<3",
    "u8darts[all]>=0.30,<1",
    "lightgbm>=4,<5",
    "holidays>=0.50,<1",
]

[dependency-groups]
dev = [
    "ruff>=0.6,<1",
    "mypy>=1.10,<2",
    "pytest>=8,<9",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
strict = true
python_version = "3.11"
EOF
/usr/bin/sed -E 's/^name = "llm-coding-example-jrc-andres"/name = "energy-load-forecast"/' "$ROOT/uv.lock" > "$WS/uv.lock"
if [ "$LEVEL" = "L3" ]; then
  mkdir -p "$WS/prompts"
  cp "$ROOT/prompts/L3.md" "$WS/prompts/L3.md"
fi
cp "$ROOT/data/opsd_de_load.csv" "$WORK/opsd_de_load.csv"
AGENTS_SHA="none"
if [ "$LEVEL" != "L1" ]; then
  cp "$ROOT/runs/$LEVEL/AGENTS.md" "$WORK/AGENTS.md"
  AGENTS_SHA="$(shasum -a 256 "$WORK/AGENTS.md" | cut -d' ' -f1)"
fi
DATA_SHA="$(shasum -a 256 "$WORK/opsd_de_load.csv" | cut -d' ' -f1)"
PROMPT_FILE="$ROOT/prompts/$LEVEL.md"
PROMPT_SHA="$(shasum -a 256 "$PROMPT_FILE" | cut -d' ' -f1)"
SUFFIX_FILE="$ROOT/prompts/one_shot_suffix.md"
SUFFIX_SHA="none"
if [ "$ONESHOT" = 1 ]; then
  SUFFIX_SHA="$(shasum -a 256 "$SUFFIX_FILE" | cut -d' ' -f1)"
fi

# --- scan the sandbox (outside the venv's library tree) for identifiers ----
# The L3 AGENTS.md and prompt legitimately name the study, so they are
# excluded; everything else must be clean.
LEAKS="$(cd "$WS" && /usr/bin/grep -rIl -E 'LLM_coding|LEGENT|jrc_andres|jrc-andres|prompt-engineering|three-level|fable51|opus5|opus4[78]|astra_L[123]|sol_L[123]|astraos|solos|gpt55|runs_2026_09' \
  --exclude-dir=lib --exclude=AGENTS.md --exclude=L3.md . 2>/dev/null || true)"
if [ -n "$LEAKS" ]; then
  echo "identifier scan failed; sandbox names the study in:" >&2
  echo "$LEAKS" >&2
  exit 4
fi

# --- stage the prompt and the supervisor under neutral names --------------
if [ "$ONESHOT" = 1 ]; then
  cat "$PROMPT_FILE" "$SUFFIX_FILE" > "$STAGE/input.txt"
else
  cp "$PROMPT_FILE" "$STAGE/input.txt"
fi
cp "$ROOT/scripts/supervise.py" "$STAGE/sv.py"

# --- seatbelt profile for the agent process --------------------------------
PROJ_KEY="$(printf '%s' "$WORK" | sed -E 's#[/_]#-#g')"
SB="$RUN/seatbelt.sb"
cat > "$SB" <<EOF
(version 1)
(allow default)
(deny file-read* file-write* (subpath "$HOME/dev"))
(allow file-read* file-write* (subpath "$WS_BASE/$TOKEN"))
(deny file-read* file-write* (subpath "$HOME/.claude/projects"))
(allow file-read* file-write* (regex #"^$HOME/\\.claude/projects/$PROJ_KEY"))
(deny file-read* file-write* (literal "$HOME/.claude/history.jsonl"))
(deny file-read* file-write* (subpath "$HOME/.codex/sessions"))
(deny file-read* file-write* (subpath "$HOME/.codex/log"))
(deny file-read* file-write* (literal "$HOME/.codex/history.jsonl"))
(deny file-read* file-write* (subpath "$HOME/cliproxyapi"))
(deny file-read* file-write* (subpath "$STAGE_BASE/hs"))
(allow file-read* file-write* (subpath "$STAGE"))
(deny process-info* (with no-log))
(allow process-info* (target self))
(allow process-info* (target children))
EOF

# --- inventory before -------------------------------------------------------
python3 "$ROOT/scripts/inventory.py" snapshot "$WS" "$RUN/inventory_before.json" --stat-only .venv > /dev/null
START_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
START_EPOCH="$(date +%s)"

python3 - "$RUN/run_meta.json" "$MODEL_TAG" "$MODEL" "$LEVEL" "$REP" "$START_ISO" "$DATA_SHA" \
          "$PROMPT_SHA" "$AGENTS_SHA" "$BIN" "$GOT_VERSION" "$CAP_SECONDS" "$WS" "$WORK" "$TOKEN" "$ROUTE" "$SUFFIX_SHA" <<'PY'
import json, sys
(out, tag, model, level, rep, start, dsha, psha, asha, binp, ver, cap, ws, work, token, route, ssha) = sys.argv[1:]
meta = {
    "model_tag": tag, "model": model, "level": level, "rep": int(rep),
    "started_utc": start,
    "data_sha256": dsha, "prompt_sha256": psha, "agents_md_sha256": asha,
    "claude_binary": binp, "claude_code_version": ver,
    "wallclock_cap_seconds": int(cap),
    "sandbox_token": token, "sandbox_root": ws, "work_dir": work,
    "sandbox_contents": [".venv (APFS clone of the repository venv, pyvenv.cfg prompt and entry-point shebangs rewritten)",
                          "pyproject.toml (generic, same dependencies)", "uv.lock (root package renamed to match)",
                          "runs/<token>/opsd_de_load.csv", "runs/<token>/AGENTS.md (L2, L3 only)", "prompts/L3.md (L3 only)"],
    "confinement": "macOS seatbelt (seatbelt.sb): denies ~/dev except this sandbox, ~/.claude/projects except this run's entry, ~/.claude/history.jsonl, the staging area except this run's, and process listing; not a VM",
    "prompt_delivery": ("stdin from a neutrally named staged copy, bytes unchanged" if ssha == "none" else
                        "stdin from a neutrally named staged copy: the frozen prompt followed by prompts/one_shot_suffix.md (Extension B, DESIGN.md section 12)"),
    "one_shot_suffix_sha256": None if ssha == "none" else ssha,
    "launch_flags": ["-p", "--model", model, "--setting-sources", "project,local",
                     "--permission-mode", "auto", "--permission-prompts", "none",
                     "--output-format", "stream-json", "--verbose"],
    "env_passed": {"HOME": "operator home", "USER": "operator user", "LOGNAME": "operator user",
                   "SHELL": "/bin/zsh", "LANG": "en_US.UTF-8",
                   "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Library/TeX/texbin",
                   "CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS": "0 (wave 2 relaunch onwards: print mode waits for the agent's background jobs instead of ending the session 600 s after a turn with no tool call)"},
    "effort_flag": None, "attempts": 1, "resumed": False, "status": "running",
    "route": route,
}
if route == "gateway":
    meta["route"] = "gateway: local CLIProxyAPI (127.0.0.1:8317) serving the model from the operator's Codex subscription (DESIGN.md section 11)"
    meta["env_passed"].update({
        "ANTHROPIC_BASE_URL": "the local gateway (value recorded in DESIGN.md section 11)",
        "ANTHROPIC_AUTH_TOKEN": "the local gateway's key (not recorded)",
        "CLAUDE_CODE_SUBAGENT_MODEL": model,
        "CLAUDE_CODE_ALWAYS_ENABLE_EFFORT": "1", "CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY": "3",
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "272000", "ENABLE_TOOL_SEARCH": "false"})
json.dump(meta, open(out, "w"), indent=2)
PY

# --- run the agent under the supervisor (cwd = run dir, child cwd = work) ---
cd "$RUN"
set +e
python3 "$STAGE/sv.py" "$CAP_SECONDS" "$STAGE/input.txt" "$WORK" -- \
  sandbox-exec -f "$SB" \
  env -i HOME="$HOME" USER="$USER" LOGNAME="$USER" SHELL=/bin/zsh LANG=en_US.UTF-8 \
      PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Library/TeX/texbin \
      CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0 \
      ${GATEWAY_ENV[@]+"${GATEWAY_ENV[@]}"} \
      "$BIN" -p --model "$MODEL" \
      --setting-sources project,local \
      --permission-mode auto --permission-prompts none \
      --output-format stream-json --verbose &
SV_PID=$!
trap 'kill -TERM "$SV_PID" 2>/dev/null' TERM INT
wait "$SV_PID"
EXIT=$?
trap - TERM INT
set -e
cd "$ROOT"
set_status "agent_done"

# --- inventory after, diff, harvest ----------------------------------------
python3 "$ROOT/scripts/inventory.py" snapshot "$WS" "$RUN/inventory_after.json" --stat-only .venv > /dev/null
python3 "$ROOT/scripts/inventory.py" diff "$RUN/inventory_before.json" "$RUN/inventory_after.json" "$RUN/changes.json" > /dev/null
python3 "$ROOT/scripts/inventory.py" harvest "$RUN/changes.json" "$WS" "$WORK_REL" "$RUN" opsd_de_load.csv AGENTS.md > /dev/null
DATA_SHA_AFTER="none"
[ -f "$WORK/opsd_de_load.csv" ] && DATA_SHA_AFTER="$(shasum -a 256 "$WORK/opsd_de_load.csv" | cut -d' ' -f1)"
AGENTS_SHA_AFTER="none"
if [ -f "$WORK/AGENTS.md" ]; then
  AGENTS_SHA_AFTER="$(shasum -a 256 "$WORK/AGENTS.md" | cut -d' ' -f1)"
elif [ "$LEVEL" != "L1" ]; then
  AGENTS_SHA_AFTER="deleted"
fi
rm -rf "$STAGE"

END_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
END_EPOCH="$(date +%s)"
python3 - "$RUN/run_meta.json" "$EXIT" "$END_ISO" "$((END_EPOCH - START_EPOCH))" \
          "$DATA_SHA_AFTER" "$AGENTS_SHA_AFTER" "$RUN/harvest.json" "$RUN/supervise_status.json" <<'PY'
import json, sys
out, code, end, secs, dsha_after, asha_after, harvest_p, sv_p = sys.argv[1:]
code = int(code)
status = {0: "completed", 124: "timeout", 143: "stopped"}.get(code, "exited_nonzero")
meta = json.load(open(out))
try:
    harvest = json.load(open(harvest_p))
except (OSError, ValueError):
    harvest = {"error": "harvest.json missing"}
try:
    sv = json.load(open(sv_p))
except (OSError, ValueError):
    sv = {"error": "supervise_status.json missing"}
meta.update({"exit_code": code, "status": status, "ended_utc": end, "wallclock_seconds": int(secs),
             "venv_entries_changed": harvest.get("venv_entries_changed"),
             "files_deleted_in_sandbox": harvest.get("deleted"),
             "data_unchanged": dsha_after == meta["data_sha256"],
             "agents_md_unchanged": asha_after == meta["agents_md_sha256"],
             "group_survivors_after_kill": sv.get("group_survivors_after_kill")})
json.dump(meta, open(out, "w"), indent=2)
PY
FINAL=1
echo "DONE ${MODEL_TAG}_${LEVEL}_r${REP} exit=$EXIT secs=$((END_EPOCH - START_EPOCH))"
exit "$EXIT"
