#!/bin/bash
# Launch one read-only GPT-5.6-Sol (codex exec) assessor per run directory and
# collect strict-JSON assessments under runs_2026_09/_assess/<run>.json.
#
# Usage:  bash scripts/assess_fleet.sh <run-name> [<run-name> ...]
#         bash scripts/assess_fleet.sh all          # every run dir with session.jsonl
# Concurrency: 4 (xargs -P). Each job has a 1500 s alarm.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="$ROOT/runs_2026_09"
OUT="$RUNS/_assess"
mkdir -p "$OUT"

if [ "${1:-}" = "all" ]; then
  set -- $(cd "$RUNS" && for d in */; do d="${d%/}"; [ -f "$d/session.jsonl" ] && echo "$d"; done)
fi

one() {
  run="$1"
  level="$(echo "$run" | sed -E 's/^[a-z0-9]+_(L[123])_r[0-9]+$/\1/')"
  case "$level" in
    L1) note="and nothing else" ;;
    L2) note="and a 7-line AGENTS.md (runs/L2/AGENTS.md)" ;;
    L3) note="and a 113-line AGENTS.md (runs/L3/AGENTS.md)" ;;
    *) echo "bad run name: $run" >&2; return 2 ;;
  esac
  brief="$OUT/$run.brief.md"
  delivery="the prompt in prompts/$level.md, which you may read"
  if python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('one_shot_suffix_sha256') else 1)" "$RUNS/$run/run_meta.json" 2>/dev/null; then
    delivery="the prompt in prompts/$level.md followed by the note in prompts/one_shot_suffix.md, both of which you may read (Extension B, DESIGN.md section 12)"
  fi
  sed -e "s|__RUN__|$run|g" -e "s|__LEVEL__|$level|g" -e "s|__AGENTS_NOTE__|$note|g" -e "s|__DELIVERY__|$delivery|g" \
      "$ROOT/scripts/assess_brief_template.md" > "$brief"
  cat "$brief" | perl -e 'alarm shift; exec @ARGV' 1500 \
    codex exec -c 'model_reasoning_effort="high"' -s read-only --skip-git-repo-check \
      --output-schema "$ROOT/scripts/assess_schema.json" \
      -C "$ROOT" -o "$OUT/$run.json" --json - > "$OUT/$run.events.jsonl" 2> "$OUT/$run.err"
  code=$?
  if python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$OUT/$run.json" 2>/dev/null; then
    echo "ASSESS_OK $run exit=$code"
  else
    echo "ASSESS_BAD $run exit=$code"
  fi
}
export -f one
export ROOT RUNS OUT
printf '%s\n' "$@" | xargs -P 4 -I{} bash -c 'one "$@"' _ {}
echo "ASSESS_ALL_DONE"
