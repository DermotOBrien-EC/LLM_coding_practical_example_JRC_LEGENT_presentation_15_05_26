# Review brief: Extension A design and harness diff (read-only)

You are reviewing a pre-registered extension to a finished experiment
before it is launched. Read-only: do not modify, create or delete anything;
you may run read-only shell commands (cat, grep, git diff, python3 for
parsing). Work from the repository root.

## What to review

1. `runs_2026_09/DESIGN.md` section 11 ("Extension A"), against sections 1
   to 7 of the same file (the original pre-registered design) and against
   the harness as it is now.
2. The harness diff for the extension: `git diff HEAD -- scripts/run_headless.sh
   scripts/launch_wave.sh runs_2026_09/waves.json scripts/build_rerun_figures.py
   scripts/launch_extension.sh` (the last file is new; read it whole).
3. For context on the route: the operator's launcher function is quoted
   below; the runner is meant to reproduce its environment exactly for the
   three OpenAI model tags and nothing else.

```
ANTHROPIC_BASE_URL="$base" ANTHROPIC_AUTH_TOKEN="$key" CLAUDE_CODE_SUBAGENT_MODEL="gpt-6-astra" \
CLAUDE_CODE_ALWAYS_ENABLE_EFFORT=1 CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY=3 \
CLAUDE_CODE_MAX_CONTEXT_TOKENS=272000 ENABLE_TOOL_SEARCH=false \
claude --model gpt-6-astra --permission-mode bypassPermissions "$@"
```

## Questions, in priority order

For each, state what you expected, what you found, and the file and line.

1. Does the runner diff do what section 11 says, and only that? In
   particular: the gateway environment is added for the three new tags
   only; the Claude arm's command line and environment are byte-for-byte
   unchanged; the key never reaches `run_meta.json`, the session log's
   argv, or the sandbox; the identifier scan still refuses a sandbox that
   names the study; the `env -i` line is valid bash 3.2 when
   `GATEWAY_ENV` is empty (macOS ships bash 3.2 and the script runs under
   it).
2. Section 11 lists eight known differences from the Claude arm. Is any of
   them wrong, and is any material difference missing? Consider the
   permission classifier, the keep-alive, the window pin, tool search,
   subagents, effort, rate limits, and anything in the launcher line above
   that section 11 does not mention (note the launcher uses
   `bypassPermissions`; the runner keeps `auto` with prompts routed to
   nobody, as the Claude arm did).
3. The waves: are the cell counts right (27 runs), is the order of
   `ext_waveB` the interleaving section 11 describes, and can the launcher
   run them as written (tag validation, concurrency, existing run
   directories, the usage gate reading stale Claude rate-limit events)?
4. Is there any way the extension could contaminate or alter the twelve
   counted Claude runs or their scoring files? Name the file if so.
5. Is anything in section 11 stated more strongly than the design supports
   (for example about comparability across vendors)?
6. What did you not check, and why.

## Output

A numbered list of findings, most serious first: severity (blocker, major,
minor, note), the claim or line, expected versus actual, and a one-sentence
suggested fix. End with the list of things not checked. No preamble.
