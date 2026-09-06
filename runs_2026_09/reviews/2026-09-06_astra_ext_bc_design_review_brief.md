# Review brief: Extensions B and C, design and harness diff (read-only)

You are reviewing two pre-registered extensions to a finished experiment
before they are launched. Read-only: do not modify, create or delete
anything; you may run read-only shell commands (cat, grep, git diff, git
show, python3 for parsing). Work from the repository root.

## What to review

1. `runs_2026_09/DESIGN.md` sections 12 and 13 (new), against sections 1
   to 11 of the same file (the original design and Extension A) and
   against `RESULTS.md` section 8 (the Extension A results these sections
   react to).
2. The harness diff for the two extensions: `git diff 0e7f0d1 b1f1fb4`
   (touches `scripts/run_headless.sh`, `scripts/launch_wave.sh`,
   `scripts/launch_extension.sh`, `runs_2026_09/waves.json`,
   `runs_2026_09/pins.sha256`, and adds `prompts/one_shot_suffix.md`).
   Read `scripts/launch_extension.sh` whole (it was rewritten).
3. The frozen prompts `prompts/L1.md`, `prompts/L2.md`, `prompts/L3.md`
   and the suffix, to judge the concatenation.

## Questions, in priority order

For each, state what you expected, what you found, and the file and line.

1. Does the runner do what section 12 says and only that? The suffix is
   appended on stdin for the three `*os` tags only; the frozen prompt files,
   the `AGENTS.md` files and the sandbox copy of `prompts/L3.md` are
   untouched; `run_meta.json` records the suffix hash and the delivery;
   the Claude arm's and Extension A's invocations are byte-for-byte
   unchanged (compare the `env -i` line and the argument list). Is the
   concatenation well formed for all three prompts (trailing newline,
   blank line before the note)? Is the suffix text itself fit for purpose:
   does it remove the option of stopping without hinting at any method,
   time zone or forecast origin?
2. Does the runner do what section 13 says for `opus48` and `opus47`: direct
   route, no gateway environment, same flags as `opus5`? Are the model ids
   right for what the design claims?
3. Waves and driver. Are the cell lists right (B: 20 runs, exactly the
   Extension A runs that ended without a forecast, see `RESULTS.md` 8.1
   and `scoring.json` entries whose `headline_model` starts with "none";
   C: 24 runs), are run-directory names unique against the existing 39, is
   the interleaving what the sections describe, and does the rewritten
   driver behave: the gate retry loop (exit 5 only), the completed-run
   count from per-run logs, the not_started branch, the exit status? Can
   the two drivers run concurrently (B on the gateway, C direct) without
   sharing anything but the machine: log names, sandboxes, the manifest,
   the gate (the gate reads every session log on disk, including the
   gateway sessions, which emit no rate_limit_event; confirm)?
4. The identifier scan: do the new tags appear in the pattern so a sandbox
   that names a run of any tag fails the scan, and is `gpt55os` covered?
5. Could either extension contaminate or alter the 39 counted runs, their
   `scoring.json` entries or derived files? Name the file if so. Note
   `opus5` reps 2 and 3 share a tag with the counted `opus5` rep 1.
6. Is anything in sections 12 or 13 stated more strongly than the design
   supports (comparability claims, what the May-versus-September Opus 4.7
   comparison can and cannot show, the mixed GPT-5.5 L3 cell)?
7. What did you not check, and why.

## Output

A numbered list of findings, most serious first: severity (blocker, major,
minor, note), the claim or line, expected versus actual, and a one-sentence
suggested fix. End with the list of things not checked. No preamble.
