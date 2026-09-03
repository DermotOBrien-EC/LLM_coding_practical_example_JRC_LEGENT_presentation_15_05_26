"""Summarise every headless run under runs_2026_09/ from its session logs.

For each run directory that holds ``session.jsonl`` (and, after a resume,
``session_resume*.jsonl``) this script extracts, without interpreting the
science:

- per attempt: model, session id, whether a terminal result event exists,
  stop reason, turns, tokens, list-price cost, tool-call counts, tool
  errors, JSON lines that failed to parse, permission denials, and every
  rate-limit or error event worth a human glance;
- across attempts: the totals, and whether ``AGENTS.md`` was read
  (``yes`` when a Read tool call or a read-capable shell utility named the
  file, ``possible`` when a glob such as ``cat *.md`` could have read it,
  ``no`` otherwise), the turn of the first confirmed read, and whether that
  came before the first implementation action (a Write/Edit tool call or a
  shell command that runs Python or redirects into a file);
- an exact manifest of what the agent left behind (path, bytes, SHA-256),
  with separate counts for PNGs, other images, HTML, documents, Python, CSV
  and JSON files;
- the agent's final message, saved as ``final_message.md`` in the run dir.

Writes ``<run>/summary.json``, ``<run>/manifest.json``,
``runs_2026_09/summary.json`` and ``runs_2026_09/summary.md``.

Usage:
    uv run python scripts/summarize_runs.py
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs_2026_09"

# Files the operator or the harness wrote at the run root; not agent output.
HARNESS_ROOT_FILES = {
    "opsd_de_load.csv", "AGENTS.md", "run_meta.json", "final_message.md",
    "summary.json", "manifest.json",
}
HARNESS_ROOT_PATTERNS = (re.compile(r"^session.*\.jsonl$"), re.compile(r"^stderr.*\.log$"))
SKIP_DIRS = {".claude", "__pycache__", "lightning_logs", "darts_logs", "cache",
             ".darts", ".ruff_cache", ".pytest_cache", "checkpoints", ".venv"}
READ_UTILS = {"cat", "head", "tail", "sed", "less", "more", "bat", "nl", "awk", "grep", "rg",
              "egrep", "fgrep", "python", "python3", "open", "strings", "view", "vim", "vi",
              "nano", "cut", "tac", "fold", "pr"}
IMG_EXT = {".jpg", ".jpeg", ".svg", ".gif", ".webp", ".tif", ".tiff", ".bmp"}
DOC_EXT = {".md", ".txt", ".rst", ".pdf", ".docx", ".tex"}
IMPL_BASH = re.compile(r"(^|[\s;&|])python3?\s+(?!--version)(?!-V\b)\S|(^|[^<>\d])>\s*(?!/dev/null|&)\S"
                       r"|<<\s*['\"]?\w+|\b(?:touch|cp|mv|tee|mkdir)\s|\bsed\s+-i")


@dataclass
class Attempt:
    file: str
    session_id: str = ""
    model: str = ""
    has_result: bool = False
    stop: str = "<no result event>"
    is_error: bool = False
    num_turns: int = 0
    assistant_messages: int = 0
    duration_s: float = 0.0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_create_tokens: int = 0
    tool_counts: dict[str, int] = field(default_factory=dict)
    tool_errors: int = 0
    parse_failures: int = 0
    permission_denials: int = 0
    denied_tools: list[str] = field(default_factory=list)
    rate_limit_max_5h: float = 0.0
    notable_events: list[str] = field(default_factory=list)
    bash_commands: list[str] = field(default_factory=list)


@dataclass
class RunSummary:
    run: str
    model_tag: str = ""
    level: str = ""
    rep: int = 0
    model: str = ""
    permission_mode: str = ""
    claude_code_version: str = ""
    mcp_servers: list[str] = field(default_factory=list)
    status: str = ""
    exit_code: int | None = None
    wallclock_s: int = 0
    resumed: bool = False
    attempts: list[Attempt] = field(default_factory=list)
    has_result: bool = False
    stop: str = ""
    is_error: bool = False
    num_turns: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_create_tokens: int = 0
    tool_counts: dict[str, int] = field(default_factory=dict)
    tool_errors: int = 0
    parse_failures: int = 0
    permission_denials: int = 0
    denied_tools: list[str] = field(default_factory=list)
    rate_limit_max_5h: float = 0.0
    notable_events: list[str] = field(default_factory=list)
    agents_md_read: str = "no"
    agents_md_evidence: str = ""
    agents_md_first_turn: str | None = None
    first_impl_turn: str | None = None
    agents_md_before_impl: bool | None = None
    outside_sandbox_refs: int = 0
    outside_sandbox_evidence: list[str] = field(default_factory=list)
    n_files: int = 0
    n_png: int = 0
    n_img_other: int = 0
    n_html: int = 0
    n_docs: int = 0
    n_python: int = 0
    n_csv: int = 0
    n_json: int = 0
    has_metrics_json: bool = False
    final_message_chars: int = 0


def _iter_events(path: Path):
    """Yield (event, None) for parsed lines and (None, raw) for bad ones."""
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line), None
            except json.JSONDecodeError:
                yield None, line


INDIRECT = re.compile(r"\$[{(]?\w|\b(?:bash|sh|zsh|source|eval|xargs)\b|python3?\s+\S+\.py")


def _classify_bash_read(cmd: str) -> tuple[str | None, str]:
    """Return ('yes'|'possible'|'indeterminate'|None, evidence) for an AGENTS.md read.

    A lexical classifier: ``yes`` when a read-capable utility names the file
    as an operand, ``possible`` for a glob that could match it,
    ``indeterminate`` when the command reaches the file system through a
    variable, a script or an interpreter whose arguments cannot be resolved,
    ``None`` otherwise. It is a warning signal, not proof.
    """
    if re.search(r"(?:open|read_text|read_bytes|Path)\(\s*['\"][^'\"]*AGENTS\.md", cmd):
        return "yes", cmd[:200]
    for seg in re.split(r"\s*(?:&&|\|\||;|\|)\s*", cmd):
        try:
            toks = shlex.split(seg)
        except ValueError:
            toks = seg.split()
        toks = [t for t in toks if "=" not in t or not re.match(r"^[A-Za-z_]\w*=", t)]
        if not toks:
            continue
        util = Path(toks[0]).name
        if util in ("sudo", "env", "nice", "time") and len(toks) > 1:
            util, toks = Path(toks[1]).name, toks[1:]
        if util not in READ_UTILS:
            continue
        args = [a for a in toks[1:] if not a.startswith("-")]
        if util in ("grep", "rg", "egrep", "fgrep") and args:
            args = args[1:]  # the first operand is the pattern, not a file
        for a in args:
            if Path(a).name == "AGENTS.md" and not a.startswith("/tmp"):
                return "yes", seg[:200]
        for a in args:
            if any(ch in a for ch in "*?[") and (a.endswith(".md") or Path(a).name in ("*", "*.*")):
                return "possible", seg[:200]
    if re.search(r"for\s+\w+\s+in\s+[^;]*\*\.md", cmd):
        return "possible", cmd[:200]
    if "AGENTS" in cmd or INDIRECT.search(cmd):
        return "indeterminate", cmd[:200]
    return None, ""


def _is_impl_bash(cmd: str) -> bool:
    return bool(IMPL_BASH.search(cmd))


def _outside_sandbox(text: str, sandbox_root: Path | None) -> bool:
    """True if a command or path reaches above the sandbox root.

    The work dir sits at <sandbox_root>/runs/<token>, so two `..` levels are
    inside the sandbox; three or more leave it. Absolute paths count when
    they are outside the sandbox root and not a system location.
    """
    if re.search(r"(?<![\w.])(?:\.\./){2,}\.\.(?![\w.])", text):
        return True
    for m in re.finditer(r"(?<![\w./])/(?:Users|home)/[^\s'\"`;|&)]+", text):
        p = m.group(0)
        if sandbox_root and p.startswith(str(sandbox_root)):
            continue
        if re.search(r"/(?:\.claude|\.local|\.cache|Library|opt|usr)/", p):
            continue
        return True
    if re.search(r"(?<![\w./])~/", text) or re.search(r"\$HOME/", text):
        return True
    return False


def parse_attempt(path: Path, work_dir: Path, sandbox_root: Path | None
                  ) -> tuple[Attempt, str, list[tuple[str, str, str]], list[str], list[str]]:
    """Parse one session log.

    Returns the Attempt, the final assistant text, a list of
    (kind, turn_label, evidence) AGENTS.md observations, the list of turn
    labels at which an implementation action happened, and the list of
    commands that reached outside the sandbox.
    """
    a = Attempt(file=path.name)
    tool_counter: Counter[str] = Counter()
    final_text = ""
    agents_obs: list[tuple[str, str, str]] = []
    impl_turns: list[str] = []
    outside: list[str] = []
    turn = 0
    for ev, bad in _iter_events(path):
        if bad is not None:
            a.parse_failures += 1
            continue
        assert ev is not None
        t = ev.get("type")
        if t == "system":
            sub = ev.get("subtype")
            if sub == "init":
                a.session_id = ev.get("session_id", "")
                a.model = ev.get("model", "")
            elif sub not in ("thinking_tokens", "task_started", "task_notification", "task_progress"):
                a.notable_events.append(f"system:{sub}:{str(ev)[:160]}")
        elif t == "assistant":
            turn += 1
            a.assistant_messages = turn
            label = f"t{turn}"
            for block in (ev.get("message") or {}).get("content", []):
                btype = block.get("type")
                if btype == "tool_use":
                    name = block.get("name", "?")
                    tool_counter[name] += 1
                    inp = block.get("input") or {}
                    if name == "Bash":
                        cmd = str(inp.get("command", ""))
                        a.bash_commands.append(cmd)
                        kind, ev_text = _classify_bash_read(cmd)
                        if kind:
                            agents_obs.append((kind, label, ev_text))
                        if _is_impl_bash(cmd):
                            impl_turns.append(label)
                        if _outside_sandbox(cmd, sandbox_root):
                            outside.append(f"{label} Bash: {cmd[:160]}")
                    elif name in ("Read", "Glob", "Grep"):
                        fp = str(inp.get("file_path") or inp.get("path") or "")
                        if name == "Read" and Path(fp).name == "AGENTS.md":
                            resolved = Path(fp) if Path(fp).is_absolute() else work_dir / fp
                            if resolved == work_dir / "AGENTS.md":
                                agents_obs.append(("yes", label, f"Read {fp}"))
                            else:
                                agents_obs.append(("possible", label, f"Read {fp} (not the work dir copy)"))
                        if fp and _outside_sandbox(fp, sandbox_root):
                            outside.append(f"{label} {name}: {fp[:160]}")
                    elif name in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
                        impl_turns.append(label)
                        fp = str(inp.get("file_path", ""))
                        if fp and _outside_sandbox(fp, sandbox_root):
                            outside.append(f"{label} {name}: {fp[:160]}")
                elif btype == "text":
                    final_text = block.get("text", "") or final_text
        elif t == "user":
            for block in (ev.get("message") or {}).get("content", []) or []:
                if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("is_error"):
                    a.tool_errors += 1
        elif t == "rate_limit_event":
            info = ev.get("rate_limit_info") or {}
            status = info.get("status")
            util = ((info.get("unifiedWindows") or {}).get("five_hour") or {}).get("utilization")
            if isinstance(util, (int, float)):
                a.rate_limit_max_5h = max(a.rate_limit_max_5h, float(util))
            if status not in (None, "allowed"):
                a.notable_events.append(f"rate_limit:{status}:{json.dumps(info)[:200]}")
        elif t == "result":
            a.has_result = True
            a.stop = ev.get("subtype", "") or "result"
            a.is_error = bool(ev.get("is_error", False))
            a.num_turns = int(ev.get("num_turns", 0))
            a.duration_s = round(float(ev.get("duration_ms", 0)) / 1000.0, 1)
            a.cost_usd = round(float(ev.get("total_cost_usd", 0.0)), 4)
            usage = ev.get("usage") or {}
            a.input_tokens = int(usage.get("input_tokens", 0))
            a.output_tokens = int(usage.get("output_tokens", 0))
            a.cache_read_tokens = int(usage.get("cache_read_input_tokens", 0))
            a.cache_create_tokens = int(usage.get("cache_creation_input_tokens", 0))
            denials = ev.get("permission_denials") or []
            a.permission_denials = len(denials)
            a.denied_tools = [str(d.get("tool_name", d)) if isinstance(d, dict) else str(d)
                              for d in denials]
            if a.is_error or a.stop not in ("success",):
                a.notable_events.append(f"result:{a.stop}:{str(ev.get('result', ''))[:200]}")
            if ev.get("result"):
                final_text = str(ev["result"])
        elif t not in ("assistant", "user", "rate_limit_event", "tool_progress"):
            a.notable_events.append(f"{t}:{str(ev)[:120]}")
    a.tool_counts = dict(tool_counter)
    return a, final_text, agents_obs, impl_turns, outside


def _manifest(run_dir: Path) -> list[dict[str, object]]:
    """Everything the agent left behind: output/ (the work dir, minus the
    data file and AGENTS.md) and output_outside/ (files written elsewhere in
    the sandbox)."""
    rows: list[dict[str, object]] = []
    for sub in ("output", "output_outside"):
        base = run_dir / sub
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            rel = p.relative_to(base)
            if any(part in SKIP_DIRS for part in rel.parts):
                continue
            if not p.is_file():
                continue
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            rows.append({"path": (str(rel) if sub == "output" else f"[outside] {rel}"),
                         "bytes": p.stat().st_size, "sha256": digest})
    return rows


def summarise(run_dir: Path) -> RunSummary:
    s = RunSummary(run=run_dir.name)
    meta_path = run_dir / "run_meta.json"
    work_dir = run_dir
    sandbox_root: Path | None = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        s.model_tag = meta.get("model_tag", "")
        s.level = meta.get("level", "")
        s.rep = int(meta.get("rep", 0))
        s.wallclock_s = int(meta.get("wallclock_seconds", 0))
        s.exit_code = meta.get("exit_code")
        s.status = meta.get("status", "")
        s.resumed = bool(meta.get("resumed", False))
        s.claude_code_version = meta.get("claude_code_version", "")
        if meta.get("work_dir"):
            work_dir = Path(meta["work_dir"])
        if meta.get("sandbox_root"):
            sandbox_root = Path(meta["sandbox_root"])
        for flag, label in (("data_unchanged", "data file modified"),
                            ("agents_md_unchanged", "AGENTS.md modified or deleted")):
            if flag in meta and not meta[flag]:
                s.notable_events.append(f"meta:{label}")
        if int(meta.get("venv_files_modified", 0)) > 0:
            s.notable_events.append(f"meta:{meta['venv_files_modified']} venv files modified")

    logs = sorted(run_dir.glob("session*.jsonl"))
    final_text = ""
    all_obs: list[tuple[str, str, str]] = []
    all_impl: list[str] = []
    tool_counter: Counter[str] = Counter()
    for i, log in enumerate(logs, start=1):
        a, text, obs, impl, outside = parse_attempt(log, work_dir, sandbox_root)
        s.attempts.append(a)
        final_text = text or final_text
        all_obs.extend((k, f"a{i}:{t}", e) for k, t, e in obs)
        all_impl.extend(f"a{i}:{t}" for t in impl)
        s.outside_sandbox_refs += len(outside)
        s.outside_sandbox_evidence.extend(f"a{i}:{o}" for o in outside)
        tool_counter.update(a.tool_counts)
        s.num_turns += a.num_turns
        s.cost_usd = round(s.cost_usd + a.cost_usd, 4)
        s.input_tokens += a.input_tokens
        s.output_tokens += a.output_tokens
        s.cache_read_tokens += a.cache_read_tokens
        s.cache_create_tokens += a.cache_create_tokens
        s.tool_errors += a.tool_errors
        s.parse_failures += a.parse_failures
        s.permission_denials += a.permission_denials
        s.denied_tools.extend(a.denied_tools)
        s.rate_limit_max_5h = max(s.rate_limit_max_5h, a.rate_limit_max_5h)
        s.notable_events.extend(f"{a.file}:{e}" for e in a.notable_events)
    s.tool_counts = dict(tool_counter)
    if s.attempts:
        # init-event facts come from the first attempt; the terminal state from the last
        first_log = logs[0]
        for ev, _bad in _iter_events(first_log):
            if ev and ev.get("type") == "system" and ev.get("subtype") == "init":
                s.model = ev.get("model", "")
                s.permission_mode = ev.get("permissionMode", "")
                s.mcp_servers = [m.get("name", "") for m in ev.get("mcp_servers", [])
                                 if m.get("status") == "connected"]
                break
        last = s.attempts[-1]
        s.has_result, s.stop, s.is_error = last.has_result, last.stop, last.is_error

    confirmed = [o for o in all_obs if o[0] == "yes"]
    possible = [o for o in all_obs if o[0] == "possible"]
    indeterminate = [o for o in all_obs if o[0] == "indeterminate"]
    if confirmed:
        s.agents_md_read = "yes"
        s.agents_md_first_turn = confirmed[0][1]
        s.agents_md_evidence = confirmed[0][2]
    elif possible:
        s.agents_md_read = "possible"
        s.agents_md_first_turn = possible[0][1]
        s.agents_md_evidence = possible[0][2]
    elif indeterminate:
        s.agents_md_read = "indeterminate"
        s.agents_md_first_turn = indeterminate[0][1]
        s.agents_md_evidence = indeterminate[0][2]
    if s.level == "L1":
        s.agents_md_read = "n/a (no AGENTS.md)"
    s.first_impl_turn = all_impl[0] if all_impl else None
    if s.agents_md_first_turn and s.first_impl_turn:
        def _key(label: str) -> tuple[int, int]:
            m = re.match(r"a(\d+):t(\d+)", label)
            return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
        s.agents_md_before_impl = _key(s.agents_md_first_turn) < _key(s.first_impl_turn)

    if final_text:
        (run_dir / "final_message.md").write_text(final_text, encoding="utf-8")
        s.final_message_chars = len(final_text)

    manifest = _manifest(run_dir)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    s.n_files = len(manifest)
    for row in manifest:
        ext = Path(str(row["path"])).suffix.lower()
        if ext == ".png":
            s.n_png += 1
        elif ext in IMG_EXT:
            s.n_img_other += 1
        elif ext in (".html", ".htm"):
            s.n_html += 1
        elif ext in DOC_EXT:
            s.n_docs += 1
        elif ext == ".py":
            s.n_python += 1
        elif ext == ".csv":
            s.n_csv += 1
        elif ext == ".json":
            s.n_json += 1
    s.has_metrics_json = (run_dir / "output" / "metrics.json").exists()
    (run_dir / "summary.json").write_text(json.dumps(asdict(s), indent=2))
    return s


def main() -> int:
    if not RUNS.exists():
        print(f"no {RUNS}", file=sys.stderr)
        return 1
    rows: list[RunSummary] = []
    for run_dir in sorted(RUNS.iterdir()):
        if run_dir.is_dir() and not run_dir.name.startswith("_") and (run_dir / "session.jsonl").exists():
            rows.append(summarise(run_dir))
    (RUNS / "summary.json").write_text(json.dumps([asdict(r) for r in rows], indent=2))

    hdr = ("| run | model | status | stop | turns | tokens in/out (k) | cost $ | wall min | py | png "
           "| img+ | docs | csv | metrics.json | AGENTS.md read (turn / first impl) | outside refs "
           "| denials | tool errs | bad lines | 5h util max |")
    sep = "|" + "---|" * 20
    lines = [hdr, sep]
    for r in rows:
        tok_in = (r.input_tokens + r.cache_read_tokens + r.cache_create_tokens) / 1000
        agents = r.agents_md_read
        if r.agents_md_first_turn:
            agents += f" ({r.agents_md_first_turn} / {r.first_impl_turn})"
        lines.append(
            f"| {r.run} | {r.model} | {r.status}{' resumed' if r.resumed else ''} "
            f"| {r.stop}{' ERR' if r.is_error else ''} | {r.num_turns} "
            f"| {tok_in:.0f}/{r.output_tokens / 1000:.1f} | {r.cost_usd:.2f} | {round(r.wallclock_s / 60)} "
            f"| {r.n_python} | {r.n_png} | {r.n_img_other + r.n_html} | {r.n_docs} | {r.n_csv} "
            f"| {r.has_metrics_json} | {agents} | {r.outside_sandbox_refs} | {r.permission_denials} "
            f"| {r.tool_errors} | {r.parse_failures} | {r.rate_limit_max_5h:.2f} |"
        )
    (RUNS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    for r in rows:
        for e in r.notable_events:
            print(f"  [{r.run}] {e}")
        for e in r.outside_sandbox_evidence:
            print(f"  [{r.run}] outside-sandbox: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
