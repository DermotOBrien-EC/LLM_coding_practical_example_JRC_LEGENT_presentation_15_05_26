"""Filesystem inventories of a sandbox, before and after a run, and their diff.

    python3 inventory.py snapshot <root> <out.json> [--stat-only <relpath> ...]
    python3 inventory.py diff <before.json> <after.json> <out.json>

A snapshot records every entry under <root> (directories, files, symlinks,
other nodes): relative path, node type, mode, size, mtime in ns, symlink
target, and for regular files the SHA-256, except under the ``--stat-only``
subtrees (the cloned venv, tens of thousands of files) where size and
mtime stand in for the digest. Symlinks are never followed.

A diff lists added, modified and deleted entries, so the harvest can copy
exactly what changed and record deletions that a snapshot copy would miss.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(root: str, out: str, stat_only: list[str]) -> int:
    root = os.path.abspath(root)
    stat_only_abs = [os.path.join(root, p.rstrip("/")) for p in stat_only]
    rows: dict[str, dict[str, object]] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in list(dirnames) + filenames:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            try:
                st = os.lstat(full)
            except OSError as exc:
                rows[rel] = {"type": "unreadable", "error": str(exc)}
                continue
            entry: dict[str, object] = {
                "type": ("symlink" if stat.S_ISLNK(st.st_mode) else "dir" if stat.S_ISDIR(st.st_mode)
                         else "file" if stat.S_ISREG(st.st_mode) else "other"),
                "mode": oct(stat.S_IMODE(st.st_mode)),
                "size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
            }
            if entry["type"] == "symlink":
                try:
                    entry["target"] = os.readlink(full)
                except OSError as exc:
                    entry["target"] = f"<unreadable: {exc}>"
            elif entry["type"] == "file":
                if any(full == s or full.startswith(s + os.sep) for s in stat_only_abs):
                    entry["sha256"] = None
                else:
                    try:
                        entry["sha256"] = _sha256(full)
                    except OSError as exc:
                        entry["sha256"] = f"<unreadable: {exc}>"
            rows[rel] = entry
    with open(out, "w") as fh:
        json.dump({"root": root, "stat_only": stat_only, "entries": rows}, fh)
    print(f"inventory: {len(rows)} entries under {root}")
    return 0


def _changed(a: dict[str, object], b: dict[str, object]) -> bool:
    if a.get("type") != b.get("type"):
        return True
    if a.get("type") == "symlink":
        return a.get("target") != b.get("target")
    if a.get("type") == "file":
        if a.get("sha256") is not None and b.get("sha256") is not None:
            return a["sha256"] != b["sha256"]
        return (a.get("size"), a.get("mtime_ns")) != (b.get("size"), b.get("mtime_ns"))
    return False


def diff(before: str, after: str, out: str) -> int:
    a = json.load(open(before))["entries"]
    b = json.load(open(after))["entries"]
    added = sorted(p for p in b if p not in a)
    deleted = sorted(p for p in a if p not in b)
    modified = sorted(p for p in b if p in a and _changed(a[p], b[p]))
    result = {"added": added, "modified": modified, "deleted": deleted,
              "counts": {"added": len(added), "modified": len(modified), "deleted": len(deleted)}}
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"diff: +{len(added)} ~{len(modified)} -{len(deleted)}")
    return 0


def harvest(changes_path: str, ws: str, work_rel: str, run_dir: str, keep_names: list[str]) -> int:
    """Copy every added or modified entry into the run directory.

    Entries under <work_rel> go to <run_dir>/output/, everything else to
    <run_dir>/output_outside/ (the venv subtree is counted, not copied).
    Harness inputs (``keep_names``, e.g. the data file and AGENTS.md) are
    copied only when the diff says they changed, as evidence.
    """
    import shutil
    ch = json.load(open(changes_path))
    out_dir = os.path.join(run_dir, "output")
    outside_dir = os.path.join(run_dir, "output_outside")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(outside_dir, exist_ok=True)
    venv_changes = 0
    copied = 0
    work_prefix = work_rel.rstrip("/") + "/"
    for rel in ch["added"] + ch["modified"]:
        if rel == ".venv" or rel.startswith(".venv/"):
            venv_changes += 1
            continue
        src = os.path.join(ws, rel)
        if rel.startswith(work_prefix):
            sub = rel[len(work_prefix):]
            if sub in keep_names and rel not in ch["modified"]:
                continue
            dst = os.path.join(out_dir, sub)
        else:
            dst = os.path.join(outside_dir, rel)
        if os.path.islink(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                os.symlink(os.readlink(src), dst)
            except FileExistsError:
                pass
            copied += 1
        elif os.path.isdir(src):
            os.makedirs(dst, exist_ok=True)
        elif os.path.isfile(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    for rel in ch["deleted"]:
        if rel.startswith(".venv/"):
            venv_changes += 1
    summary = {"copied": copied, "venv_entries_changed": venv_changes,
               "deleted": ch["deleted"], "counts": ch["counts"]}
    with open(os.path.join(run_dir, "harvest.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"harvest: copied {copied}, venv changes {venv_changes}, deleted {len(ch['deleted'])}")
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    if argv[0] == "harvest" and len(argv) >= 5:
        return harvest(argv[1], argv[2], argv[3], argv[4], argv[5:])
    if argv[0] == "snapshot":
        root, out = argv[1], argv[2]
        stat_only: list[str] = []
        rest = argv[3:]
        while rest:
            if rest[0] == "--stat-only" and len(rest) > 1:
                stat_only.append(rest[1])
                rest = rest[2:]
            else:
                print(f"unexpected argument: {rest[0]}", file=sys.stderr)
                return 2
        return snapshot(root, out, stat_only)
    if argv[0] == "diff" and len(argv) == 4:
        return diff(argv[1], argv[2], argv[3])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
