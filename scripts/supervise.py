"""Run a command in its own process group under a wall-clock cap.

The child gets a new session, so every process it spawns that does not
itself change process group shares one group. Its stdout and stderr are
captured through anonymous pipes into ``session.jsonl`` and ``stderr.log``
in the supervisor's current directory, so the child never holds a file
descriptor whose path names the run. Its stdin is the file named on the
command line and its working directory is the one named on the command
line; nothing in the child's argv, fds or cwd names the run.

On timeout, or if the supervisor receives SIGTERM or SIGINT, the whole
group gets SIGTERM, then SIGKILL after a 30 s grace period. Afterwards the
supervisor checks whether anything in the group survived and records that
in ``supervise_status.json`` beside the logs.

Exit status: the child's own status; 124 on timeout; 143 when the
supervisor was told to stop.

Usage:
    python3 supervise.py <cap-seconds> <stdin-file> <child-cwd> -- <cmd> [args...]
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time

GRACE_SECONDS = 30.0


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _kill_group(pgid: int, proc: subprocess.Popen[bytes]) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        except PermissionError:
            print(f"supervise: cannot send {sig.name} to group {pgid}", file=sys.stderr)
            return
        deadline = time.monotonic() + GRACE_SECONDS
        while time.monotonic() < deadline:
            proc.poll()  # reap the group leader so a zombie does not read as alive
            if not _group_alive(pgid):
                return
            time.sleep(0.5)


def _pump(src, dst_path: str) -> None:
    # read1 returns as soon as any data is available, so the log file is
    # current while the session runs (read would wait for a full 64 KiB)
    with open(dst_path, "ab", buffering=0) as dst:
        for chunk in iter(lambda: src.read1(65536), b""):
            dst.write(chunk)


def main(argv: list[str]) -> int:
    if len(argv) < 5 or argv[3] != "--":
        print(__doc__, file=sys.stderr)
        return 2
    cap = int(argv[0])
    stdin_path = argv[1]
    child_cwd = argv[2]
    cmd = argv[4:]
    started = time.time()
    with open(stdin_path, "rb") as fin:
        proc = subprocess.Popen(cmd, stdin=fin, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                cwd=child_cwd, start_new_session=True)
    pgid = os.getpgid(proc.pid)
    pumps = [threading.Thread(target=_pump, args=(proc.stdout, "session.jsonl"), daemon=True),
             threading.Thread(target=_pump, args=(proc.stderr, "stderr.log"), daemon=True)]
    for t in pumps:
        t.start()
    state = {"timeout": False, "stopped": False}

    def _on_signal(signum: int, _frame: object) -> None:
        state["stopped"] = True
        _kill_group(pgid, proc)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    try:
        code = proc.wait(timeout=cap)
    except subprocess.TimeoutExpired:
        state["timeout"] = True
        _kill_group(pgid, proc)
        code = proc.wait()
    _kill_group(pgid, proc)  # reap any background helpers the agent left in the group
    for t in pumps:
        t.join(timeout=10)
    survivors = _group_alive(pgid)
    if state["timeout"]:
        exit_code = 124
    elif state["stopped"]:
        exit_code = 143
    else:
        exit_code = int(code)
    with open("supervise_status.json", "w") as fh:
        json.dump({"child_exit": int(code), "exit": exit_code, "timeout": state["timeout"],
                   "stopped": state["stopped"], "group_survivors_after_kill": survivors,
                   "seconds": round(time.time() - started, 1)}, fh, indent=2)
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
