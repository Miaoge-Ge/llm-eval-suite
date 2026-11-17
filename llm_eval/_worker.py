"""Trusted bootstrap, sent to the isolated interpreter via -c.

The parent attaches OS process limits before releasing the stdin envelope.
The receipt detects accidental early exits, not deliberate grader tampering.
"""

import json
import os
import runpy
import sys


def main():
    envelope = json.loads(sys.stdin.buffer.readline())
    limits = envelope["limits"]
    if sys.platform != "win32":
        import resource

        memory = limits["memory_mb"] * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        size = limits["tmpfs_mb"] * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (size, size))
        resource.setrlimit(resource.RLIMIT_CPU, (envelope["timeout"], envelope["timeout"] + 1))
        # Docker enforces the aggregate process limit through cgroups. For
        # explicit local mode NPROC is per user and cannot provide that guarantee.

    with open("program.py", "w", encoding="utf-8") as handle:
        handle.write(envelope["code"])
    with open("stdin.txt", "wb") as handle:
        handle.write(envelope["stdin"].encode("utf-8"))
    # Replace fd 0 as well as the Python wrapper: competitive-programming
    # candidates may use os.read(0, ...) or open(0), not just input().
    with open("stdin.txt", "rb") as source:
        os.dup2(source.fileno(), 0)
    with open(0, encoding="utf-8", closefd=False) as stdin:
        sys.stdin = stdin
        try:
            runpy.run_path("program.py", run_name="__main__")
        except SystemExit as exc:
            if envelope["assertions"] or exc.code not in (None, 0):
                raise
    sys.stderr.buffer.write(envelope["receipt"].encode("utf-8"))
    sys.stderr.buffer.flush()


if __name__ == "__main__":
    main()
