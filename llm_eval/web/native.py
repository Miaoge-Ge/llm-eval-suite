"""Launch the compiled production service from the existing console command."""

import os
from pathlib import Path

from llm_eval.core.utils import repo_root


def main() -> None:
    root = Path(os.getenv("LLM_EVAL_ROOT", str(repo_root()))).resolve()
    binary = Path(os.getenv("LLM_EVAL_NATIVE_BINARY", str(root / "build/server/llm-eval-server")))
    if os.name != "posix" or not binary.is_file():
        raise SystemExit(
            "Build the C++ service on Linux first; see docs/deployment.md. The CLI remains cross-platform."
        )
    os.environ.setdefault("LLM_EVAL_ROOT", str(root))
    os.execv(str(binary.resolve()), [str(binary.resolve())])
