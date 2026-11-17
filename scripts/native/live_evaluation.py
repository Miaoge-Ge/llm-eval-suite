"""Run the fixed real-model suites through the production C++ HTTP service."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/native-live"))
    parser.add_argument("--port", type=int, default=18081)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    password = secrets.token_urlsafe(32)
    environment = dict(
        os.environ,
        LLM_EVAL_ROOT=str(root),
        LLM_EVAL_STATE_DIR=str(output),
        LLM_EVAL_PYTHON=sys.executable,
        LLM_EVAL_PORT=str(args.port),
        LLM_EVAL_ADMIN_PASSWORD=password,
        LLM_EVAL_SESSION_SECRET=secrets.token_hex(32),
    )
    if not environment.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY is required")
    failures = 0
    with (output / "service.log").open("w") as log:
        process = subprocess.Popen([str(args.binary.resolve())], cwd=root, env=environment, stdout=log, stderr=log)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=60) as client:
                for _ in range(150):
                    try:
                        response = client.get("/api/health")
                        if response.status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if process.poll() is not None:
                        raise RuntimeError("Native service exited before startup")
                    time.sleep(0.2)
                client.post("/api/login", json={"password": password}).raise_for_status()
                for mode, count in (("api", 30), ("benchmark", 40)):
                    response = client.post(
                        "/api/jobs",
                        json={
                            "mode": mode,
                            "task": "quick40",
                            "workers": 10,
                            "max_output_tokens": 16384,
                            "thinking_enabled": True,
                        },
                    )
                    response.raise_for_status()
                    identifier = response.json()["id"]
                    deadline = time.monotonic() + 1800
                    while time.monotonic() < deadline:
                        response = client.get(f"/api/jobs/{identifier}")
                        response.raise_for_status()
                        row = response.json()
                        if row["status"] not in {"QUEUED", "RUNNING", "CANCELLING"}:
                            break
                        time.sleep(2)
                    else:
                        raise RuntimeError("Native evaluation exceeded 30 minutes")
                    (output / f"{mode}.json").write_text(
                        json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    summary = row.get("summary", {})
                    print(
                        json.dumps(
                            {
                                "mode": mode,
                                "id": identifier,
                                "status": row["status"],
                                "cases": summary.get("total_cases"),
                                "attempts": summary.get("api_attempts"),
                                "status_counts": summary.get("status_counts"),
                            }
                        ),
                        flush=True,
                    )
                    if row["status"] != "COMPLETED" or summary.get("total_cases") != count:
                        failures += 1
                    if mode == "api" and summary.get("api_attempts") != 30:
                        failures += 1
        finally:
            process.terminate()
            process.wait(timeout=150)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
