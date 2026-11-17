"""Run persistence and report rendering.

Each run gets its own directory under ``results/<model>/<task>/<timestamp>/``
containing:

- ``results.jsonl`` — one line per case, written incrementally so a crashed
  run keeps everything graded so far
- ``config.json`` — snapshot of the effective configuration (API key masked)
- ``report.md`` — the final human-readable report
- ``status.json`` — atomically updated lifecycle state
- ``summary.json`` — final machine-readable metrics
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO

from llm_eval.core.artifacts import RunLock, encode_json, write_atomic, write_json
from llm_eval.core.settings import FrameworkConfig
from llm_eval.core.status import ERROR_STATUSES, GRADED_STATUSES, RunStatus
from llm_eval.core.utils import slugify


@dataclass(frozen=True)
class RunSummary:
    task: str
    model: str
    total_cases: int
    completed_cases: int
    pass_rate: float
    wall_clock_seconds: float
    wall_clock_human: str
    average_case_seconds: float
    throughput_tokens_per_second: float
    status_counts: dict[str, int]
    token_usage: dict[str, int]
    run_status: str = RunStatus.COMPLETED
    error: str | None = None
    cancelled_cases: int = 0
    operation: str = "run"
    api_attempts: int | None = None
    new_api_attempts: int = 0
    new_token_usage: dict[str, int] = field(default_factory=dict)
    source_run: str | None = None
    incomplete_usage_cases: int | None = None
    test_coverage: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def coverage_rate(self) -> float:
        graded = sum(self.status_counts.get(status, 0) for status in GRADED_STATUSES)
        return graded / self.total_cases if self.total_cases else 0.0

    @property
    def error_rate(self) -> float:
        errors = sum(self.status_counts.get(status, 0) for status in ERROR_STATUSES)
        return errors / self.total_cases if self.total_cases else 0.0


@dataclass(frozen=True)
class RunPaths:
    root: Path
    results_jsonl: Path
    report_md: Path
    config_json: Path
    status_json: Path
    summary_json: Path


class ResultWriter:
    def __init__(self, config: FrameworkConfig, root: Path | None = None) -> None:
        self.config = config
        root = root or _allocate_run_dir(config)
        self.paths = RunPaths(
            root=root,
            results_jsonl=root / "results.jsonl",
            report_md=root / "report.md",
            config_json=root / "config.json",
            status_json=root / "status.json",
            summary_json=root / "summary.json",
        )
        self._results_handle: TextIO | None = None
        self._lock = RunLock(root)

    def __enter__(self) -> ResultWriter:
        self._lock.__enter__()
        return self

    def start(self, checkpoints: list[dict[str, Any]] | None = None) -> None:
        if not self.paths.config_json.exists():
            write_json(self.paths.config_json, self.config.to_public_dict())
        write_json(self.paths.root / "sessions" / f"{uuid.uuid4().hex}.json", self.config.to_public_dict())
        if checkpoints is not None:
            if self.paths.results_jsonl.exists():
                history = self.paths.root / "history"
                history.mkdir(exist_ok=True)
                shutil.copyfile(self.paths.results_jsonl, history / f"results-{uuid.uuid4().hex}.jsonl")
            self.finalize_results(checkpoints)
        self._results_handle = self.paths.results_jsonl.open("a", encoding="utf-8", newline="\n")
        self.write_state(RunStatus.RUNNING)

    def write_result(self, payload: dict[str, Any]) -> None:
        if self._results_handle is None:
            raise RuntimeError("ResultWriter must be entered before writing results")
        write_json(self.paths.root / "cases" / f"{payload['case_index']:08d}.json", payload)
        self._results_handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        self._results_handle.flush()
        os.fsync(self._results_handle.fileno())

    def finalize_results(self, results: list[dict[str, Any]]) -> None:
        if self._results_handle is not None:
            self._results_handle.close()
            self._results_handle = None
        ordered = sorted(results, key=lambda item: item["case_index"])
        write_atomic(self.paths.results_jsonl, "".join(encode_json(item) + "\n" for item in ordered))

    def write_report(self, content: str) -> None:
        write_atomic(self.paths.report_md, content)

    def write_state(self, status: str, **fields: Any) -> None:
        _write_json_atomic(self.paths.status_json, {"status": status, "updated_at": time.time(), **fields})

    def write_summary(self, summary: RunSummary) -> None:
        _write_json_atomic(
            self.paths.summary_json,
            {
                **asdict(summary),
                "coverage_rate": summary.coverage_rate,
                "error_rate": summary.error_rate,
            },
        )
        self.write_state(
            summary.run_status,
            total_cases=summary.total_cases,
            completed_cases=summary.completed_cases,
            cancelled_cases=summary.cancelled_cases,
            error=summary.error,
        )

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._results_handle is not None:
            self._results_handle.close()
            self._results_handle = None
        self._lock.__exit__(exc_type, exc_val, exc_tb)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def _allocate_run_dir(config: FrameworkConfig) -> Path:
    base = config.run.output_dir / slugify(config.run.task)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    candidate = base / stamp
    counter = 2
    while True:
        try:
            candidate.mkdir(parents=True)
            return candidate
        except FileExistsError:
            candidate = base / f"{stamp}-{counter}"
            counter += 1


def render_markdown_report(summary: RunSummary, config: FrameworkConfig, results: list[dict[str, Any]]) -> str:
    if summary.task == "api_experience":
        from llm_eval.api_testing.report import render_experience

        return render_experience(summary, config, results)
    from llm_eval.benchmark.report import render_benchmark

    return render_benchmark(summary, config, results)
