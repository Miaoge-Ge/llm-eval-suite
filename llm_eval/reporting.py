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

from .artifacts import RunLock, encode_json, write_atomic, write_json
from .experience_report import render_experience
from .settings import FrameworkConfig
from .status import ERROR_STATUSES, GRADED_STATUSES, RunStatus, Status
from .utils import natural_sort_key, slugify


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


def render_markdown_report(
    summary: RunSummary,
    config: FrameworkConfig,
    results: list[dict[str, Any]],
) -> str:
    if summary.task == "api_experience":
        return render_experience(summary, config, results)
    usage = summary.token_usage
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    graded = sum(summary.status_counts.get(status, 0) for status in GRADED_STATUSES)
    attempts = summary.api_attempts
    if attempts is None:
        attempts = sum(item.get("api_attempts", 0) for item in results)
    incomplete = summary.incomplete_usage_cases
    if incomplete is None:
        incomplete = sum(not item.get("usage_complete", True) for item in results)
    lines = [
        f"# Evaluation Report - {summary.task}",
        "",
        f"_Generated at {generated_at}_",
        "",
        "## Overview",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Task | `{summary.task}` |",
        f"| Operation | `{summary.operation}` |",
        f"| Source run | {_cell(summary.source_run or 'none')} |",
        f"| Model | `{summary.model}` |",
        f"| Dataset | `{config.dataset.path}` |",
        f"| Workers | `{config.run.workers}` |",
        f"| Thinking enabled | `{config.run.thinking_enabled}` |",
        f"| Reasoning effort | `{config.run.reasoning_display}` |",
        f"| Run status | **{summary.run_status}** |",
        f"| Execution backend (code tasks) | `{config.run.execution.backend}` |",
        f"| Execution image | `{config.run.execution.image}` |",
        f"| Execution timeout | {config.run.execution_timeout_seconds}s |",
        f"| Execution limits | memory={config.run.execution.memory_mb} MiB; "
        f"cpus={config.run.execution.cpus}; pids={config.run.execution.pids_limit}; "
        f"output={config.run.execution.max_output_bytes} bytes |",
        f"| Stop reason | {_cell(summary.error or 'none')} |",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Pass rate | **{summary.pass_rate:.2%}** |",
        f"| Total cases | {summary.total_cases} |",
        f"| Completed cases | {summary.completed_cases} |",
        f"| Cancelled cases | {summary.cancelled_cases} |",
        f"| Graded cases | {graded} |",
        f"| Grading coverage | {summary.coverage_rate:.2%} |",
        f"| Infrastructure error rate | {summary.error_rate:.2%} |",
        f"| Wall clock | {summary.wall_clock_human} |",
        f"| Average case time | {summary.average_case_seconds:.2f}s |",
        f"| Throughput | {summary.throughput_tokens_per_second:.1f} tokens/s |",
        f"| Prompt tokens | {usage.get('prompt_tokens', 0):,} |",
        f"| Completion tokens | {usage.get('completion_tokens', 0):,} |",
        f"| Total tokens | {usage.get('total_tokens', 0):,} |",
        f"| API attempts | {attempts} |",
        f"| API attempts this session | {summary.new_api_attempts} |",
        f"| Tokens this session | {summary.new_token_usage.get('total_tokens', 0)} |",
        f"| Cases with incomplete token usage | {incomplete} |",
        "",
        "Token totals include only usage returned by the provider; missing usage is not estimated.",
        "Pass rate uses graded cases only. Coverage and run status indicate whether the evaluation is complete.",
        "",
        "### Status counts",
        "",
        "| Status | Count |",
        "| --- | --- |",
    ]
    for status, count in sorted(summary.status_counts.items()):
        lines.append(f"| {status} | {count} |")

    if summary.test_coverage:
        scope = summary.test_coverage
        lines.extend(
            [
                "",
                "### Test coverage",
                "",
                f"Scope: **{scope['scope']}**. This is the stdin/stdout subset, not the full LiveCodeBench benchmark.",
                f"Public tests: {scope['public_tests']}; private tests: {scope['private_tests']}; "
                f"problems with private tests: {scope['problems_with_private_tests']}; "
                f"public-only problems: {scope['public_only_problems']}.",
            ]
        )
        if scope["public_only_problems"]:
            lines.append(
                "**Incomplete test coverage: the pass rate includes public-only samples and must not be "
                "reported as a full-test benchmark score.**"
            )
    if summary.task == "quick40":
        lines.extend(
            [
                "",
                "### Fixed 40-case coverage sample",
                "",
                "Original-language questions; ten per dimension. This is not a calibrated benchmark or "
                "a general model ranking. Each case changes its dimension count by one out of ten. "
                "IFEval checks explicit constraints, not subjective writing quality. "
                "Errors and cancellations are not silently counted as wrong answers.",
                "",
                "| Dimension | Passed / planned | Graded | Errors | Cancelled | Truncated |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for name, stats in summary.metrics.get("dimensions", {}).items():
            lines.append(
                f"| {name} | {stats['passed']}/{stats['planned']} | {stats['graded']} | "
                f"{stats['errors']} | {stats['cancelled']} | {stats['truncated']} |"
            )
    domain_rows = _domain_breakdown(results)
    if domain_rows:
        lines.extend(
            [
                "",
                "### Accuracy by domain",
                "",
                "| Domain | Passed | Total | Pass rate |",
                "| --- | --- | --- | --- |",
            ]
        )
        for domain, passed, total in domain_rows:
            rate = passed / total if total else 0.0
            lines.append(f"| {domain} | {passed} | {total} | {rate:.2%} |")

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| # | Case | Status | Time | Tokens | Detail |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    ordered = sorted(results, key=lambda r: natural_sort_key(str(r.get("case_id", ""))))
    for index, item in enumerate(ordered, start=1):
        lines.append(
            "| {idx} | {case} | {status} | {time} | {tokens} | {detail} |".format(
                idx=index,
                case=_cell(item.get("case_id", "unknown")),
                status=item.get("status", "unknown"),
                time=item.get("duration_human", "n/a"),
                tokens=item.get("total_tokens", 0),
                detail=_cell(_detail(item)),
            )
        )

    return "\n".join(lines).strip() + "\n"


def _domain_breakdown(results: list[dict[str, Any]]) -> list[tuple[str, int, int]]:
    stats: dict[str, list[int]] = {}
    for item in results:
        domain = (item.get("metadata") or {}).get("domain")
        if not domain or item.get("status") not in GRADED_STATUSES:
            continue
        entry = stats.setdefault(domain, [0, 0])
        entry[1] += 1
        if item.get("status") == Status.PASSED:
            entry[0] += 1
    return [(domain, passed, total) for domain, (passed, total) in sorted(stats.items())]


def _shorten(value: str, limit: int = 160) -> str:
    compact = " ".join(str(value).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _cell(value: Any) -> str:
    text = _shorten(str(value)).replace("|", "\\|")
    return text or "n/a"


def _detail(item: dict[str, Any]) -> str:
    if item.get("status") == Status.PASSED:
        return "passed"
    error = item.get("error")
    if error:
        return str(error)
    expected = item.get("expected")
    actual = item.get("actual")
    if expected is not None or actual is not None:
        return f"expected {expected} -> got {actual}"
    return "n/a"
