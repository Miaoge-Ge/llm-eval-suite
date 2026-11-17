"""Benchmark accuracy and coverage reports."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from llm_eval.core.status import GRADED_STATUSES, Status
from llm_eval.core.utils import natural_sort_key

if TYPE_CHECKING:
    from llm_eval.core.reporting import RunSummary
    from llm_eval.core.settings import FrameworkConfig


def render_benchmark(
    summary: RunSummary,
    config: FrameworkConfig,
    results: list[dict[str, Any]],
) -> str:
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
