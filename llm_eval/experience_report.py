"""Readable client-observed API experience metrics, separated from answer accuracy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .reporting import RunSummary
    from .settings import FrameworkConfig


def display(value: Any) -> str:
    return "n/a" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def render_experience(summary: RunSummary, config: FrameworkConfig, results: list[dict[str, Any]]) -> str:
    metrics = summary.metrics
    overall = metrics.get("overall", {})
    lines = [
        "# API experience — 30 requests",
        "",
        f"Model: `{summary.model}`; endpoint: `{config.model.base_url}`.",
        f"Status: **{summary.run_status}**; stop reason: {summary.error or 'none'}.",
        f"Thinking requested: {config.run.thinking_enabled}; effort: {config.run.reasoning_display}.",
        f"Temperature: {config.run.temperature}; seed: {config.run.seed}; timeout: {config.model.timeout_seconds}s.",
        "Concurrency: **1**; automatic retries: **0**; warmup: **none**.",
        f"Attempted/planned: {overall.get('attempted', 0)}/30; successful streams: {overall.get('successful', 0)}; "
        f"truncated: {overall.get('truncated', 0)}.",
        "",
        "Success means a nonempty answer with a stream finish marker; it does not measure answer correctness. "
        "Length-limited responses are counted as completed streams and separately marked truncated.",
        "All times are measured at the client, including network and channel delays. TTFT is the first "
        "non-whitespace content or reasoning; first body excludes reasoning. Empty role packets do not count.",
        "",
        "## Scenario metrics",
        "",
        "Each scenario has only 10 requests. P95 is an interpolated sample statistic, not a reliable tail SLA. "
        "Latency rows include observed partial failures; missing events have no latency value. "
        "summary.json also contains success-only distributions and each distribution's sample count.",
        "",
        "| Scenario | Attempted | Successful | First-attempt success | Truncated | Stalls | SLO pass |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, group in metrics.get("scenarios", {}).items():
        lines.append(
            f"| {name} | {group['attempted']}/10 | {group['successful']} | "
            f"{display(group['first_attempt_success_rate'])} | {group['truncated']} | "
            f"{group['stalls']} | {display(group['slo_pass_rate'])} |"
        )
    lines += ["", "| Scenario | Metric | n | P10 | P50 | P95 | Max |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, group in metrics.get("scenarios", {}).items():
        for metric, stats in group["observed"].items():
            lines.append(
                f"| {name} | {metric} | "
                + " | ".join(display(stats[k]) for k in ("n", "p10", "p50", "p95", "max"))
                + " |"
            )
    lines += [
        "",
        "## Measurement rules",
        "",
        "TPS is an estimate: (provider visible completion tokens − 1) / (last body chunk − first body chunk). "
        "Chunks may batch multiple tokens, so this is not exact token-by-token decoding speed. "
        "Reported reasoning tokens are subtracted. Unknown reasoning token counts, missing usage, "
        "one body chunk, or zero generation window produce n/a. SSE chunks are never counted as tokens.",
        "Character speed excludes the first body chunk. Max gap measures consecutive body chunks; "
        "a single body chunk has no measurable gap. E2E ends when the stream closes, including final usage packets.",
        f"Stall threshold: {config.run.stall_threshold_seconds}s. SLO thresholds: "
        f"first body ≤ {config.run.slo_ttft_seconds}s; TPS ≥ {config.run.slo_tps}; "
        f"gap ≤ {config.run.slo_max_gap_seconds}s. Unconfigured SLO is n/a. "
        "Configured SLO requires stop completion and every configured metric to be measurable and pass.",
        "Long input is approximately 4K cl100k_base content tokens; actual provider tokenization and "
        "chat framing differ. Prompts are fixed and distinct, but provider prefix caching may still occur. "
        "Reported cached tokens are preserved; missing cache usage is unknown, not zero.",
        "Compare the same suite, model, thinking settings and token caps from the same client/network. "
        "Repeat channels in alternating order across time periods; this report describes one session.",
        "",
        f"Error categories: {overall.get('errors', {})}",
        f"Returned model labels (provider claims, not identity verification): {overall.get('response_models', [])}",
        "",
        "## Requests",
        "",
        "| Case | Status | TTFT s | First body s | E2E s | TPS estimate | Max gap s | Finish |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in sorted(results, key=lambda row: row.get("case_index", 0)):
        row = result.get("metadata", {}).get("experience", {})
        values = [
            row.get(k)
            for k in (
                "first_content_seconds",
                "first_body_seconds",
                "end_to_end_seconds",
                "output_tps_estimate",
                "max_gap_seconds",
                "finish_reason",
            )
        ]
        lines.append(f"| {result['case_id']} | {result['status']} | " + " | ".join(display(v) for v in values) + " |")
    lines += [
        "",
        "Raw answers, reasoning, per-chunk timelines, usage and request parameters are saved in generations/. "
        "results.jsonl contains per-request metrics; summary.json contains distributions. Offline grade "
        "recomputes metrics from saved timelines without making API calls.",
    ]
    return "\n".join(lines) + "\n"
