"""Client-observed streaming timelines; never equate SSE chunks with tokens."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


class StreamTimeline:
    def __init__(self, started: float) -> None:
        self.started = started
        self.events: list[dict[str, Any]] = []

    def observe(self, now: float, content: str = "", reasoning: str = "") -> None:
        if content or reasoning:
            self.events.append({"at_seconds": max(0.0, now - self.started), "content": content, "reasoning": reasoning})

    def finish(self, now: float) -> dict[str, Any]:
        return {"duration_seconds": max(0.0, now - self.started), "events": self.events}


def request_metrics(
    attempt: dict[str, Any], *, thinking: bool, stall_threshold: float, slo: dict[str, float | None]
) -> dict[str, Any]:
    timeline = attempt.get("timing") or {}
    events = timeline.get("events", [])
    visible = [event for event in events if event["content"]]
    meaningful = [event for event in events if event["content"].strip() or event["reasoning"].strip()]
    body = [event for event in visible if event["content"].strip()]
    thought = [event for event in events if event["reasoning"].strip()]
    first_body = body[0]["at_seconds"] if body else None
    body_events = [event for event in visible if first_body is not None and event["at_seconds"] >= first_body]
    gaps = [b["at_seconds"] - a["at_seconds"] for a, b in zip(body_events, body_events[1:], strict=False)]
    window = body_events[-1]["at_seconds"] - first_body if body_events else 0.0
    chars = sum(len(event["content"]) for event in visible)
    usage = attempt.get("usage") or {}
    completion = usage.get("completion_tokens")
    reasoning_tokens = attempt.get("reasoning_tokens")
    visible_tokens = None
    token_source = "unavailable"
    if type(completion) is int and completion >= 0:
        if type(reasoning_tokens) is int and 0 <= reasoning_tokens <= completion:
            visible_tokens = completion - reasoning_tokens
            token_source = "provider_completion_minus_reported_reasoning"
        elif reasoning_tokens is None and not thinking and not thought:
            visible_tokens = completion
            token_source = "provider_completion_usage_no_reasoning_observed"
    tps = (
        (visible_tokens - 1) / window
        if (visible_tokens is not None and visible_tokens > 1 and window > 0 and len(body_events) > 1)
        else None
    )
    finish_reason = attempt.get("finish_reason")
    success = not attempt.get("error") and bool(body) and bool(finish_reason)
    duration = timeline.get("duration_seconds", attempt.get("duration_seconds"))
    metrics = {
        "attempted": True,
        "success": success,
        "first_content_seconds": meaningful[0]["at_seconds"] if meaningful else None,
        "first_body_seconds": first_body,
        "first_reasoning_seconds": thought[0]["at_seconds"] if thought else None,
        "end_to_end_seconds": duration,
        "last_body_seconds": body_events[-1]["at_seconds"] if body_events else None,
        "output_tps_estimate": tps,
        "output_tps_source": token_source,
        "visible_tokens": visible_tokens,
        "output_characters": chars,
        "characters_per_second": (sum(len(event["content"]) for event in body_events[1:]) / window)
        if window > 0
        else None,
        "visible_chunk_count": len(body_events),
        "max_gap_seconds": max(gaps) if gaps else None,
        "stall_count": sum(gap > stall_threshold for gap in gaps),
        "finish_reason": finish_reason,
        "truncated": finish_reason == "length",
        "response_model": attempt.get("response_model"),
        "http_status_code": attempt.get("http_status_code"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion,
        "reasoning_tokens": reasoning_tokens,
        "cached_prompt_tokens": attempt.get("cached_prompt_tokens"),
        "error_category": error_category(attempt) if not success else None,
    }
    enabled = any(value is not None for value in slo.values())
    checks = [success and finish_reason == "stop"]
    for name, measured, lower_is_better in (
        ("ttft_seconds", first_body, True),
        ("tps", tps, False),
        ("max_gap_seconds", metrics["max_gap_seconds"], True),
    ):
        threshold = slo.get(name)
        if threshold is not None:
            checks.append(
                measured is not None and (measured <= threshold if lower_is_better else measured >= threshold)
            )
    metrics["slo_pass"] = all(checks) if enabled else None
    return metrics


def error_category(attempt: dict[str, Any]) -> str:
    status = attempt.get("http_status_code")
    kind = attempt.get("error_type", "")
    message = str(attempt.get("error") or "").lower()
    if status in (401, 403):
        return "authentication"
    if status == 429:
        return "rate_limit"
    if isinstance(status, int) and status >= 500:
        return "server_error"
    if "timeout" in kind.lower() or "timed out" in message:
        return "timeout"
    if kind == "IncompleteStream":
        return "incomplete_stream"
    if "empty completion" in message:
        return "empty_answer"
    if "cancel" in message:
        return "cancelled"
    if isinstance(status, int) and status >= 400:
        return "http_client_error"
    return "transport_or_protocol"


def distribution(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(value for value in values if math.isfinite(value))

    def percentile(q: float) -> float | None:
        if not ordered:
            return None
        position = (len(ordered) - 1) * q
        left = int(position)
        right = min(left + 1, len(ordered) - 1)
        return ordered[left] + (ordered[right] - ordered[left]) * (position - left)

    return {
        "n": len(ordered),
        "p10": percentile(0.1),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "max": ordered[-1] if ordered else None,
    }


def summarize_experience(results: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {name: [] for name in ("short_answer", "long_answer", "long_context")}
    for result in results:
        meta = result.get("metadata", {})
        if meta.get("scenario") in groups:
            groups[meta["scenario"]].append(meta.get("experience", {}))

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        attempts = [row for row in rows if row.get("attempted")]
        successes = [row for row in attempts if row.get("success")]
        latency_names = (
            "first_content_seconds",
            "first_body_seconds",
            "first_reasoning_seconds",
            "end_to_end_seconds",
            "max_gap_seconds",
            "output_tps_estimate",
            "characters_per_second",
        )
        return {
            "planned": len(rows),
            "attempted": len(attempts),
            "successful": len(successes),
            "first_attempt_success_rate": len(successes) / len(attempts) if attempts else None,
            "truncated": sum(bool(row.get("truncated")) for row in attempts),
            "truncation_rate": sum(bool(row.get("truncated")) for row in attempts) / len(attempts)
            if attempts
            else None,
            "stalls": sum(row.get("stall_count", 0) for row in attempts),
            "slo_pass_rate": sum(bool(row.get("slo_pass")) for row in attempts) / len(attempts)
            if attempts and any(row.get("slo_pass") is not None for row in attempts)
            else None,
            "errors": dict(Counter(row.get("error_category", "unknown") for row in attempts if not row.get("success"))),
            "response_models": sorted({row["response_model"] for row in attempts if row.get("response_model")}),
            "observed": {
                name: distribution([row[name] for row in attempts if row.get(name) is not None])
                for name in latency_names
            },
            "successful_latency": {
                name: distribution([row[name] for row in successes if row.get(name) is not None])
                for name in latency_names
            },
        }

    return {
        "suite_version": "api-experience-v1",
        "concurrency": 1,
        "automatic_retries": 0,
        "scenarios": {name: aggregate(rows) for name, rows in groups.items()},
        "overall": aggregate([row for rows in groups.values() for row in rows]),
    }
