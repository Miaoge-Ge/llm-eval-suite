import pytest

from llm_eval.api_testing.metrics import distribution, request_metrics, summarize_experience
from llm_eval.core.timing import StreamTimeline


def measure(attempt, thinking=False, slo=None):
    return request_metrics(attempt, thinking=thinking, stall_threshold=1, slo=slo or {})


def timeline_attempt():
    timeline = StreamTimeline(100)
    timeline.observe(100.1)  # Empty role packet.
    timeline.observe(100.2, "  ")
    timeline.observe(101, reasoning="Let me think")
    timeline.observe(103, "hello")
    timeline.observe(105, " world")
    return {
        "timing": timeline.finish(108),
        "usage": {"completion_tokens": 13},
        "reasoning_tokens": 8,
        "finish_reason": "stop",
        "response_model": "test",
    }


def test_meaningful_content_body_reasoning_and_usage_tail_are_separate():
    row = measure(timeline_attempt(), thinking=True)
    assert row["first_content_seconds"] == row["first_reasoning_seconds"] == 1
    assert row["first_body_seconds"] == 3
    assert row["end_to_end_seconds"] == 8
    assert row["last_body_seconds"] == 5
    assert row["output_tps_estimate"] == 2  # (13 - 8 - 1) / (5 - 3), NOT chunks / elapsed.
    assert row["characters_per_second"] == 3
    assert row["max_gap_seconds"] == 2
    assert row["stall_count"] == 1
    assert row["success"]
    assert row["cached_prompt_tokens"] is None
    assert row["slo_pass"] is None


@pytest.mark.parametrize("reasoning_tokens", [None, -1, 20, True])
def test_unknown_or_invalid_reasoning_usage_does_not_invent_tps(reasoning_tokens):
    attempt = timeline_attempt()
    attempt["reasoning_tokens"] = reasoning_tokens
    assert measure(attempt)["output_tps_estimate"] is None


@pytest.mark.parametrize("kind", ["missing_usage", "one_chunk", "zero_window"])
def test_unmeasurable_speed_is_missing_not_zero(kind):
    attempt = timeline_attempt()
    if kind == "missing_usage":
        attempt["usage"] = {}
    elif kind == "one_chunk":
        attempt["timing"]["events"] = attempt["timing"]["events"][-1:]
    else:
        for event in attempt["timing"]["events"]:
            event["at_seconds"] = 1
    assert measure(attempt)["output_tps_estimate"] is None


def test_nonreasoning_usage_and_char_rate_have_distinct_sources():
    attempt = timeline_attempt()
    attempt.pop("reasoning_tokens")
    attempt["timing"]["events"] = attempt["timing"]["events"][-2:]
    row = measure(attempt)
    assert row["output_tps_estimate"] == 6
    assert row["visible_tokens"] == 13
    assert row["output_tps_source"] == "provider_completion_usage_no_reasoning_observed"
    assert measure(attempt, thinking=True)["output_tps_estimate"] is None


def test_partial_failure_and_truncation_are_not_hidden_by_latency_or_slo():
    attempt = timeline_attempt()
    assert measure(attempt, slo={"ttft_seconds": 3, "tps": 2, "max_gap_seconds": 2})["slo_pass"]
    assert not measure(attempt, slo={"ttft_seconds": 2})["slo_pass"]
    attempt["finish_reason"] = "length"
    row = measure(attempt, slo={"tps": 1})
    assert row["success"] and row["truncated"] and not row["slo_pass"]
    attempt.update(error="connection broke", error_type="IncompleteStream")
    failed = measure(attempt)
    assert not failed["success"]
    assert failed["error_category"] == "incomplete_stream"

    def result(data):
        return {"metadata": {"scenario": "short_answer", "experience": data}}

    summary = summarize_experience([result(row), result(failed), result({})])["scenarios"]["short_answer"]
    assert summary["planned"] == 3 and summary["attempted"] == 2
    assert summary["first_attempt_success_rate"] == 0.5
    assert summary["observed"]["first_body_seconds"]["n"] == 2
    assert summary["successful_latency"]["first_body_seconds"]["n"] == 1


def test_percentiles_and_empty_samples():
    assert distribution([]) == {"n": 0, "p10": None, "p50": None, "p95": None, "max": None}
    assert distribution([1, 3, float("nan")])["p95"] == pytest.approx(2.9)


def test_token_totals_include_failed_and_truncated_requests_without_inventing_missing_usage():
    rows = [
        {"attempted": True, "success": True, "prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        {"attempted": True, "success": False, "prompt_tokens": 5, "completion_tokens": 0},
        {"attempted": True, "success": True, "truncated": True, "completion_tokens": 7},
        {"attempted": True},
        {"attempted": False},
    ]
    metrics = summarize_experience([{"metadata": {"scenario": "short_answer", "experience": row}} for row in rows])
    usage = metrics["overall"]["token_usage"]
    assert usage["prompt_tokens"] == {"tokens": 15, "reported_requests": 2, "missing_requests": 2}
    assert usage["completion_tokens"] == {"tokens": 27, "reported_requests": 3, "missing_requests": 1}
    assert usage["total_tokens"] == {"tokens": 35, "reported_requests": 2, "missing_requests": 2}
    assert metrics["scenarios"]["short_answer"]["token_usage"] == usage


@pytest.mark.parametrize("value", [None, -1, True, "12", 1.5])
def test_invalid_token_counts_remain_unknown(value):
    metrics = summarize_experience(
        [
            {
                "metadata": {
                    "scenario": "short_answer",
                    "experience": {
                        "attempted": True,
                        "prompt_tokens": value,
                        "completion_tokens": value,
                        "total_tokens": value,
                    },
                }
            }
        ]
    )
    for item in metrics["overall"]["token_usage"].values():
        assert item == {"tokens": None, "reported_requests": 0, "missing_requests": 1}
