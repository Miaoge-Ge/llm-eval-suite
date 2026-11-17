"""The scoring boundary must preserve requests when scoring is interrupted."""

import json
from dataclasses import asdict

from llm_eval.core.clients import GenerationResult
from llm_eval.native_bridge import finish, prepare


def test_recover_ungraded_response_without_losing_attempts(tmp_path):
    root = tmp_path / "native"
    request = prepare(
        root,
        {
            "mode": "benchmark",
            "task": "gsm",
            "limit": 1,
            "base_url": "https://example.com/v1",
            "model": "test",
            "workers": 1,
            "max_output_tokens": 32,
            "thinking_enabled": False,
        },
    )["requests"][0]
    pending = root / "native_pending"
    pending.mkdir()
    generation = GenerationResult(
        content="42", usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}, attempts=1
    )
    (pending / "0.json").write_text(
        json.dumps({"index": 0, "request": request["body"], "generation": asdict(generation), "duration_seconds": 1})
    )
    result = finish(root, {"status": "FAILED", "error": "Scorer exited", "wall_clock_seconds": 2})
    assert result["summary"]["api_attempts"] == 1
    assert result["summary"]["token_usage"]["total_tokens"] == 4
    assert result["cases"][0]["status"] == "INTERNAL_ERROR"
    assert result["summary"]["cancelled_cases"] == 0
    again = finish(root, {"status": "FAILED", "error": "Scorer exited", "wall_clock_seconds": 2})
    assert again["summary"]["api_attempts"] == 1
