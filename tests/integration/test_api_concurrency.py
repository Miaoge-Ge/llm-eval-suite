import json
import threading
from dataclasses import replace

import httpx
from openai import OpenAI

from llm_eval.core.artifacts import load_manifest
from llm_eval.core.runner import EvaluationRunner
from llm_eval.core.settings import DEFAULT_DATASET_PATHS, DatasetConfig, config_from_snapshot
from llm_eval.core.utils import repo_root


def test_ten_workers_keep_exact_attempts_and_original_measurement_concurrency(fake_config, monkeypatch):
    config = replace(
        fake_config,
        dataset=DatasetConfig("api_experience", repo_root() / DEFAULT_DATASET_PATHS["api_experience"]),
        run=replace(fake_config.run, task="api_experience", workers=10),
    )
    lock = threading.Lock()
    barrier = threading.Barrier(10, timeout=10)
    calls = active = peak = 0

    def handler(request):
        nonlocal calls, active, peak
        with lock:
            calls += 1
            number = calls
            active += 1
            peak = max(peak, active)
        barrier.wait()
        with lock:
            active -= 1
        if number == 2:
            return httpx.Response(429, json={"error": {"message": "limited"}})
        chunks = [{"choices": [{"index": 0, "delta": {"content": "answer"}}]}]
        if number != 3:
            chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        data = b"".join(("data: " + json.dumps(chunk) + "\n\n").encode() for chunk in chunks)
        return httpx.Response(200, content=data)

    monkeypatch.setattr(
        "llm_eval.core.clients.OpenAI",
        lambda **kwargs: OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler))),
    )
    summary, writer = EvaluationRunner(config).run()
    assert calls == summary.api_attempts == 60 and peak == 10
    assert summary.metrics["concurrency"] == 10
    assert summary.metrics["overall"]["successful"] == 58
    assert summary.metrics["overall"]["errors"] == {"rate_limit": 1, "incomplete_stream": 1}
    assert "Concurrency: **10**" in writer.paths.report_md.read_text(encoding="utf-8")

    offline = config_from_snapshot(load_manifest(writer.paths.root)["config"], writer.paths.root)
    offline = replace(offline, run=replace(offline.run, workers=3))
    regraded, _ = EvaluationRunner(offline, operation="grade", source_dir=writer.paths.root).run()
    assert regraded.metrics == summary.metrics
    assert regraded.new_api_attempts == 0 and calls == 60
