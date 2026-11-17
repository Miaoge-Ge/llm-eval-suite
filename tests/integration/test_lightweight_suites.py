import json
from dataclasses import replace

import httpx
import pytest
from openai import OpenAI

from llm_eval.cli import build_parser, main, rebuild_report
from llm_eval.core.artifacts import RunArchive, load_manifest
from llm_eval.core.clients import GenerationResult
from llm_eval.core.runner import EvaluationRunner
from llm_eval.core.settings import (
    DEFAULT_DATASET_PATHS,
    DatasetConfig,
    config_from_snapshot,
    load_framework_config,
)
from llm_eval.core.utils import repo_root
from llm_eval.tasks import ApiExperienceTask


def suite_config(config, task):
    return replace(
        config,
        dataset=DatasetConfig(task, repo_root() / DEFAULT_DATASET_PATHS[task]),
        run=replace(config.run, task=task, workers=1),
    )


def test_api_pack_has_60_interleaved_unique_prompts_and_reference_lengths(fake_config):
    task = ApiExperienceTask(suite_config(fake_config, "api_experience"))
    cases = task.load_cases()
    assert len({json.dumps(task.build_messages(case)) for case in cases}) == 60
    assert [case.payload["scenario"] for case in cases[:3]] == ["short_answer", "long_answer", "long_context"]
    for case in cases:
        options = task.generation_options(case)
        assert options["max_attempts"] == 1 and options["capture_timing"]
        assert options["parameters"]["max_tokens"] in (256, 768)
        if case.payload["scenario"] == "long_context":
            assert 4000 <= case.payload["reference_input_tokens"] <= 4100


@pytest.mark.parametrize("task", ["api_experience"])
def test_fixed_suites_reject_limit(fake_config, task):
    config = suite_config(fake_config, task)
    with pytest.raises(ValueError, match="--limit"):
        EvaluationRunner(replace(config, run=replace(config.run, limit=1)))


def test_api_refuses_resume_and_generate(fake_config, tmp_path):
    config = suite_config(fake_config, "api_experience")
    with pytest.raises(ValueError, match="cannot be resumed"):
        EvaluationRunner(config, resume_dir=tmp_path)
    with pytest.raises(ValueError, match="api-test"):
        EvaluationRunner(config, operation="generate")


def test_command_defaults():
    parser = build_parser()
    assert parser.parse_args(["api-test"]).task == "api_experience"
    with pytest.raises(SystemExit):
        parser.parse_args(["quick-test"])
    assert parser.parse_args(["api", "--workers", "10"]).workers == 10
    assert parser.parse_args(["benchmark", "--task", "gsm"]).task == "gsm"


@pytest.mark.parametrize("command,task", [("api-test", "api_experience")])
def test_cli_dispatch_and_effective_defaults(tmp_path, monkeypatch, command, task):
    path = tmp_path / "model.yaml"
    path.write_text("api_key: test\nbase_url: http://localhost/v1\nmodel_name: demo\nworkers: 10\n", encoding="utf-8")
    config = load_framework_config(path, task=task)
    assert config.run.workers == 10
    assert load_framework_config(path, task=task, workers=2).run.workers == 2
    assert config.run.max_output_tokens is None

    def capture(config, **kwargs):
        assert config.run.task == task and kwargs["operation"] == "run"
        raise ValueError("dispatch verified")

    monkeypatch.setattr("llm_eval.cli.EvaluationRunner", capture)
    assert main([command, "--config", str(path)]) == 1


def test_api_output_parameter_override_is_recorded(fake_config):
    config = suite_config(fake_config, "api_experience")
    config = replace(
        config, run=replace(config.run, max_output_tokens=1024, output_token_parameter="max_completion_tokens")
    )
    task = ApiExperienceTask(config)
    assert task.generation_options(task.load_cases()[0])["parameters"] == {"max_completion_tokens": 1024}


@pytest.mark.parametrize(
    "settings",
    [
        {"max_output_tokens": 0},
        {"max_output_tokens": True},
        {"output_token_parameter": "guess"},
        {"stall_threshold_seconds": None},
        {"stall_threshold_seconds": -1},
        {"slo_tps": float("nan")},
        {"temperature": float("inf")},
        {"temperature": True},
        {"seed": 1.5},
        {"seed": True},
        {"max_output_tokens": "1024"},
        {"temperature": -0.1},
        {"temperature": 2.1},
        {"stall_threshold_seconds": 0},
        {"stall_threshold_seconds": "1"},
        {"slo_ttft_seconds": 0},
        {"slo_tps": True},
        {"slo_max_gap_seconds": float("inf")},
        {"workers": 0},
        {"workers": True},
        {"workers": 1.5},
    ],
)
def test_invalid_measurement_settings_fail_early(fake_config, settings):
    with pytest.raises(ValueError):
        replace(fake_config.run, **settings)


class EventStream(httpx.SyncByteStream):
    def __init__(self, number):
        self.number = number

    def __iter__(self):
        yield packet({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})
        for text in ("An answer", " with more content."):
            yield packet({"model": "returned-model", "choices": [{"index": 0, "delta": {"content": text}}]})
        if self.number == 3:  # Partial stream: never retry or silently score as successful.
            return
        yield packet(
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "length" if self.number == 4 else "stop"}]}
        )
        yield packet(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 10,
                    "total_tokens": 60,
                    "prompt_tokens_details": {"cached_tokens": 20},
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            }
        )


def packet(data):
    return ("data: " + json.dumps(data) + "\n\n").encode()


def test_full_60_api_requests_no_retries_and_offline_metrics(fake_config, monkeypatch):
    config = suite_config(fake_config, "api_experience")
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 2:
            return httpx.Response(429, json={"error": {"message": "limited"}})
        return httpx.Response(200, stream=EventStream(len(calls)))

    monkeypatch.setattr(
        "llm_eval.core.clients.OpenAI",
        lambda **kwargs: OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler))),
    )
    runner = EvaluationRunner(config)
    summary, writer = runner.run()
    assert len(calls) == summary.api_attempts == 60
    assert summary.run_status == "FAILED"
    overall = summary.metrics["overall"]
    assert overall["attempted"] == 60 and overall["successful"] == 58
    assert overall["truncated"] == 1
    assert overall["errors"] == {"rate_limit": 1, "incomplete_stream": 1}
    assert summary.pass_rate == pytest.approx(58 / 60)
    assert [call["max_tokens"] for call in calls[:3]] == [256, 768, 256]
    raw = list((writer.paths.root / "generations").glob("*/*.json"))
    assert len(raw) == 60
    records = [json.loads(path.read_text(encoding="utf-8")) for path in raw]
    complete = next(record for record in records if record["case_index"] == 0)["generation"]["attempt_records"][0]
    assert complete["cached_prompt_tokens"] == 20
    assert complete["reasoning_tokens"] == 0
    assert len(complete["timing"]["events"]) == 2
    root = writer.paths.root
    assert "First-attempt success" in rebuild_report(root).read_text(encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("Offline operation attempted an API request")

    monkeypatch.setattr("llm_eval.core.runner.OpenAICompatibleClient", forbidden)
    offline = config_from_snapshot(load_manifest(root)["config"], root)
    regraded, _ = EvaluationRunner(offline, operation="grade", source_dir=root).run()
    assert regraded.metrics == summary.metrics
    assert regraded.new_api_attempts == 0
    assert regraded.token_usage["total_tokens"] == 0


def test_fatal_api_error_preserves_planned_and_attempted_counts(fake_config, monkeypatch):
    config = suite_config(fake_config, "api_experience")
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401, json={"error": {"message": "unauthorized"}})

    monkeypatch.setattr(
        "llm_eval.core.clients.OpenAI",
        lambda **kwargs: OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler))),
    )
    summary, _ = EvaluationRunner(config).run()
    assert summary.run_status == "ABORTED"
    assert summary.cancelled_cases == 59 and len(calls) == 1
    overall = summary.metrics["overall"]
    assert overall["planned"] == 60 and overall["attempted"] == 1
    assert overall["errors"] == {"authentication": 1}
    assert overall["observed"]["first_body_seconds"]["n"] == 0


def test_interrupted_api_stream_preserves_timeline_and_offline_counts(fake_config, monkeypatch):
    config = suite_config(fake_config, "api_experience")
    runner = EvaluationRunner(config)
    calls = []

    class InterruptedStream(httpx.SyncByteStream):
        closed = False

        def __iter__(self):
            yield packet({"choices": [{"index": 0, "delta": {"content": "partial answer"}}]})
            runner.cancel_event.set()

        def close(self):
            self.closed = True

    stream = InterruptedStream()

    def handler(request):
        calls.append(request)
        return httpx.Response(200, stream=stream)

    monkeypatch.setattr(
        "llm_eval.core.clients.OpenAI",
        lambda **kwargs: OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler))),
    )
    summary, writer = runner.run()
    assert summary.run_status == "INTERRUPTED"
    assert len(calls) == summary.api_attempts == 1 and stream.closed
    assert summary.status_counts == {"CANCELLED": 60}
    assert summary.metrics["overall"]["attempted"] == 1
    assert summary.metrics["overall"]["errors"] == {"cancelled": 1}
    assert summary.metrics["overall"]["observed"]["first_body_seconds"]["n"] == 1
    raw = json.loads(next((writer.paths.root / "generations").glob("*/*.json")).read_text(encoding="utf-8"))
    assert raw["generation"]["content"] == "partial answer"

    def forbidden(*args, **kwargs):
        pytest.fail("Offline grading must not call the API")

    monkeypatch.setattr("llm_eval.core.runner.OpenAICompatibleClient", forbidden)
    offline = config_from_snapshot(load_manifest(writer.paths.root)["config"], writer.paths.root)
    regraded, _ = EvaluationRunner(offline, operation="grade", source_dir=writer.paths.root).run()
    assert regraded.metrics == summary.metrics
    assert regraded.new_api_attempts == 0


def test_api_storage_failure_keeps_observed_request_count(fake_config, monkeypatch):
    config = suite_config(fake_config, "api_experience")
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, stream=EventStream(1))

    def disk_full(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(
        "llm_eval.core.clients.OpenAI",
        lambda **kwargs: OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler))),
    )
    monkeypatch.setattr(RunArchive, "save_generation", disk_full)
    summary, _ = EvaluationRunner(config).run()
    assert summary.run_status == "ABORTED"
    assert summary.status_counts == {"STORAGE_ERROR": 1, "CANCELLED": 59}
    assert len(calls) == summary.api_attempts == summary.metrics["overall"]["attempted"] == 1
    assert summary.metrics["overall"]["successful"] == 1


@pytest.mark.parametrize("setting", ["max_output_tokens", "output_token_parameter", "temperature", "seed"])
def test_benchmark_resume_rejects_changed_generation_settings(fake_config, monkeypatch, setting):
    config = suite_config(fake_config, "gsm")
    config = replace(config, run=replace(config.run, limit=1))
    runner = EvaluationRunner(config, operation="generate")

    class AnswerClient:
        def generate(self, messages):
            return GenerationResult("saved answer", {})

    runner.client = AnswerClient()
    _, writer = runner.run()
    before = writer.paths.summary_json.read_bytes()
    changes = {
        "max_output_tokens": 1024,
        "output_token_parameter": "max_completion_tokens",
        "temperature": 0,
        "seed": 42,
    }
    changed = replace(config, run=replace(config.run, **{setting: changes[setting]}))
    monkeypatch.setattr("llm_eval.core.runner.OpenAICompatibleClient", lambda *a: pytest.fail("Unexpected API call"))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        EvaluationRunner(changed, resume_dir=writer.paths.root).run()
    assert writer.paths.summary_json.read_bytes() == before
