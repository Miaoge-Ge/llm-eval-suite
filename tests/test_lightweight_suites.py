import json
import os
from collections import Counter
from dataclasses import replace

import httpx
import pytest
from openai import OpenAI

from llm_eval.artifacts import RunArchive, load_manifest
from llm_eval.cli import build_parser, main, rebuild_report
from llm_eval.clients import GenerationResult
from llm_eval.plus import prepare_test_code
from llm_eval.runner import EvaluationRunner
from llm_eval.settings import (
    DEFAULT_DATASET_PATHS,
    NUMPY_IMAGE,
    DatasetConfig,
    ExecutionConfig,
    config_from_snapshot,
    load_framework_config,
)
from llm_eval.tasks import ApiExperienceTask, Quick40Task
from llm_eval.utils import repo_root


def suite_config(config, task):
    return replace(
        config,
        dataset=DatasetConfig(task, repo_root() / DEFAULT_DATASET_PATHS[task]),
        run=replace(config.run, task=task, workers=1),
    )


def test_fixed_selection_original_prompts_full_tests_and_dimensions(fake_config):
    task = Quick40Task(suite_config(fake_config, "quick40"))
    cases = task.load_cases()
    assert len(cases) == len({case.case_id for case in cases}) == 40
    assert Counter(case.payload["dimension"] for case in cases) == dict.fromkeys(
        ("code", "math", "knowledge", "instructions"), 10
    )
    for case in cases:
        delegate, original = task._delegate(case)
        assert task.build_messages(case) == delegate.build_messages(original)
        assert task.validate_case(case) is None
        assert case.payload["reason"]
        if case.payload["dimension"] == "code":
            # The delegated grader receives the entire source test, not a reduced list.
            assert len(original.payload["test"]) > 1000
            program = delegate.build_test_program(original, reference_answer(case))
            full_test = prepare_test_code(original.payload["test"], original.payload["entry_point"]).strip()
            assert full_test in program


def test_changed_source_hash_is_rejected(fake_config, tmp_path):
    config = suite_config(fake_config, "quick40")
    manifest = json.loads(config.dataset.path.read_text(encoding="utf-8"))
    manifest["cases"][0]["sha256"] = "0" * 64
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    task = Quick40Task(replace(config, dataset=replace(config.dataset, path=path)))
    with pytest.raises(ValueError, match="row changed"):
        task.load_cases()


def test_api_pack_has_30_interleaved_unique_prompts_and_reference_lengths(fake_config):
    task = ApiExperienceTask(suite_config(fake_config, "api_experience"))
    cases = task.load_cases()
    assert len({json.dumps(task.build_messages(case)) for case in cases}) == 30
    assert [case.payload["scenario"] for case in cases[:3]] == ["short_answer", "long_answer", "long_context"]
    for case in cases:
        options = task.generation_options(case)
        assert options["max_attempts"] == 1 and options["capture_timing"]
        assert options["parameters"]["max_tokens"] in (256, 768)
        if case.payload["scenario"] == "long_context":
            assert 4000 <= case.payload["reference_input_tokens"] <= 4100


@pytest.mark.parametrize("task", ["api_experience", "quick40"])
def test_fixed_suites_reject_limit(fake_config, task):
    config = suite_config(fake_config, task)
    with pytest.raises(ValueError, match="--limit"):
        EvaluationRunner(replace(config, run=replace(config.run, limit=1)))


def test_api_refuses_resume_parallelism_and_generate(fake_config, tmp_path):
    config = suite_config(fake_config, "api_experience")
    with pytest.raises(ValueError, match="cannot be resumed"):
        EvaluationRunner(config, resume_dir=tmp_path)
    with pytest.raises(ValueError, match="one worker"):
        EvaluationRunner(replace(config, run=replace(config.run, workers=2)))
    with pytest.raises(ValueError, match="api-test"):
        EvaluationRunner(config, operation="generate")


def test_command_defaults():
    parser = build_parser()
    assert parser.parse_args(["api-test"]).task == "api_experience"
    args = parser.parse_args(["quick-test"])
    assert args.task == "quick40" and args.workers == 1 and args.limit is None


@pytest.mark.parametrize("command,task", [("api-test", "api_experience"), ("quick-test", "quick40")])
def test_cli_dispatch_and_effective_defaults(tmp_path, monkeypatch, command, task):
    path = tmp_path / "model.yaml"
    path.write_text("api_key: test\nbase_url: http://localhost/v1\nmodel_name: demo\nworkers: 10\n", encoding="utf-8")
    config = load_framework_config(path, task=task)
    assert config.run.workers == 1
    assert config.run.max_output_tokens == (4096 if task == "quick40" else None)
    if task == "quick40":
        assert config.run.execution.image == NUMPY_IMAGE

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


def test_full_30_api_requests_no_retries_and_offline_metrics(fake_config, monkeypatch):
    config = suite_config(fake_config, "api_experience")
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 2:
            return httpx.Response(429, json={"error": {"message": "limited"}})
        return httpx.Response(200, stream=EventStream(len(calls)))

    monkeypatch.setattr(
        "llm_eval.clients.OpenAI",
        lambda **kwargs: OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler))),
    )
    runner = EvaluationRunner(config)
    summary, writer = runner.run()
    assert len(calls) == summary.api_attempts == 30
    assert summary.run_status == "FAILED"
    overall = summary.metrics["overall"]
    assert overall["attempted"] == 30 and overall["successful"] == 28
    assert overall["truncated"] == 1
    assert overall["errors"] == {"rate_limit": 1, "incomplete_stream": 1}
    assert summary.pass_rate == pytest.approx(28 / 30)
    assert [call["max_tokens"] for call in calls[:3]] == [256, 768, 256]
    raw = list((writer.paths.root / "generations").glob("*/*.json"))
    assert len(raw) == 30
    records = [json.loads(path.read_text(encoding="utf-8")) for path in raw]
    complete = next(record for record in records if record["case_index"] == 0)["generation"]["attempt_records"][0]
    assert complete["cached_prompt_tokens"] == 20
    assert complete["reasoning_tokens"] == 0
    assert len(complete["timing"]["events"]) == 2
    root = writer.paths.root
    assert "First-attempt success" in rebuild_report(root).read_text(encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("Offline operation attempted an API request")

    monkeypatch.setattr("llm_eval.runner.OpenAICompatibleClient", forbidden)
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
        "llm_eval.clients.OpenAI",
        lambda **kwargs: OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler))),
    )
    summary, _ = EvaluationRunner(config).run()
    assert summary.run_status == "ABORTED"
    assert summary.cancelled_cases == 29 and len(calls) == 1
    overall = summary.metrics["overall"]
    assert overall["planned"] == 30 and overall["attempted"] == 1
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
        "llm_eval.clients.OpenAI",
        lambda **kwargs: OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler))),
    )
    summary, writer = runner.run()
    assert summary.run_status == "INTERRUPTED"
    assert len(calls) == summary.api_attempts == 1 and stream.closed
    assert summary.status_counts == {"CANCELLED": 30}
    assert summary.metrics["overall"]["attempted"] == 1
    assert summary.metrics["overall"]["errors"] == {"cancelled": 1}
    assert summary.metrics["overall"]["observed"]["first_body_seconds"]["n"] == 1
    raw = json.loads(next((writer.paths.root / "generations").glob("*/*.json")).read_text(encoding="utf-8"))
    assert raw["generation"]["content"] == "partial answer"

    def forbidden(*args, **kwargs):
        pytest.fail("Offline grading must not call the API")

    monkeypatch.setattr("llm_eval.runner.OpenAICompatibleClient", forbidden)
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
        "llm_eval.clients.OpenAI",
        lambda **kwargs: OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler))),
    )
    monkeypatch.setattr(RunArchive, "save_generation", disk_full)
    summary, _ = EvaluationRunner(config).run()
    assert summary.run_status == "ABORTED"
    assert summary.status_counts == {"STORAGE_ERROR": 1, "CANCELLED": 29}
    assert len(calls) == summary.api_attempts == summary.metrics["overall"]["attempted"] == 1
    assert summary.metrics["overall"]["successful"] == 1


@pytest.mark.parametrize("setting", ["max_output_tokens", "output_token_parameter", "temperature", "seed"])
def test_quick40_resume_rejects_changed_generation_settings(fake_config, monkeypatch, setting):
    config = suite_config(fake_config, "quick40")
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
    monkeypatch.setattr("llm_eval.runner.OpenAICompatibleClient", lambda *a: pytest.fail("Unexpected API call"))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        EvaluationRunner(changed, resume_dir=writer.paths.root).run()
    assert writer.paths.summary_json.read_bytes() == before


def test_quick40_infrastructure_errors_are_distinct_from_incorrect_answers(fake_config):
    task = Quick40Task(suite_config(fake_config, "quick40"))
    cases = task.load_cases()
    rows = [
        {"status": status, "metadata": task.case_metadata(case)}
        for case, status in zip(cases[:4], ["PASSED", "FAILED", "API_ERROR", "CANCELLED"], strict=True)
    ]
    stats = task.summarize(rows)["dimensions"]["code"]
    assert stats == {"planned": 10, "passed": 1, "graded": 2, "errors": 1, "cancelled": 1, "truncated": 0}


def reference_answer(case):
    row = case.payload["source_case"]
    task = case.payload["source_task"]
    if task == "humanevalplus":
        return row["prompt"] + row["canonical_solution"]
    if task == "mbppplus":
        return {
            "similar_elements": "def similar_elements(a, b):\n    return tuple(set(a) & set(b))",
            "square_nums": "def square_nums(a):\n    return [x*x for x in a]",
            "check": "def check(n):\n    return n == 2 * int(str(n)[::-1]) - 1",
            "dif_Square": "def dif_Square(n):\n    return n % 4 != 2",
            "frequency_lists": (
                "from collections import Counter\ndef frequency_lists(a):\n"
                "    return dict(Counter(x for r in a for x in r))"
            ),
        }[row["entry_point"]]
    if task == "gsm":
        return row["answer"]
    if task in {"gpqa", "mmlu_pro", "aime2025"}:
        return "\\boxed{" + str(row["answer"]) + "}"
    return {
        1001: "Hark! On thy first day explore Tokyo. Then seek Kyoto and its temples.",
        1005: " ".join(f"[placeholder{i}]" for i in range(12)),
        1012: (row["kwargs"][0].get("prompt_to_repeat") or "") + "\n<<Resignation>>\nDear Boss I am quitting.",
        1019: "what are the two young boys playing with?",
        102: '* Gown: "Shall we go?"\n* Sweats: "I am ready."\n* Gown: "Let us leave."',
        1075: '{"product": "Diaper", "message": "Soft comfort for your baby."}',
        1082: "He woke inside a video game.\n***\nHe found the exit and escaped.",
        1098: "God is called the Lord.\n******\nGod is called the Creator.",
        1128: "The sentiment is negative. Is there anything else I can help with?",
        1147: "We have implemented a correction and will continue to work closely with our partners.",
    }[row["key"]]


@pytest.mark.skipif(os.getenv("LLM_EVAL_DOCKER_TESTS") != "1", reason="Requires real Docker and NumPy image")
def test_full_quick40_reference_run_resume_and_offline_grading(fake_config, monkeypatch):
    config = suite_config(fake_config, "quick40")
    config = replace(config, run=replace(config.run, execution=ExecutionConfig(image=NUMPY_IMAGE)))
    task = Quick40Task(config)
    cases = task.load_cases()
    answers = {json.dumps(task.build_messages(case)): reference_answer(case) for case in cases}

    class ReferenceClient:
        calls = 0

        def generate(self, messages):
            self.calls += 1
            return GenerationResult(
                answers[json.dumps(messages)], {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            )

    runner = EvaluationRunner(config)
    runner.client = client = ReferenceClient()
    summary, writer = runner.run()
    assert summary.status_counts == {"PASSED": 40}, writer.paths.root
    assert client.calls == 40
    assert all(stats["passed"] == stats["graded"] == 10 for stats in summary.metrics["dimensions"].values())

    def forbidden(*args, **kwargs):
        pytest.fail("Resume/offline grade attempted an API request")

    monkeypatch.setattr("llm_eval.runner.OpenAICompatibleClient", forbidden)
    resumed, _ = EvaluationRunner(config, resume_dir=writer.paths.root).run()
    assert resumed.new_api_attempts == 0 and resumed.status_counts == summary.status_counts
    offline = config_from_snapshot(load_manifest(writer.paths.root)["config"], writer.paths.root)
    monkeypatch.setattr(Quick40Task, "load_cases", forbidden)  # Regrade must use the snapshot, not current source rows.
    regraded, _ = EvaluationRunner(offline, operation="grade", source_dir=writer.paths.root).run()
    assert regraded.status_counts == summary.status_counts
    assert regraded.metrics == summary.metrics
    assert regraded.new_api_attempts == 0
