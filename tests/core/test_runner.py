import json
import time

import pytest

from llm_eval.core.clients import GenerationResult
from llm_eval.core.runner import EvaluationRunner
from llm_eval.core.settings import DatasetConfig, FrameworkConfig, ModelConfig, RunConfig
from llm_eval.tasks import TASK_REGISTRY


class FatalClient:
    """Simulates an invalid API key: every call fails fast with a fatal error."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        time.sleep(0.1)
        return GenerationResult(
            content="",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            error="Incorrect API key provided",
            fatal=True,
            http_status_code=401,
        )


class EchoClient:
    """Always answers `#### 4`, so questions expecting 4 pass."""

    def generate(self, messages):
        return GenerationResult(
            content="#### 4",
            usage={"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
            http_status_code=200,
        )


def _gsm_config(tmp_path, rows, **run_kwargs):
    dataset = tmp_path / "gsm.jsonl"
    dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return FrameworkConfig(
        model=ModelConfig(api_key="x", base_url="https://example.invalid/v1", model_name="demo"),
        dataset=DatasetConfig(name="gsm", path=dataset),
        run=RunConfig(task="gsm", workers=1, output_dir=tmp_path / "results", **run_kwargs),
        model_config_path=tmp_path / "model.yaml",
    )


def test_runner_aborts_on_fatal_api_error(tmp_path):
    rows = [{"question": f"What is {i}+{i}?", "answer": f"#### {2 * i}"} for i in range(20)]
    runner = EvaluationRunner(_gsm_config(tmp_path, rows))
    runner.client = FatalClient()

    summary, writer = runner.run()

    assert summary.status_counts.get("API_ERROR_FATAL", 0) >= 1
    assert summary.completed_cases < summary.total_cases
    # every completed case is persisted incrementally
    persisted = writer.paths.results_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(persisted) == summary.total_cases
    assert sum(json.loads(row)["status"] != "CANCELLED" for row in persisted) == summary.completed_cases


def test_runner_writes_run_artifacts(tmp_path):
    rows = [{"question": "What is 2+2?", "answer": "#### 4"} for _ in range(3)]
    runner = EvaluationRunner(_gsm_config(tmp_path, rows))
    runner.client = EchoClient()

    summary, writer = runner.run()

    assert summary.pass_rate == 1.0
    assert writer.paths.report_md.exists()
    assert writer.paths.config_json.exists()
    config_snapshot = json.loads(writer.paths.config_json.read_text(encoding="utf-8"))
    assert config_snapshot["model"]["api_key"] == "***"
    persisted = [json.loads(line) for line in writer.paths.results_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(persisted) == 3
    assert all(item["status"] == "PASSED" for item in persisted)


def test_runner_respects_case_limit(tmp_path):
    rows = [{"question": "What is 2+2?", "answer": "#### 4"} for _ in range(10)]
    runner = EvaluationRunner(_gsm_config(tmp_path, rows, limit=3))
    runner.client = EchoClient()

    summary, _ = runner.run()

    assert summary.total_cases == 3
    assert summary.completed_cases == 3


def test_task_registry_auto_registers_all_tasks():
    expected = {
        "humaneval",
        "humanevalplus",
        "mbpp",
        "mbppplus",
        "gsm",
        "aime2025",
        "aime2026",
        "gpqa",
        "mmlu_pro",
        "ifeval",
        "livecodebench",
        "api_experience",
    }
    assert expected == set(TASK_REGISTRY)


def test_runner_persists_and_reports_usage_from_failed_requests(tmp_path):
    class UsageErrorClient:
        def generate(self, messages):
            return GenerationResult(
                content="",
                usage={"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
                error="stream interrupted",
                attempts=3,
                usage_complete=False,
            )

    runner = EvaluationRunner(_gsm_config(tmp_path, [{"question": "2+2?", "answer": "#### 4"}]))
    runner.client = UsageErrorClient()
    summary, writer = runner.run()
    assert summary.token_usage == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
    result = json.loads(writer.paths.results_jsonl.read_text(encoding="utf-8"))
    assert result["status"] == "API_ERROR"
    assert result["total_tokens"] == 12
    assert result["api_attempts"] == 3
    assert result["usage_complete"] is False
    report = writer.paths.report_md.read_text(encoding="utf-8")
    assert "| API attempts | 3 |" in report
    assert "| Cases with incomplete token usage | 1 |" in report
    assert summary.run_status == "FAILED"
    assert summary.error_rate == 1.0
    assert summary.coverage_rate == 0.0


def test_fatal_error_stops_scheduling_and_persists_aborted_state(tmp_path):
    runner = EvaluationRunner(_gsm_config(tmp_path, [{"question": "2+2?", "answer": "#### 4"}] * 8))
    runner.client = FatalClient()
    summary, writer = runner.run()
    assert runner.client.calls == 1
    assert summary.run_status == "ABORTED"
    assert summary.completed_cases == 1
    assert summary.cancelled_cases == 7
    assert json.loads(writer.paths.status_json.read_text())["status"] == "ABORTED"
    assert json.loads(writer.paths.summary_json.read_text())["cancelled_cases"] == 7


def test_keyboard_interrupt_writes_partial_results_and_report(tmp_path, monkeypatch):
    from llm_eval.core import runner as runner_module

    original_wait = runner_module.concurrent.futures.wait
    calls = 0

    def interrupt_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original_wait(*args, **kwargs)

    monkeypatch.setattr(runner_module.concurrent.futures, "wait", interrupt_once)
    runner = EvaluationRunner(_gsm_config(tmp_path, [{"question": "2+2?", "answer": "#### 4"}] * 8))
    runner.client = EchoClient()
    summary, writer = runner.run()
    assert summary.run_status == "INTERRUPTED"
    assert 1 <= summary.completed_cases < 8
    assert summary.cancelled_cases == 8 - summary.completed_cases
    assert writer.paths.report_md.exists()
    assert json.loads(writer.paths.status_json.read_text())["status"] == "INTERRUPTED"
    assert len(writer.paths.results_jsonl.read_text().splitlines()) == 8


def test_missing_execution_backend_fails_before_model_request(fake_config, monkeypatch):
    from dataclasses import replace

    from llm_eval.core.settings import ExecutionConfig

    fake_config.dataset.path.write_text(
        json.dumps({"task_id": "h/0", "prompt": "def f(): pass", "test": "assert True"}), encoding="utf-8"
    )
    config = replace(fake_config, run=replace(fake_config.run, execution=ExecutionConfig()))
    monkeypatch.setattr("llm_eval.core.execution.shutil.which", lambda _: None)
    runner = EvaluationRunner(config)

    class NoCallClient:
        def generate(self, messages):
            pytest.fail("Preflight must prevent paid model requests")

    runner.client = NoCallClient()
    summary, writer = runner.run()
    assert summary.run_status == "FAILED"
    assert "Docker" in summary.error
    assert summary.completed_cases == 0
    assert json.loads(writer.paths.status_json.read_text())["status"] == "FAILED"


def test_wrong_model_answers_are_a_completed_run(tmp_path):
    runner = EvaluationRunner(_gsm_config(tmp_path, [{"question": "2+3?", "answer": "#### 5"}]))
    runner.client = EchoClient()
    summary, _ = runner.run()
    assert summary.run_status == "COMPLETED"
    assert summary.pass_rate == 0.0
    assert summary.coverage_rate == 1.0
    assert summary.error_rate == 0.0


def test_execution_backend_failure_aborts_remaining_cases(fake_config, monkeypatch):
    from llm_eval.core.execution import ExecutionResult

    row = {"task_id": "h/0", "prompt": "def f(): pass", "entry_point": "f", "test": "def check(f): assert True"}
    fake_config.dataset.path.write_text("\n".join(json.dumps(row) for _ in range(5)), encoding="utf-8")
    runner = EvaluationRunner(fake_config)
    runner.client = EchoClient()
    monkeypatch.setattr(
        "llm_eval.core.execution.ExecutionEngine.run",
        lambda *args, **kwargs: ExecutionResult("EXECUTION_ERROR", detail="daemon disconnected"),
    )
    summary, _ = runner.run()
    assert summary.run_status == "ABORTED"
    assert summary.status_counts["EXECUTION_ERROR"] >= 1
    assert summary.coverage_rate == 0.0
    assert "daemon disconnected" in summary.error


@pytest.mark.parametrize("status", ["TIMEOUT", "OUTPUT_LIMIT", "RESOURCE_LIMIT"])
def test_execution_limits_are_graded_outcomes(fake_config, monkeypatch, status):
    from llm_eval.core.execution import ExecutionResult

    row = {"task_id": "h/0", "prompt": "def f(): pass", "entry_point": "f", "test": "def check(f): assert True"}
    fake_config.dataset.path.write_text(json.dumps(row), encoding="utf-8")
    runner = EvaluationRunner(fake_config)
    runner.client = EchoClient()
    monkeypatch.setattr(
        "llm_eval.core.execution.ExecutionEngine.run",
        lambda *args, **kwargs: ExecutionResult(status, detail="limit exceeded"),
    )
    summary, _ = runner.run()
    assert summary.run_status == "COMPLETED"
    assert summary.coverage_rate == 1.0
    assert summary.error_rate == 0.0
    assert summary.pass_rate == 0.0
