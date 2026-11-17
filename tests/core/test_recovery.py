import json
from dataclasses import replace

import pytest

from llm_eval.cli import main, rebuild_report
from llm_eval.core.artifacts import RunArchive, RunLock, load_manifest
from llm_eval.core.clients import GenerationResult
from llm_eval.core.runner import EvaluationRunner
from llm_eval.core.settings import DatasetConfig, FrameworkConfig, ModelConfig, RunConfig, config_from_snapshot
from llm_eval.tasks import GSM8KTask


class AnswerClient:
    def __init__(self, answer="  work\n#### 4\n ", error=None):
        self.answer = answer
        self.error = error
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        return GenerationResult(
            self.answer, {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}, error=self.error
        )


@pytest.fixture
def config(tmp_path):
    dataset = tmp_path / "input.jsonl"
    dataset.write_text(json.dumps({"question": "What is 2+2?", "answer": "#### 4"}) + "\n", encoding="utf-8")
    return FrameworkConfig(
        ModelConfig("test-secret", "https://example.invalid/v1", "test-model"),
        DatasetConfig("gsm", dataset),
        RunConfig("gsm", workers=1, output_dir=tmp_path / "results"),
        tmp_path / "model.yaml",
    )


def run(config, **kwargs):
    runner = EvaluationRunner(config, **kwargs)
    client = AnswerClient()
    runner.client = client
    summary, writer = runner.run()
    return summary, writer.paths.root, client


def deny_api(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("This operation must not create an API client")

    monkeypatch.setattr("llm_eval.core.runner.OpenAICompatibleClient", forbidden)


def test_raw_answer_messages_and_snapshot_are_saved(config):
    summary, root, client = run(config)
    record = json.loads(next((root / "generations").glob("*/*.json")).read_text(encoding="utf-8"))
    assert record["generation"]["content"] == client.answer
    assert record["messages"][1]["content"] == "What is 2+2?"
    assert record["request_parameters"]["model"] == "test-model"
    assert record["generation"]["usage"]["total_tokens"] == 12
    assert (root / "dataset.jsonl").is_file()
    assert "test-secret" not in (root / "manifest.json").read_text(encoding="utf-8")
    assert summary.new_api_attempts == 1


def test_raw_answer_survives_grader_failure_and_resume_reuses_it(config, monkeypatch):
    original_grade = GSM8KTask.grade_case

    def broken_grader(self, *args, **kwargs):
        assert list(config.run.output_dir.glob("**/generations/*/*.json")), "Raw answer must precede grading"
        raise RuntimeError("simulated grading failure")

    monkeypatch.setattr(GSM8KTask, "grade_case", broken_grader)
    summary, root, _ = run(config)
    assert summary.run_status == "FAILED"
    monkeypatch.setattr(GSM8KTask, "grade_case", original_grade)
    deny_api(monkeypatch)
    summary, _ = EvaluationRunner(config, resume_dir=root).run()
    assert summary.pass_rate == 1.0
    assert summary.new_api_attempts == 0
    assert summary.api_attempts == 1
    assert summary.token_usage["total_tokens"] == 12


def test_resume_keeps_finished_cases_and_rebuilds_truncated_jsonl(config, monkeypatch):
    _, root, _ = run(config)
    (root / "results.jsonl").write_text('{"partial":', encoding="utf-8")
    deny_api(monkeypatch)
    monkeypatch.setattr(GSM8KTask, "grade_case", lambda *a, **kw: pytest.fail("Finished case must not be graded twice"))
    summary, _ = EvaluationRunner(config, resume_dir=root).run()
    assert summary.run_status == "COMPLETED"
    assert summary.new_api_attempts == 0
    assert json.loads((root / "results.jsonl").read_text())["status"] == "PASSED"
    assert list((root / "history").glob("results-*.jsonl"))


def test_resume_recovers_generation_written_before_checkpoint(config, monkeypatch):
    _, root, _ = run(config)
    (root / "cases" / "00000000.json").unlink()
    (root / "results.jsonl").write_text("", encoding="utf-8")
    deny_api(monkeypatch)
    summary, _ = EvaluationRunner(config, resume_dir=root).run()
    assert summary.pass_rate == 1.0
    assert summary.new_token_usage["total_tokens"] == 0
    assert summary.token_usage["total_tokens"] == 12


def test_resume_retries_api_errors_preserving_usage_history(config):
    runner = EvaluationRunner(config)
    runner.client = AnswerClient("partial", error="connection failed")
    summary, writer = runner.run()
    assert summary.run_status == "FAILED"
    summary, root, client = run(config, resume_dir=writer.paths.root)
    assert client.calls == 1
    assert summary.api_attempts == 2
    assert summary.new_api_attempts == 1
    assert summary.token_usage["total_tokens"] == 24
    assert summary.new_token_usage["total_tokens"] == 12
    assert len(list((root / "generations" / "00000000").glob("*.json"))) == 2


@pytest.mark.parametrize("change", ["dataset", "model", "thinking", "execution", "implementation"])
def test_resume_rejects_identity_changes_without_mutating_results(config, monkeypatch, change):
    _, root, _ = run(config)
    before = {
        name: (root / name).read_bytes() for name in ("results.jsonl", "summary.json", "status.json", "config.json")
    }
    if change == "dataset":
        config.dataset.path.write_text('{"question":"different", "answer":"#### 4"}\n', encoding="utf-8")
    elif change == "model":
        config = replace(config, model=replace(config.model, model_name="other-model"))
    elif change == "thinking":
        config = replace(config, run=replace(config.run, thinking_enabled=True))
    elif change == "execution":
        config = replace(config, run=replace(config.run, execution_timeout_seconds=99))
    else:
        monkeypatch.setattr("llm_eval.core.artifacts.implementation_hash", lambda: "changed")
    deny_api(monkeypatch)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        EvaluationRunner(config, resume_dir=root).run()
    assert all((root / name).read_bytes() == data for name, data in before.items())


def test_resume_allows_new_api_key_and_worker_count(config, monkeypatch):
    _, root, _ = run(config)
    changed = replace(config, model=replace(config.model, api_key="rotated-key"), run=replace(config.run, workers=3))
    deny_api(monkeypatch)
    summary, _ = EvaluationRunner(changed, resume_dir=root).run()
    assert summary.pass_rate == 1.0


def test_generate_does_not_grade_and_grade_uses_snapshot_without_api(config, monkeypatch):
    original = GSM8KTask.grade_case
    monkeypatch.setattr(GSM8KTask, "grade_case", lambda *a, **kw: pytest.fail("Generation must not grade"))
    summary, root, _ = run(config, operation="generate")
    assert summary.status_counts == {"GENERATED": 1}
    monkeypatch.setattr(GSM8KTask, "grade_case", original)
    config.dataset.path.unlink()
    deny_api(monkeypatch)
    offline = config_from_snapshot(load_manifest(root)["config"], root)
    summary, writer = EvaluationRunner(offline, operation="grade", source_dir=root).run()
    assert summary.pass_rate == 1.0
    assert summary.token_usage["total_tokens"] == 0
    assert summary.api_attempts == 0
    assert writer.paths.root != root
    assert json.loads((root / "summary.json").read_text())["operation"] == "generate"


def test_regrade_uses_current_grader_and_preserves_original_results(config, monkeypatch):
    _, root, _ = run(config)
    original = (root / "results.jsonl").read_bytes()
    deny_api(monkeypatch)
    monkeypatch.setattr(GSM8KTask, "_numbers_match", lambda *a: False)
    offline = config_from_snapshot(load_manifest(root)["config"], root)
    summary, _ = EvaluationRunner(offline, operation="grade", source_dir=root).run()
    assert summary.pass_rate == 0.0
    assert summary.run_status == "COMPLETED"
    assert (root / "results.jsonl").read_bytes() == original


def test_incomplete_response_is_not_graded_offline(config, monkeypatch):
    runner = EvaluationRunner(config)
    runner.client = AnswerClient("#### 4", error="stream failed")
    _, writer = runner.run()
    root = writer.paths.root
    deny_api(monkeypatch)
    offline = config_from_snapshot(load_manifest(root)["config"], root)
    summary, _ = EvaluationRunner(offline, operation="grade", source_dir=root).run()
    assert summary.status_counts == {"MISSING_GENERATION": 1}
    assert summary.run_status == "FAILED"
    assert summary.api_attempts == 0


def test_tampered_snapshot_is_rejected(config, monkeypatch):
    _, root, _ = run(config)
    (root / "dataset.jsonl").write_text("{}\n", encoding="utf-8")
    deny_api(monkeypatch)
    with pytest.raises(ValueError, match="snapshot has changed"):
        EvaluationRunner(config, resume_dir=root).run()


def test_run_lock_prevents_concurrent_writers_and_releases(config):
    _, root, _ = run(config)
    with RunLock(root), pytest.raises(ValueError, match="already in use"):
        EvaluationRunner(config, resume_dir=root).run()
    with RunLock(root):
        pass


def test_report_and_grade_cli_do_not_need_model_config_or_key(config, monkeypatch):
    _, root, _ = run(config)
    deny_api(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (root / "report.md").unlink()
    assert main(["report", "--run-dir", str(root)]) == 0
    assert rebuild_report(root).is_file()
    assert main(["grade", "--run-dir", str(root)]) == 0


def test_failed_raw_storage_aborts_without_grading(config, monkeypatch):
    def storage_failure(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(RunArchive, "save_generation", storage_failure)
    monkeypatch.setattr(GSM8KTask, "grade_case", lambda *a, **kw: pytest.fail("Unsaved answer must not be graded"))
    summary, _, _ = run(config)
    assert summary.run_status == "ABORTED"
    assert summary.status_counts == {"STORAGE_ERROR": 1}
    assert summary.api_attempts == summary.new_api_attempts == 1
    assert summary.token_usage["total_tokens"] == summary.new_token_usage["total_tokens"] == 12


def test_resume_keeps_known_cost_when_raw_storage_failed(config, monkeypatch):
    def storage_failure(*args, **kwargs):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(RunArchive, "save_generation", storage_failure)
        _, root, _ = run(config)
        summary, _, _ = run(config, resume_dir=root)
        assert summary.api_attempts == 2
        assert summary.new_api_attempts == 1
        assert summary.token_usage["total_tokens"] == 24

    summary, _, client = run(config, resume_dir=root)
    assert client.calls == 1
    assert summary.status_counts == {"PASSED": 1}
    assert summary.api_attempts == 3
    assert summary.new_api_attempts == 1
    assert summary.token_usage["total_tokens"] == 36
    assert summary.new_token_usage["total_tokens"] == 12
    deny_api(monkeypatch)
    summary, _ = EvaluationRunner(config, resume_dir=root).run()
    assert summary.api_attempts == 3
    assert summary.new_api_attempts == 0
    assert summary.token_usage["total_tokens"] == 36


def test_resume_rejects_checkpoint_linked_to_another_case(config):
    config.dataset.path.write_text(
        "\n".join(json.dumps({"question": "2+2?", "answer": "#### 4"}) for _ in range(2)), encoding="utf-8"
    )
    _, root, _ = run(config)
    first_path = root / "cases" / "00000000.json"
    first = json.loads(first_path.read_text())
    second = json.loads((root / "cases" / "00000001.json").read_text())
    first["metadata"]["generation_id"] = second["metadata"]["generation_id"]
    first_path.write_text(json.dumps(first), encoding="utf-8")
    before = (root / "results.jsonl").read_bytes()
    with pytest.raises(ValueError, match="another case"):
        EvaluationRunner(config, resume_dir=root).run()
    assert (root / "results.jsonl").read_bytes() == before


@pytest.mark.parametrize("usage", [None, {"total_tokens": -1}, {"total_tokens": "12"}, {"total_tokens": True}])
def test_invalid_saved_usage_fails_cleanly_without_overwriting_results(config, usage, capsys):
    _, root, _ = run(config)
    path = next((root / "generations").glob("*/*.json"))
    record = json.loads(path.read_text())
    record["generation"]["usage"] = usage
    path.write_text(json.dumps(record), encoding="utf-8")
    before = (root / "results.jsonl").read_bytes()
    assert main(["grade", "--run-dir", str(root)]) == 2
    assert "Invalid generation record" in capsys.readouterr().err
    assert (root / "results.jsonl").read_bytes() == before


@pytest.mark.parametrize("contents", ["{}", "[]", "null", '{"broken":'])
def test_malformed_saved_generation_is_rejected(config, contents):
    _, root, _ = run(config)
    path = next((root / "generations").glob("*/*.json"))
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid generation record"):
        EvaluationRunner(config, resume_dir=root).run()


def test_duplicate_generation_cannot_double_count_usage(config):
    _, root, _ = run(config)
    path = next((root / "generations").glob("*/*.json"))
    path.with_name("00000002.json").write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="duplicate generation"):
        EvaluationRunner(config, resume_dir=root).run()


def test_report_rejects_unknown_checkpoint_status(config):
    _, root, _ = run(config)
    path = root / "cases" / "00000000.json"
    result = json.loads(path.read_text())
    result["status"] = "UNKNOWN"
    path.write_text(json.dumps(result), encoding="utf-8")
    before = (root / "report.md").read_bytes()
    with pytest.raises(ValueError, match="Invalid checkpoint status"):
        rebuild_report(root)
    assert (root / "report.md").read_bytes() == before
