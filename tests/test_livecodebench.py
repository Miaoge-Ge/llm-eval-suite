import base64
import json
import pickle
import zlib
from dataclasses import replace

import pytest

from llm_eval.artifacts import load_manifest
from llm_eval.clients import GenerationResult
from llm_eval.livecodebench import (
    asset_path,
    decode_private_tests,
    iter_private_tests,
    normalize_problem,
    outputs_match,
    write_private_tests,
)
from llm_eval.runner import EvaluationRunner
from llm_eval.settings import config_from_snapshot
from llm_eval.tasks import LiveCodeBenchTask, TaskCase


def encoded(tests):
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode()


def test_private_decoder_supports_both_upstream_encodings():
    tests = [{"input": "10\n", "output": "20\n", "testtype": "stdin"}]
    assert decode_private_tests(json.dumps(tests)) == tests
    assert decode_private_tests(encoded(tests)) == tests


def test_private_decoder_rejects_executable_pickle_and_oversized_data(monkeypatch):
    class GlobalCall:
        def __reduce__(self):
            return str, ("not test data",)

    payload = base64.b64encode(zlib.compress(pickle.dumps(GlobalCall()))).decode()
    with pytest.raises(ValueError, match="globals are forbidden"):
        decode_private_tests(payload)
    monkeypatch.setattr("llm_eval.livecodebench.MAX_PRIVATE_BYTES", 16)
    with pytest.raises(ValueError, match="too large"):
        decode_private_tests(encoded([{"input": "x" * 100, "output": "y", "testtype": "stdin"}]))


def problem(private=None):
    return {
        "question_id": "q",
        "question_title": "Double",
        "platform": "atcoder",
        "difficulty": "easy",
        "question_content": "Read n and print 2n",
        "public_test_cases": json.dumps([{"input": "3\n", "output": "6\n", "testtype": "stdin"}]),
        "private_test_cases": encoded(
            private if private is not None else [{"input": "10\n", "output": "20\n", "testtype": "stdin"}]
        ),
    }


def test_import_keeps_private_tests_and_revision(tmp_path):
    row = normalize_problem(problem(), tmp_path, "pinned-revision")
    assert row["source"]["revision"] == "pinned-revision"
    assert row["test_scope"] == "stdin_public_and_private"
    assert list(iter_private_tests(tmp_path, row["private_tests"]))[0]["input"] == "10\n"


@pytest.mark.parametrize("private", [[], [{"input": "1", "output": "2", "testtype": "functional"}]])
def test_import_never_silently_drops_missing_or_incompatible_private_tests(tmp_path, private):
    with pytest.raises(ValueError):
        normalize_problem(problem(private), tmp_path, "revision")


@pytest.mark.parametrize(
    ("actual", "expected", "matches"),
    [
        (" 1.0   2e0 \n 3.00\n", "1 2\n3\n", True),
        ("9007199254740993", "9007199254740992", False),
        ("1.000000001", "1", False),
        ("nan", "1", False),
        ("yes", "YES", False),
        ("1 2", "2 1", False),
        ("1 2", "1\n2", False),
        ("hello world", "hello  world", False),
        ("x\r\n y  ", "x\ny", True),
    ],
)
def test_output_comparison_preserves_order_lines_and_exact_numbers(actual, expected, matches):
    assert outputs_match(actual, expected) is matches


@pytest.mark.parametrize(("answer", "status"), [("print(6)", "FAILED"), ("print(int(input()) * 2)", "PASSED")])
def test_private_tests_catch_public_sample_overfitting(fake_config, answer, status):
    row = normalize_problem(problem(), fake_config.dataset.path.parent, "revision")
    task = LiveCodeBenchTask(fake_config)
    case = TaskCase("q", row)
    result = task.grade_case(case, GenerationResult(answer, {}))
    assert result.status == status
    assert result.metadata["tests_executed"] == 2
    assert result.metadata["private_tests_total"] == 1


def test_run_snapshots_private_tests_for_offline_grading(fake_config, monkeypatch):
    row = normalize_problem(problem(), fake_config.dataset.path.parent, "revision")
    fake_config.dataset.path.write_text(json.dumps(row), encoding="utf-8")
    config = replace(fake_config, run=replace(fake_config.run, task="livecodebench"))

    class Client:
        def generate(self, messages):
            return GenerationResult("print(int(input()) * 2)", {})

    runner = EvaluationRunner(config)
    runner.client = Client()
    summary, writer = runner.run()
    assert summary.status_counts == {"PASSED": 1}
    assert summary.test_coverage["private_tests"] == 1
    root = writer.paths.root
    (fake_config.dataset.path.parent / row["private_tests"]["path"]).unlink()
    monkeypatch.setattr("llm_eval.runner.OpenAICompatibleClient", lambda *a: pytest.fail("Offline must not call model"))
    config = config_from_snapshot(load_manifest(root)["config"], root)
    summary, graded_writer = EvaluationRunner(config, operation="grade", source_dir=root).run()
    assert summary.status_counts == {"PASSED": 1}
    assert summary.api_attempts == 0
    assert asset_path(graded_writer.paths.root, row["private_tests"]).is_file()


def test_missing_or_changed_private_assets_fail_before_generation(fake_config, monkeypatch):
    row = normalize_problem(problem(), fake_config.dataset.path.parent, "revision")
    fake_config.dataset.path.write_text(json.dumps(row), encoding="utf-8")
    config = replace(fake_config, run=replace(fake_config.run, task="livecodebench"))
    (fake_config.dataset.path.parent / row["private_tests"]["path"]).write_bytes(b"corrupt")
    monkeypatch.setattr("llm_eval.runner.OpenAICompatibleClient", lambda *a: pytest.fail("Unexpected API call"))
    summary, _ = EvaluationRunner(config).run()
    assert summary.run_status == "FAILED"
    assert "checksum" in summary.error
    assert summary.api_attempts == 0


def test_public_only_report_cannot_be_confused_with_full_test_coverage(fake_config):
    row = {"question_id": "q", "problem": "Print 6", "tests": [{"input": "", "output": "6"}]}
    fake_config.dataset.path.write_text(json.dumps(row), encoding="utf-8")
    config = replace(fake_config, run=replace(fake_config.run, task="livecodebench"))
    runner = EvaluationRunner(config)
    runner.client = type("Client", (), {"generate": lambda *a: GenerationResult("print(6)", {})})()
    summary, writer = runner.run()
    assert summary.test_coverage["scope"] == "stdin_public_only"
    assert "Incomplete test coverage" in writer.paths.report_md.read_text(encoding="utf-8")


def test_private_asset_count_mismatch_is_framework_error(fake_config):
    descriptor = write_private_tests(fake_config.dataset.path.parent, [{"input": "", "output": "6"}])
    descriptor["count"] = 2
    case = TaskCase("q", {"problem": "print6", "tests": [{"input": "", "output": "6"}], "private_tests": descriptor})
    result = LiveCodeBenchTask(fake_config).grade_case(case, GenerationResult("print(6)", {}))
    assert result.status == "INTERNAL_ERROR"
