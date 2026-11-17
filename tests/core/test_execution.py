from functools import partial

import pytest

from llm_eval.core.execution import execute_python, run_python_with_stdin
from llm_eval.core.settings import ExecutionConfig
from llm_eval.tasks import HumanEvalPlusTask, HumanEvalTask, MBPPPlusTask, MBPPTask, TaskCase

execute_python = partial(execute_python, config=ExecutionConfig(backend="local"))
run_python_with_stdin = partial(run_python_with_stdin, config=ExecutionConfig(backend="local"))


@pytest.mark.parametrize("exit_code", ["import sys; sys.exit(0)", "raise SystemExit(0)", "import os; os._exit(0)"])
def test_clean_exit_before_assertions_is_not_a_pass(exit_code):
    status, detail = execute_python(exit_code + "\nassert False", 5)
    assert status == "FAILED"
    assert "before completion" in detail


def test_normal_assertions_and_output():
    status, output = execute_python("assert 1 + 1 == 2\nprint('completed')", 5)
    assert (status, output) == ("PASSED", "completed")


def test_printing_completion_text_cannot_pass_skipped_assertions():
    assert execute_python("print('completed')\nraise SystemExit(0)\nassert False", 5)[0] == "FAILED"


def test_assertion_failure_is_preserved():
    status, detail = execute_python("assert False, 'wrong answer'", 5)
    assert status == "FAILED"
    assert "AssertionError: wrong answer" in detail


def test_assertions_cannot_be_disabled_by_parent_environment(monkeypatch):
    monkeypatch.setenv("PYTHONOPTIMIZE", "1")
    assert execute_python("assert False", 5)[0] == "FAILED"


def test_execution_still_times_out():
    assert execute_python("while True: pass", 1)[0] == "TIMEOUT"


def test_stdin_program_may_exit_after_producing_correct_output():
    # Completion receipts apply to appended assertions, not stdout judging.
    status, output, _ = run_python_with_stdin("import sys\nprint(input())\nsys.exit(0)", "42\n", 5)
    assert status == "OK"
    assert output.strip() == "42"


@pytest.mark.parametrize("task_type", [HumanEvalTask, HumanEvalPlusTask, MBPPTask, MBPPPlusTask])
def test_code_tasks_reject_empty_tests_before_generation(fake_config, task_type):
    class NoCallClient:
        def generate(self, messages):
            pytest.fail("Invalid test data must not trigger generation")

    case = TaskCase("empty", {"test": "  ", "test_list": [], "challenge_test_list": []})
    result = task_type(fake_config).evaluate_case(case, NoCallClient())
    assert result.status == "INTERNAL_ERROR"


@pytest.mark.parametrize("task_type", [HumanEvalTask, HumanEvalPlusTask, MBPPTask, MBPPPlusTask])
@pytest.mark.parametrize("body", ["return a + b", "import sys; sys.exit(0)"])
def test_code_tasks_require_test_program_completion(fake_config, task_type, body):
    from llm_eval.core.clients import GenerationResult

    class CandidateClient:
        def generate(self, messages):
            return GenerationResult("def add(a, b):\n    " + body, {})

    case = TaskCase(
        "add",
        {
            "prompt": 'def add(a, b):\n    """Add two numbers."""',
            "text": "Add two numbers",
            "entry_point": "add",
            "test": (
                "assert add(1, 2) == 3"
                if task_type is MBPPPlusTask
                else "def check(candidate):\n    assert candidate(1, 2) == 3"
            ),
            "test_list": ["assert add(1, 2) == 3"],
        },
    )
    result = task_type(fake_config).evaluate_case(case, CandidateClient())
    assert result.status == ("PASSED" if body.startswith("return") else "FAILED")
