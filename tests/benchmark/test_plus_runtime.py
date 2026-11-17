from dataclasses import replace

import pytest

from llm_eval.core.clients import GenerationResult
from llm_eval.core.execution import ExecutionResult
from llm_eval.core.settings import NUMPY_IMAGE, load_framework_config
from llm_eval.tasks import HumanEvalPlusTask, MBPPPlusTask, TaskCase


@pytest.mark.parametrize("task_type", [HumanEvalPlusTask, MBPPPlusTask])
@pytest.mark.parametrize(
    ("expression", "assertion", "status"),
    [
        ("np.array([1, 2])", "assert np.allclose(f(), [[1, 2], [1, 2]])", "PASSED"),
        ("[np.inf, -np.inf]", "assert np.allclose(f(), [np.inf, -np.inf])", "PASSED"),
        ("np.allclose(1., 1.000001)", "assert f()", "PASSED"),
        ("np.isclose([1, 2], [1, 3]).tolist()", "assert f() == [True, False]", "PASSED"),
        ("np.float32(16777217)", "assert f() == 16777216", "PASSED"),
        ("np.allclose('same', 'same')", "assert f()", "FAILED"),
    ],
)
def test_plus_uses_real_numpy_semantics(fake_config, task_type, expression, assertion, status):
    task = task_type(fake_config)
    test = "import numpy as np\n" + assertion
    if task_type is HumanEvalPlusTask:
        test = "import numpy as np\ndef check(f):\n    " + assertion
    case = TaskCase("plus", {"prompt": "def f(): pass", "entry_point": "f", "test": test})
    answer = "import numpy as np\ndef f():\n    return " + expression
    result = task.grade_case(case, GenerationResult(answer, {}))
    assert result.status == status, result.error
    assert result.metadata["numpy_version"] == "2.2.6"


def test_plus_missing_dependency_is_not_a_wrong_model_answer(fake_config, monkeypatch):
    task = HumanEvalPlusTask(fake_config)
    monkeypatch.setattr(
        task.executor, "run", lambda *a, **kw: ExecutionResult("FAILED", detail="No module named numpy")
    )
    case = TaskCase("plus", {"prompt": "def f(): pass", "entry_point": "f", "test": "def check(f): assert f() == 1"})

    class NoCallClient:
        def generate(self, messages):
            pytest.fail("Dependency failures must precede model requests")

    assert task.evaluate_case(case, NoCallClient()).status == "EXECUTION_ERROR"


@pytest.mark.parametrize("task", ["humanevalplus", "mbppplus"])
def test_plus_selects_numpy_image_unless_explicitly_overridden(tmp_path, task):
    path = tmp_path / "model.yaml"
    basic = "api_key: dummy\nbase_url: https://example.invalid/v1\nmodel_name: test\n"
    path.write_text(basic, encoding="utf-8")
    assert load_framework_config(path, task=task).run.execution.image == NUMPY_IMAGE
    path.write_text(basic + "execution:\n  image: my-custom-image\n", encoding="utf-8")
    assert load_framework_config(path, task=task).run.execution.image == "my-custom-image"


def test_plus_preflight_failure_stops_runner_before_api(fake_config, monkeypatch):
    import json

    from llm_eval.core.runner import EvaluationRunner

    fake_config.dataset.path.write_text(
        json.dumps(
            {"task_id": "h/0", "prompt": "def f(): pass", "entry_point": "f", "test": "def check(f): assert f() == 1"}
        ),
        encoding="utf-8",
    )
    config = replace(fake_config, run=replace(fake_config.run, task="humanevalplus"))
    monkeypatch.setattr(
        "llm_eval.core.execution.ExecutionEngine.run", lambda *a, **kw: ExecutionResult("FAILED", detail="numpy")
    )
    monkeypatch.setattr("llm_eval.core.runner.OpenAICompatibleClient", lambda *a: pytest.fail("Unexpected API client"))
    summary, _ = EvaluationRunner(config).run()
    assert summary.run_status == "FAILED"
    assert summary.api_attempts == 0
    assert summary.coverage_rate == 0
    assert "NumPy" in summary.error


@pytest.mark.parametrize(
    ("answer", "status"), [("1.00000001", "FAILED"), ("[1.00000001]", "FAILED"), ("[1.00000001, 1.]", "PASSED")]
)
def test_tolerant_plus_comparison_does_not_broadcast_wrong_types_or_lengths(fake_config, answer, status):
    test = (
        "import numpy as np\n"
        "def assertion(out, exp, atol):\n"
        "    if out != exp:\n"
        "        assert np.allclose(out, exp, rtol=1e-07, atol=atol)\n"
        "def check(candidate):\n"
        "    assertion(candidate(), [1., 1.], 1e-6)\n"
    )
    case = TaskCase("h/tolerance", {"entry_point": "f", "prompt": "def f(): pass", "test": test})
    result = HumanEvalPlusTask(fake_config).grade_case(case, GenerationResult("def f(): return " + answer, {}))
    assert result.status == status, result.error


@pytest.mark.parametrize(("answer", "status"), [("-1.0", "PASSED"), ("100.0", "FAILED")])
def test_find_zero_oracle_passes_scalar_root_and_rejects_negative_residual(fake_config, answer, status):
    test = (
        "def _poly(xs, x): return sum(c * x**i for i, c in enumerate(xs))\n"
        "def check(candidate):\n"
        "    inp = [[-1, -1]]\n"
        "    assert _poly(*candidate(*inp), inp) <= 0.0001\n"
    )
    case = TaskCase("HumanEval/32", {"entry_point": "find_zero", "prompt": "def find_zero(xs): pass", "test": test})
    result = HumanEvalPlusTask(fake_config).grade_case(
        case, GenerationResult("def find_zero(xs): return " + answer, {})
    )
    assert result.status == status, result.error
