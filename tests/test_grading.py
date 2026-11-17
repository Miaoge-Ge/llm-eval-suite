import pytest

from llm_eval.clients import GenerationResult
from llm_eval.extraction import (
    extract_choice_letter,
    extract_last_boxed,
    last_numeric_token,
)
from llm_eval.reporting import _domain_breakdown
from llm_eval.tasks import AIMETask, GPQATask, GSM8KTask, HumanEvalTask, LiveCodeBenchTask, MMLUProTask, TaskCase
from llm_eval.utils import natural_sort_key


class FakeClient:
    def __init__(self, content: str) -> None:
        self.content = content

    def generate(self, messages):
        return GenerationResult(
            content=self.content,
            usage={"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
            http_status_code=200,
        )


def test_extract_last_boxed_handles_nested_braces():
    assert extract_last_boxed(r"so \boxed{\frac{\pi}{2}} done") == r"\frac{\pi}{2}"


def test_extract_last_boxed_returns_last_match():
    assert extract_last_boxed(r"\boxed{A} then \boxed{B}") == "B"


def test_extract_last_boxed_missing_returns_none():
    assert extract_last_boxed("no box here") is None


def test_extract_choice_letter_variants():
    assert extract_choice_letter(r"reasoning ... \boxed{D}") == "D"
    assert extract_choice_letter("I think the answer is (B).") == "B"
    assert extract_choice_letter("Answer: C") == "C"
    assert extract_choice_letter("no clear choice") is None


def test_extract_choice_letter_ignores_lowercase_prose():
    # The article "a" at the end of a line must not be read as answer A.
    assert extract_choice_letter("I think the best option is clearly a") is None
    assert extract_choice_letter("pick choice (b) or others") is None
    # An explicit "answer is" context still accepts lowercase.
    assert extract_choice_letter("The answer is b") == "B"


def test_last_numeric_token_requires_digits():
    assert last_numeric_token("no digits, just commas , here") is None
    assert last_numeric_token("#### 1,234") == "1234"
    assert last_numeric_token("the total is 42.") == "42"


def test_extract_choice_letter_extends_beyond_d():
    assert extract_choice_letter(r"so the answer is \boxed{H}", last_letter="J") == "H"
    # default range stops at D, so H is not picked up
    assert extract_choice_letter(r"\boxed{H}") is None


def test_natural_sort_orders_numeric_suffixes():
    ids = ["HumanEval/10", "HumanEval/2", "HumanEval/1", "HumanEval/20"]
    assert sorted(ids, key=natural_sort_key) == [
        "HumanEval/1",
        "HumanEval/2",
        "HumanEval/10",
        "HumanEval/20",
    ]


def test_domain_breakdown_aggregates_by_metadata():
    results = [
        {"status": "PASSED", "metadata": {"domain": "Physics"}},
        {"status": "FAILED", "metadata": {"domain": "Physics"}},
        {"status": "PASSED", "metadata": {"domain": "Biology"}},
        {"status": "PASSED", "metadata": {}},
    ]
    assert _domain_breakdown(results) == [("Biology", 1, 1), ("Physics", 1, 2)]


def test_aime_grades_integer_answer(fake_config):
    task = AIMETask(fake_config)
    case = TaskCase(case_id="AIME2025-I/0", payload={"problem": "Find x.", "answer": "70"})

    passed = task.evaluate_case(case, FakeClient(r"After working it out, \boxed{070}."))
    assert passed.status == "PASSED"

    failed = task.evaluate_case(case, FakeClient(r"The answer is \boxed{71}."))
    assert failed.status == "FAILED"


def test_livecodebench_grades_stdin_program(fake_config):
    task = LiveCodeBenchTask(fake_config)
    case = TaskCase(
        case_id="lcb/1",
        payload={
            "problem": "Read n, print 2n.",
            "tests": [{"input": "3\n", "output": "6\n"}, {"input": "10\n", "output": "20\n"}],
            "difficulty": "easy",
        },
    )

    passed = task.evaluate_case(case, FakeClient("n = int(input())\nprint(n * 2)"))
    assert passed.status == "PASSED"

    failed = task.evaluate_case(case, FakeClient("n = int(input())\nprint(n + 1)"))
    assert failed.status == "FAILED"


def test_mmlu_pro_grades_letter_beyond_d(fake_config):
    task = MMLUProTask(fake_config)
    case = TaskCase(
        case_id="mmlu_pro/1",
        payload={
            "question": "Pick the right one.",
            "options": ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
            "answer": "H",
            "category": "biology",
        },
    )

    passed = task.evaluate_case(case, FakeClient(r"Reasoning... \boxed{H}"))
    assert passed.status == "PASSED"
    assert passed.metadata["domain"] == "biology"

    failed = task.evaluate_case(case, FakeClient(r"\boxed{A}"))
    assert failed.status == "FAILED"


def test_gpqa_grades_choice_letter(fake_config):
    task = GPQATask(fake_config)
    case = TaskCase(case_id="gpqa/1", payload={"problem": "Q?", "answer": "B", "domain": "Physics"})

    passed = task.evaluate_case(case, FakeClient(r"Reasoning... \boxed{B}"))
    assert passed.status == "PASSED"
    assert passed.metadata["domain"] == "Physics"

    failed = task.evaluate_case(case, FakeClient(r"\boxed{A}"))
    assert failed.status == "FAILED"


@pytest.mark.parametrize("answer", ["70.4", "69.6", "70.0", "140/2", r"\frac{140}{2}", "invalid70", "", "1000", "-70"])
def test_aime_rejects_non_integer_boxed_answers(fake_config, answer):
    case = TaskCase("aime", {"problem": "Find x", "answer": "70"})
    result = AIMETask(fake_config).evaluate_case(case, FakeClient(r"\boxed{" + answer + "}"))
    assert result.status == "FAILED"


@pytest.mark.parametrize("content", [r"\boxed{070}", "Reasoning\n70", r"\boxed{71} then \boxed{70}"])
def test_aime_accepts_whole_final_integer(fake_config, content):
    case = TaskCase("aime", {"problem": "Find x", "answer": "70"})
    assert AIMETask(fake_config).evaluate_case(case, FakeClient(content)).status == "PASSED"


@pytest.mark.parametrize(
    "content", [r"\boxed{70} then \boxed{}", r"\boxed{70} then \boxed{", "Reasoning includes 70 apples"]
)
def test_aime_does_not_fall_back_from_invalid_final_answer(fake_config, content):
    case = TaskCase("aime", {"problem": "Find x", "answer": "70"})
    assert AIMETask(fake_config).evaluate_case(case, FakeClient(content)).status == "FAILED"


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (r"\boxed{Answer: B}", "B"),
        (r"\boxed{answer is (b)}", "B"),
        (r"\boxed{invalid}", None),
        (r"\boxed{A or B}", None),
        (r"\boxed{AB}", None),
        (r"\boxed{A} then \boxed{}", None),
        (r"\boxed{A} then \boxed{", None),
        ("The answer is A. Correction: the answer is B.", "B"),
        ("Answer: A. Final answer: invalid", None),
        ("Answer: A or B", None),
        ("Answer: Banana", None),
        ("Reasoning\nB", "B"),
        ("   ", None),
    ],
)
def test_choice_parser_requires_unambiguous_final_answer(content, expected):
    assert extract_choice_letter(content) == expected


def test_gsm_uses_last_answer_marker():
    assert last_numeric_token("#### 10\nCorrection\n#### 20") == "20"
    assert last_numeric_token("#### 10\nCorrection\n#### unknown") is None
    assert last_numeric_token("#### 10\n#### 20/2") is None


def test_livecodebench_rejects_empty_tests_before_generation(fake_config):
    class NoCallClient:
        def generate(self, messages):
            pytest.fail("Invalid test data must not trigger generation")

    task = LiveCodeBenchTask(fake_config)
    case = TaskCase("lcb", {"problem": "Read input", "tests": []})
    assert task.evaluate_case(case, NoCallClient()).status == "INTERNAL_ERROR"
    assert task._run_tests("raise RuntimeError()", [], 1)[0] == "INTERNAL_ERROR"


@pytest.mark.parametrize(
    ("actual", "expected", "correct"),
    [
        ("9007199254740993", "9007199254740992", False),
        ("1.0000000001", "1", False),
        ("0.0000000001", "0", False),
        ("9" * 400, "8" * 400, False),
        ("18.000", "18", True),
        ("-0", "0", True),
        ("nan", "nan", False),
        ("Infinity", "Infinity", False),
        ("invalid", "invalid", False),
    ],
)
def test_gsm_compares_exact_finite_decimals(fake_config, actual, expected, correct):
    assert GSM8KTask(fake_config)._numbers_match(actual, expected) is correct


@pytest.mark.parametrize("response", ["1/2", "1e2", "2 + 2", "There were 2 apples", "1,2", "#### 1/2"])
def test_gsm_does_not_grade_numeric_fragments(fake_config, response):
    case = TaskCase("gsm", {"question": "Compute the answer", "answer": "#### 2"})
    assert GSM8KTask(fake_config).evaluate_case(case, FakeClient(response)).status == "FAILED"


def test_gsm_invalid_reference_is_not_a_model_failure(fake_config):
    class NoCallClient:
        def generate(self, messages):
            pytest.fail("Invalid reference must not call the API")

    case = TaskCase("gsm", {"question": "Compute", "answer": "#### unknown"})
    assert GSM8KTask(fake_config).evaluate_case(case, NoCallClient()).status == "INTERNAL_ERROR"


@pytest.mark.parametrize("fenced", [False, True])
@pytest.mark.parametrize("indent", ["    ", "\t"])
def test_humaneval_preserves_relative_body_indentation(fake_config, fenced, indent):
    body = f"{indent}if x > 0:\n{indent}{indent}return x\n{indent}return -x\n"
    response = f"```python\n{body}```" if fenced else body
    case = TaskCase(
        "h/body",
        {
            "prompt": 'def absolute(x):\n    """Absolute value."""\n',
            "entry_point": "absolute",
            "test": "def check(f):\n    assert f(2) == 2\n    assert f(-2) == 2",
        },
    )
    result = HumanEvalTask(fake_config).evaluate_case(case, FakeClient(response))
    assert result.status == "PASSED", result
