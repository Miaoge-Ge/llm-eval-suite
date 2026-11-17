from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from itertools import chain
from pathlib import Path
from typing import Any

from .clients import GenerationResult, OpenAICompatibleClient
from .execution import ExecutionEngine, ExecutionUnavailable
from .extraction import (
    dedent_code,
    extract_choice_letter,
    extract_last_boxed,
    extract_python_code,
    indent_block,
    last_numeric_token,
)
from .ifeval import evaluate as evaluate_ifeval
from .livecodebench import iter_private_tests, outputs_match, prepare_assets, validate_tests
from .plus import prepare_test_code
from .settings import NUMPY_VERSION, FrameworkConfig
from .status import Status
from .utils import load_jsonl

DEFAULT_TASK_NAME = "humaneval"

# Populated automatically: any subclass that defines its own `task_name`
# registers itself via BaseEvaluationTask.__init_subclass__.
TASK_REGISTRY: dict[str, type[BaseEvaluationTask]] = {}


@dataclass(frozen=True)
class TaskCase:
    case_id: str
    payload: dict[str, Any]


@dataclass
class TaskResult:
    case_id: str
    status: str
    duration_seconds: float
    http_status_code: int | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_attempts: int = 0
    usage_complete: bool = True
    expected: Any = None
    actual: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    case_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["duration_human"] = f"{self.duration_seconds:.2f}s"
        return payload


class BaseEvaluationTask(ABC):
    task_name: str = ""
    requires_execution = False
    requires_numpy = False
    # Looked up in order to build a case id; falls back to the row index.
    id_fields: tuple[str, ...] = ("id",)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        registered_name = cls.__dict__.get("task_name")
        if registered_name:
            TASK_REGISTRY[registered_name] = cls

    def __init__(self, config: FrameworkConfig) -> None:
        self.config = config
        self.dataset_path = config.dataset.path
        self.executor = ExecutionEngine(config.run.execution)

    def load_cases(self) -> list[TaskCase]:
        rows = load_jsonl(self.dataset_path)
        return [TaskCase(case_id=self.case_id_for(row), payload=row) for row in rows]

    def case_id_for(self, row: dict[str, Any]) -> str:
        for field_name in self.id_fields:
            if field_name in row:
                return str(row[field_name])
        return str(row.get("_row_index", "unknown"))

    def evaluate_case(self, case: TaskCase, client: OpenAICompatibleClient) -> TaskResult:
        started_at = time.time()
        if error := self.validate_case(case):
            return TaskResult(case.case_id, Status.INTERNAL_ERROR, 0.0, error=error)
        if self.requires_numpy:
            try:
                self.check_ready()
            except ExecutionUnavailable as exc:
                return TaskResult(case.case_id, Status.EXECUTION_ERROR, 0.0, error=str(exc))
        generation = client.generate(self.build_messages(case))
        if generation.error:
            return self._api_error_result(case, started_at, generation)
        return self.grade_case(case, generation, started_at)

    def validate_case(self, case: TaskCase) -> str | None:
        return None

    def check_ready(self) -> None:
        self.executor.check_ready()
        if self.requires_numpy:
            self.executor.check_numpy()

    def prepare_assets(self, cases: list[TaskCase], destination: Path, source: Path | None = None) -> None:
        """Tasks with external grading data override this snapshot hook."""
        return None

    def coverage_info(self, cases: list[TaskCase]) -> dict[str, Any]:
        return {}

    @abstractmethod
    def build_messages(self, case: TaskCase) -> list[dict[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def grade_case(
        self,
        case: TaskCase,
        generation: GenerationResult,
        started_at: float | None = None,
    ) -> TaskResult:
        raise NotImplementedError

    def _api_error_result(self, case: TaskCase, started_at: float, generation: GenerationResult) -> TaskResult:
        return self._usage_result(
            case,
            started_at,
            generation,
            status=(
                Status.CANCELLED
                if generation.cancelled
                else Status.API_ERROR_FATAL
                if generation.fatal
                else Status.API_ERROR
            ),
            error=generation.error,
        )

    def _usage_result(
        self,
        case: TaskCase,
        started_at: float,
        generation: GenerationResult,
        status: str,
        **kwargs: Any,
    ) -> TaskResult:
        return TaskResult(
            case_id=case.case_id,
            status=status,
            duration_seconds=time.time() - started_at,
            http_status_code=generation.http_status_code,
            prompt_tokens=generation.usage.get("prompt_tokens", 0),
            completion_tokens=generation.usage.get("completion_tokens", 0),
            total_tokens=generation.usage.get("total_tokens", 0),
            api_attempts=generation.attempts,
            usage_complete=generation.usage_complete,
            **kwargs,
        )


class CodeGenerationTask(BaseEvaluationTask):
    requires_execution = True
    system_prompt = (
        "You are a rigorous Python engineer. "
        "Return only executable Python code. "
        "Do not add Markdown, commentary, or explanations."
    )

    shared_header = "\n".join(
        [
            "from typing import Any, Dict, List, Optional, Set, Tuple, Union",
            "import collections",
            "import functools",
            "import heapq",
            "import itertools",
            "import math",
            "import re",
            "import sys",
        ]
    )

    def build_messages(self, case: TaskCase) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt(case)},
        ]

    @abstractmethod
    def user_prompt(self, case: TaskCase) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_test_program(self, case: TaskCase, generated_code: str) -> str:
        raise NotImplementedError

    def validate_case(self, case: TaskCase) -> str | None:
        return None if self._has_tests(case) else "Dataset case has no tests"

    def grade_case(
        self,
        case: TaskCase,
        generation: GenerationResult,
        started_at: float | None = None,
    ) -> TaskResult:
        started_at = time.time() if started_at is None else started_at
        if self.requires_numpy:
            try:
                self.check_ready()
            except ExecutionUnavailable as exc:
                return self._usage_result(case, started_at, generation, Status.EXECUTION_ERROR, error=str(exc))
        code = self.normalize_candidate(case, generation.content)
        test_program = self.build_test_program(case, code)
        execution = self.executor.run(test_program, self.config.run.execution_timeout_seconds)
        status = execution.status
        details = execution.stdout if status == Status.PASSED else execution.detail
        return self._usage_result(
            case,
            started_at,
            generation,
            status=status,
            actual=details if status != Status.PASSED else "passed",
            error=details if status != Status.PASSED else None,
            metadata={"numpy_version": NUMPY_VERSION} if self.requires_numpy else {},
        )

    def normalize_candidate(self, case: TaskCase, raw_text: str) -> str:
        return extract_python_code(raw_text)

    def _has_tests(self, case: TaskCase) -> bool:
        test = case.payload.get("test")
        return isinstance(test, str) and bool(test.strip())


class HumanEvalTask(CodeGenerationTask):
    task_name = "humaneval"
    id_fields = ("task_id",)

    def user_prompt(self, case: TaskCase) -> str:
        prompt = case.payload["prompt"].rstrip()
        return f"Complete the Python function below. Keep the original signature and docstring.\n\n{prompt}"

    def normalize_candidate(self, case: TaskCase, raw_text: str) -> str:
        code = super().normalize_candidate(case, raw_text)
        entry_point = case.payload["entry_point"]
        if f"def {entry_point}" in code:
            return code
        prompt = case.payload["prompt"].rstrip()
        return f"{prompt}\n{indent_block(dedent_code(code), 4)}"

    def build_test_program(self, case: TaskCase, generated_code: str) -> str:
        return "\n\n".join(
            [
                self.shared_header,
                generated_code,
                case.payload["test"].strip(),
                f"check({case.payload['entry_point']})",
            ]
        )


class HumanEvalPlusTask(HumanEvalTask):
    task_name = "humanevalplus"
    requires_numpy = True

    def build_test_program(self, case: TaskCase, generated_code: str) -> str:
        payload = {**case.payload, "test": prepare_test_code(case.payload["test"], case.payload["entry_point"])}
        return super().build_test_program(TaskCase(case.case_id, payload), generated_code)


class MBPPTask(CodeGenerationTask):
    task_name = "mbpp"
    id_fields = ("task_id",)

    def _has_tests(self, case: TaskCase) -> bool:
        tests = [*(case.payload.get("test_list") or []), *(case.payload.get("challenge_test_list") or [])]
        return any(isinstance(test, str) and test.strip() for test in tests)

    def user_prompt(self, case: TaskCase) -> str:
        tests = case.payload.get("test_list") or []
        function_name_hint = ""
        if tests:
            match = re.search(r"assert\s+(\w+)\(", tests[0])
            if match:
                function_name_hint = f"\nThe public function name must be `{match.group(1)}`."
        return (
            f"Write a Python solution for this MBPP problem:\n{case.payload['text']}"
            f"{function_name_hint}\n"
            "Return only code."
        )

    def build_test_program(self, case: TaskCase, generated_code: str) -> str:
        sections = [self.shared_header, generated_code]
        setup = (case.payload.get("test_setup_code") or "").strip()
        if setup:
            sections.append(setup)
        test_list = list(case.payload.get("test_list") or [])
        challenge_list = list(case.payload.get("challenge_test_list") or [])
        if challenge_list:
            test_list.extend(challenge_list)
        sections.append("\n".join(test_list))
        return "\n\n".join(section for section in sections if section)


class MBPPPlusTask(CodeGenerationTask):
    task_name = "mbppplus"
    id_fields = ("task_id",)

    requires_numpy = True

    def user_prompt(self, case: TaskCase) -> str:
        return (
            f"Write a Python solution for this problem:\n{case.payload['prompt']}\n"
            f"The public function must be named `{case.payload['entry_point']}`.\n"
            "Return only code."
        )

    def build_test_program(self, case: TaskCase, generated_code: str) -> str:
        sections = [self.shared_header, generated_code]
        sections.extend(case.payload.get("test_imports") or [])
        sections.append(prepare_test_code(case.payload["test"], case.payload["entry_point"]).strip())
        return "\n\n".join(section for section in sections if section)


class GSM8KTask(BaseEvaluationTask):
    task_name = "gsm"
    id_fields = ("_row_index",)

    def validate_case(self, case: TaskCase) -> str | None:
        answer = case.payload.get("answer")
        if not isinstance(answer, str) or last_numeric_token(answer) is None:
            return "Dataset case has no valid numeric reference answer"
        return None

    def build_messages(self, case: TaskCase) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "Solve grade-school math carefully. End with a single line in the format `#### <answer>`.",
            },
            {"role": "user", "content": case.payload["question"]},
        ]

    def grade_case(
        self,
        case: TaskCase,
        generation: GenerationResult,
        started_at: float | None = None,
    ) -> TaskResult:
        started_at = time.time() if started_at is None else started_at
        expected = last_numeric_token(case.payload["answer"])
        actual = last_numeric_token(generation.content)
        is_correct = self._numbers_match(actual, expected)
        return self._usage_result(
            case,
            started_at,
            generation,
            status=Status.PASSED if is_correct else Status.FAILED,
            expected=expected,
            actual=actual,
        )

    def _numbers_match(self, actual: str | None, expected: str | None) -> bool:
        if actual is None or expected is None:
            return False
        try:
            actual_number, expected_number = Decimal(actual), Decimal(expected)
            return actual_number.is_finite() and expected_number.is_finite() and actual_number == expected_number
        except InvalidOperation:
            return False


class AIMETask(BaseEvaluationTask):
    task_name = "aime2025"

    def build_messages(self, case: TaskCase) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "Solve the AIME competition math problem step by step. "
                    "The final answer is an integer between 0 and 999. "
                    "Put it inside \\boxed{} on the last line."
                ),
            },
            {"role": "user", "content": case.payload["problem"]},
        ]

    def grade_case(
        self,
        case: TaskCase,
        generation: GenerationResult,
        started_at: float | None = None,
    ) -> TaskResult:
        started_at = time.time() if started_at is None else started_at
        boxed = extract_last_boxed(generation.content)
        # Preserve the whole answer: extracting a number from a fraction or a
        # decimal can silently turn an invalid answer into a correct integer.
        lines = generation.content.strip().splitlines()
        actual = boxed if "\\boxed" in generation.content else (lines[-1] if lines else None)
        expected = str(case.payload["answer"])
        is_correct = self._integers_match(actual, expected)
        return self._usage_result(
            case,
            started_at,
            generation,
            status=Status.PASSED if is_correct else Status.FAILED,
            expected=case.payload["answer"],
            actual=actual,
        )

    def _integers_match(self, actual: str | None, expected: str | None) -> bool:
        if actual is None or expected is None:
            return False
        actual, expected = actual.strip(), expected.strip()
        if not re.fullmatch(r"[0-9]{1,3}", actual) or not re.fullmatch(r"[0-9]{1,3}", expected):
            return False
        return int(actual) == int(expected)


class AIME2026Task(AIMETask):
    task_name = "aime2026"


class MultipleChoiceTask(BaseEvaluationTask):
    """Shared grading skeleton: ask the question, extract a letter, compare."""

    domain_field = "domain"

    @abstractmethod
    def system_prompt(self, row: dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def user_prompt(self, row: dict[str, Any]) -> str:
        raise NotImplementedError

    def last_letter(self, row: dict[str, Any]) -> str:
        return "D"

    def build_messages(self, case: TaskCase) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt(case.payload)},
            {"role": "user", "content": self.user_prompt(case.payload)},
        ]

    def grade_case(
        self,
        case: TaskCase,
        generation: GenerationResult,
        started_at: float | None = None,
    ) -> TaskResult:
        started_at = time.time() if started_at is None else started_at
        row = case.payload
        expected = str(row["answer"]).strip().upper()
        actual = extract_choice_letter(generation.content, last_letter=self.last_letter(row))
        is_correct = actual is not None and actual == expected
        return self._usage_result(
            case,
            started_at,
            generation,
            status=Status.PASSED if is_correct else Status.FAILED,
            expected=expected,
            actual=actual,
            metadata={"domain": row.get(self.domain_field, "")},
        )


class GPQATask(MultipleChoiceTask):
    task_name = "gpqa"

    def system_prompt(self, row: dict[str, Any]) -> str:
        return (
            "Answer the multiple-choice question. Reason briefly, "
            "then give the final answer as \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}."
        )

    def user_prompt(self, row: dict[str, Any]) -> str:
        return row["problem"]


class MMLUProTask(MultipleChoiceTask):
    task_name = "mmlu_pro"
    choice_letters = "ABCDEFGHIJ"
    domain_field = "category"

    def system_prompt(self, row: dict[str, Any]) -> str:
        return (
            "Answer the multiple-choice question. Reason briefly, then give the final "
            "answer as \\boxed{X}, where X is the letter of the correct option."
        )

    def user_prompt(self, row: dict[str, Any]) -> str:
        letters = self._letters(row)
        rendered = "\n".join(f"{letter}. {text}" for letter, text in zip(letters, row["options"], strict=False))
        return f"{row['question']}\n\n{rendered}"

    def last_letter(self, row: dict[str, Any]) -> str:
        return self._letters(row)[-1]

    def _letters(self, row: dict[str, Any]) -> str:
        return self.choice_letters[: len(row["options"])]


class IFEvalTask(BaseEvaluationTask):
    task_name = "ifeval"
    id_fields = ("key",)

    def build_messages(self, case: TaskCase) -> list[dict[str, str]]:
        return [{"role": "user", "content": case.payload["prompt"]}]

    def grade_case(
        self,
        case: TaskCase,
        generation: GenerationResult,
        started_at: float | None = None,
    ) -> TaskResult:
        started_at = time.time() if started_at is None else started_at
        report = evaluate_ifeval(
            case.payload["prompt"],
            generation.content or "",
            list(case.payload["instruction_id_list"]),
            list(case.payload.get("kwargs") or []),
        )
        total = report["instructions_total"]
        strict = report["instructions_followed_strict"]
        loose = report["instructions_followed_loose"]
        return self._usage_result(
            case,
            started_at,
            generation,
            status=Status.PASSED if report["prompt_strict"] else Status.FAILED,
            expected=f"follow all {total} instructions",
            actual=f"strict {strict}/{total}, loose {loose}/{total}",
            metadata={
                "prompt_loose": report["prompt_loose"],
                "instructions_total": total,
                "instructions_followed_strict": strict,
                "instructions_followed_loose": loose,
            },
        )


class LiveCodeBenchTask(BaseEvaluationTask):
    task_name = "livecodebench"
    requires_execution = True
    id_fields = ("question_id",)

    system_prompt = (
        "You are a competitive programming expert. Write a complete Python 3 program "
        "that reads from standard input and prints the answer to standard output. "
        "Return only the program code, without explanations or Markdown."
    )

    def validate_case(self, case: TaskCase) -> str | None:
        tests = case.payload.get("tests")
        if not tests:
            return "Dataset case has no tests"
        try:
            validate_tests(tests)
            if any(test.get("testtype", "stdin") != "stdin" for test in tests):
                return "Functional tests cannot be graded as stdin/stdout"
            private = case.payload.get("private_tests")
            if private is not None:
                if not isinstance(private, dict) or not all(key in private for key in ("path", "sha256", "count")):
                    return "Invalid private test descriptor"
                if type(private["count"]) is not int or private["count"] <= 0:
                    return "Private test count must be positive"
                if not isinstance(private["path"], str) or not re.fullmatch(r"[0-9a-f]{64}", private["sha256"]):
                    return "Invalid private test path or checksum"
            elif case.payload.get("test_scope") == "stdin_public_and_private":
                return "Full test coverage declared but private tests are missing"
        except (TypeError, ValueError):
            return "Invalid LiveCodeBench test data"
        return None

    def prepare_assets(self, cases: list[TaskCase], destination: Path, source: Path | None = None) -> None:
        for case in cases:
            if error := self.validate_case(case):
                raise ValueError(f"{case.case_id}: {error}")
        prepare_assets(cases, source or self.dataset_path.parent, destination)
        self.assets_root = destination

    def coverage_info(self, cases: list[TaskCase]) -> dict[str, Any]:
        full = sum(bool(case.payload.get("private_tests")) for case in cases)
        return {
            "scope": "stdin_public_and_private"
            if full == len(cases) and cases
            else "stdin_public_only"
            if not full
            else "stdin_mixed_coverage",
            "public_only_problems": len(cases) - full,
            "problems_with_private_tests": full,
            "public_tests": sum(len(case.payload.get("tests") or []) for case in cases),
            "private_tests": sum((case.payload.get("private_tests") or {}).get("count", 0) for case in cases),
            "functional_problems_supported": False,
        }

    def build_messages(self, case: TaskCase) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": case.payload["problem"]},
        ]

    def grade_case(
        self,
        case: TaskCase,
        generation: GenerationResult,
        started_at: float | None = None,
    ) -> TaskResult:
        started_at = time.time() if started_at is None else started_at
        public = case.payload.get("tests") or []
        descriptor = case.payload.get("private_tests")
        private = (
            iter_private_tests(getattr(self, "assets_root", self.dataset_path.parent), descriptor) if descriptor else []
        )
        code = extract_python_code(generation.content)
        timeout = self.config.run.execution_timeout_seconds
        tested = 0

        def counted_tests():
            nonlocal tested
            for test in chain(public, private):
                tested += 1
                yield test

        try:
            status, detail = self._run_tests(code, counted_tests(), timeout)
        except (OSError, ValueError, KeyError) as exc:
            status, detail = Status.INTERNAL_ERROR, f"Cannot read test data: {exc}"
        return self._usage_result(
            case,
            started_at,
            generation,
            status=status,
            actual="passed" if status == Status.PASSED else detail,
            error=None if status == Status.PASSED else detail,
            metadata={
                "domain": case.payload.get("difficulty", ""),
                "platform": case.payload.get("platform", ""),
                "test_scope": "stdin_public_and_private" if descriptor else "stdin_public_only",
                "public_tests_total": len(public),
                "private_tests_total": descriptor["count"] if descriptor else 0,
                "tests_executed": tested,
            },
        )

    def _run_tests(self, code: str, tests, timeout: int) -> tuple[str, str]:
        count = 0
        for index, test in enumerate(tests):
            count += 1
            result = self.executor.run(code, timeout, test["input"], assertions=False)
            if result.status != Status.PASSED:
                return result.status, f"test {index}: {result.detail[:160]}"
            if not outputs_match(result.stdout, test["output"]):
                return Status.FAILED, f"test {index}: wrong answer"
        if not count:
            return Status.INTERNAL_ERROR, "Dataset case has no tests"
        return Status.PASSED, ""
