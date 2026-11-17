from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from llm_eval.core.clients import GenerationResult, OpenAICompatibleClient
from llm_eval.core.execution import ExecutionEngine, ExecutionUnavailable
from llm_eval.core.settings import FrameworkConfig
from llm_eval.core.status import Status
from llm_eval.core.utils import load_jsonl

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
    can_grade_failed_generations = False
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
        generation = client.generate(self.build_messages(case), **self.generation_options(case))
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

    def generation_options(self, case: TaskCase) -> dict[str, Any]:
        return {}

    def case_metadata(self, case: TaskCase) -> dict[str, Any]:
        return {}

    def summarize(self, results: list[dict[str, Any]]) -> dict[str, Any]:
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
