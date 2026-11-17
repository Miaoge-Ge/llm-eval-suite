from __future__ import annotations

from typing import Any

from llm_eval.benchmark.suite_data import selected_rows
from llm_eval.core.clients import GenerationResult
from llm_eval.core.status import ERROR_STATUSES, GRADED_STATUSES, Status
from llm_eval.core.suites import load_suite
from llm_eval.core.task import TASK_REGISTRY, BaseEvaluationTask, TaskCase, TaskResult


class Quick40Task(BaseEvaluationTask):
    """Original prompts and full existing graders, selected by a pinned manifest."""

    task_name = "quick40"
    requires_execution = True
    requires_numpy = True

    def load_cases(self) -> list[TaskCase]:
        manifest = load_suite(self.dataset_path, "quick40-v1", 40)
        rows = selected_rows(manifest)
        for dimension in ("code", "math", "knowledge", "instructions"):
            if sum(row["dimension"] == dimension for row in rows) != 10:
                raise ValueError("quick40 must contain ten cases per dimension")
        return [TaskCase(row["id"], row) for row in rows]

    def _delegate(self, case: TaskCase) -> tuple[BaseEvaluationTask, TaskCase]:
        name = case.payload["source_task"]
        if name not in {"humanevalplus", "mbppplus", "gsm", "aime2025", "mmlu_pro", "gpqa", "ifeval"}:
            raise ValueError(f"Unsupported quick40 source: {name}")
        delegate = TASK_REGISTRY[name](self.config)
        delegate.executor = self.executor
        return delegate, TaskCase(case.payload["source_id"], case.payload["source_case"])

    def validate_case(self, case: TaskCase) -> str | None:
        task, original = self._delegate(case)
        return task.validate_case(original)

    def build_messages(self, case: TaskCase) -> list[dict[str, str]]:
        task, original = self._delegate(case)
        return task.build_messages(original)

    def grade_case(self, case: TaskCase, generation: GenerationResult, started_at: float | None = None) -> TaskResult:
        task, original = self._delegate(case)
        result = task.grade_case(original, generation, started_at)
        result.case_id = case.case_id
        result.metadata.update(self.case_metadata(case))
        finish = generation.attempt_records[-1].get("finish_reason") if generation.attempt_records else None
        result.metadata["truncated"] = finish == "length"
        return result

    def case_metadata(self, case: TaskCase) -> dict[str, Any]:
        return {key: case.payload[key] for key in ("dimension", "source_task", "source_id", "suite_version", "reason")}

    def _api_error_result(self, case: TaskCase, started_at: float, generation: GenerationResult) -> TaskResult:
        result = super()._api_error_result(case, started_at, generation)
        finish = generation.attempt_records[-1].get("finish_reason") if generation.attempt_records else None
        result.metadata["truncated"] = finish == "length"
        return result

    def summarize(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        dimensions = {}
        for dimension in ("code", "math", "knowledge", "instructions"):
            rows = [row for row in results if row.get("metadata", {}).get("dimension") == dimension]
            dimensions[dimension] = {
                "planned": 10,
                "passed": sum(row["status"] == Status.PASSED for row in rows),
                "graded": sum(row["status"] in GRADED_STATUSES for row in rows),
                "errors": sum(row["status"] in ERROR_STATUSES for row in rows),
                "cancelled": sum(row["status"] == Status.CANCELLED for row in rows),
                "truncated": sum(bool(row.get("metadata", {}).get("truncated")) for row in rows),
            }
        return {
            "suite_version": "quick40-v1",
            "dimensions": dimensions,
            "interpretation": "Curated coverage sample; not a calibrated benchmark or model ranking.",
        }
