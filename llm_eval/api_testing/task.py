from __future__ import annotations

import time
from typing import Any

from llm_eval.api_testing.metrics import request_metrics, summarize_experience
from llm_eval.core.clients import GenerationResult
from llm_eval.core.status import Status
from llm_eval.core.suites import load_suite
from llm_eval.core.task import BaseEvaluationTask, TaskCase, TaskResult


class ApiExperienceTask(BaseEvaluationTask):
    task_name = "api_experience"
    can_grade_failed_generations = True

    def load_cases(self) -> list[TaskCase]:
        manifest = load_suite(self.dataset_path, "api-experience-v1", 30)
        rows = manifest["cases"]
        for scenario in ("short_answer", "long_answer", "long_context"):
            if sum(row["scenario"] == scenario for row in rows) != 10:
                raise ValueError("API experience must contain ten cases per scenario")
        return [TaskCase(row["id"], row) for row in rows]

    def build_messages(self, case: TaskCase) -> list[dict[str, str]]:
        return case.payload["messages"]

    def generation_options(self, case: TaskCase) -> dict[str, Any]:
        run = self.config.run
        return {
            "max_attempts": 1,
            "capture_timing": True,
            "parameters": {run.output_token_parameter: run.max_output_tokens or case.payload["max_output_tokens"]},
        }

    def case_metadata(self, case: TaskCase) -> dict[str, Any]:
        return {
            "scenario": case.payload["scenario"],
            "suite_version": "api-experience-v1",
            "reference_input_tokens": case.payload["reference_input_tokens"],
            "reference_tokenizer": "cl100k_base",
            "target_output_tokens": case.payload["target_output_tokens"],
        }

    def _metrics(self, generation: GenerationResult) -> dict[str, Any]:
        if not generation.attempt_records:
            return {"attempted": False}
        run = self.config.run
        return request_metrics(
            generation.attempt_records[0],
            thinking=run.thinking_enabled,
            stall_threshold=run.stall_threshold_seconds,
            slo={"ttft_seconds": run.slo_ttft_seconds, "tps": run.slo_tps, "max_gap_seconds": run.slo_max_gap_seconds},
        )

    def grade_case(self, case: TaskCase, generation: GenerationResult, started_at: float | None = None) -> TaskResult:
        metrics = self._metrics(generation)
        return self._usage_result(
            case,
            started_at or time.time(),
            generation,
            Status.PASSED if metrics.get("success") else Status.API_ERROR,
            metadata={"experience": metrics},
        )

    def _usage_result(
        self,
        case: TaskCase,
        started_at: float,
        generation: GenerationResult,
        status: str,
        **kwargs: Any,
    ) -> TaskResult:
        result = super()._usage_result(case, started_at, generation, status, **kwargs)
        # An observed request still counts if persisting or processing its answer fails.
        result.metadata["experience"] = self._metrics(generation)
        return result

    def summarize(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        return summarize_experience(results, concurrency=self.config.run.workers)
