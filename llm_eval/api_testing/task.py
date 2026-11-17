from __future__ import annotations

import json
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
        version = json.loads(self.dataset_path.read_text(encoding="utf-8")).get("version")
        count = {"api-experience-v1": 30, "api-experience-v2": 60}.get(version)
        if count is None:
            raise ValueError("Unsupported API experience suite version")
        manifest = load_suite(self.dataset_path, version, count)
        rows = manifest["cases"]
        scenarios = ["short_answer", "long_answer", "long_context"]
        if count == 60:
            scenarios += ["structured_output", "multi_turn", "multilingual"]
        for scenario in scenarios:
            if sum(row["scenario"] == scenario for row in rows) != 10:
                raise ValueError("API experience must contain ten cases per scenario")
        return [TaskCase(row["id"], {**row, "suite_version": version}) for row in rows]

    def build_messages(self, case: TaskCase) -> list[dict[str, str]]:
        return case.payload["messages"]

    def generation_options(self, case: TaskCase) -> dict[str, Any]:
        run = self.config.run
        parameters: dict[str, Any] = {
            run.output_token_parameter: run.max_output_tokens or case.payload["max_output_tokens"]
        }
        if case.payload["scenario"] == "structured_output":
            parameters["response_format"] = {"type": "json_object"}
        return {
            "max_attempts": 1,
            "capture_timing": True,
            "parameters": parameters,
        }

    def case_metadata(self, case: TaskCase) -> dict[str, Any]:
        return {
            "scenario": case.payload["scenario"],
            "suite_version": case.payload.get("suite_version", "api-experience-v1"),
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
        check = case.payload.get("output_check")
        if check:
            observed = result.metadata["experience"]
            passed = None
            if observed.get("success") and not observed.get("truncated"):
                if check["kind"] == "json_object":
                    try:
                        value = json.loads(generation.content)
                        # JSON booleans and numbers must not compare as equal.
                        passed = json.dumps(value, sort_keys=True, ensure_ascii=False) == json.dumps(
                            check["expected"], sort_keys=True, ensure_ascii=False
                        )
                    except (ValueError, TypeError):
                        passed = False
                else:
                    passed = generation.content.strip() == check["expected"]
            observed["output_check"] = {"kind": check["kind"], "passed": passed}
        return result

    def summarize(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        return summarize_experience(results, concurrency=self.config.run.workers)
