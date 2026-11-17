from __future__ import annotations

import concurrent.futures
import json
import threading
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .artifacts import RunArchive, RunLock, copy_generations, identity, load_snapshot
from .clients import GenerationResult, OpenAICompatibleClient, request_parameters
from .reporting import ResultWriter, RunSummary, render_markdown_report
from .settings import FrameworkConfig
from .status import ERROR_STATUSES, GRADED_STATUSES, RunStatus, Status
from .tasks import TASK_REGISTRY, BaseEvaluationTask, TaskCase, TaskResult
from .utils import format_seconds


class EvaluationRunner:
    def __init__(
        self,
        config: FrameworkConfig,
        *,
        operation: str = "run",
        resume_dir: Path | None = None,
        source_dir: Path | None = None,
    ) -> None:
        if operation not in {"run", "generate", "grade"}:
            raise ValueError(f"Unsupported operation: {operation}")
        self.config = config
        self.operation = operation
        self.resume_dir = resume_dir.resolve() if resume_dir else None
        self.source_dir = source_dir.resolve() if source_dir else None
        self.cancel_event = threading.Event()
        self.client: Any = None
        self._owned_client: OpenAICompatibleClient | None = None
        self._client_lock = threading.Lock()
        self.archive: RunArchive | None = None
        self._unpersisted_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._unpersisted_attempts = 0

    def build_task(self) -> BaseEvaluationTask:
        task_factory = TASK_REGISTRY.get(self.config.run.task)
        if not task_factory:
            available = ", ".join(sorted(TASK_REGISTRY))
            raise ValueError(f"Unsupported task '{self.config.run.task}'. Available tasks: {available}")
        task = task_factory(self.config)
        task.executor.cancel_event = self.cancel_event
        return task

    def _get_client(self) -> Any:
        if self.operation == "grade":
            raise RuntimeError("Offline grading must never construct an API client")
        with self._client_lock:
            if self.client is None:
                self._owned_client = OpenAICompatibleClient(self.config)
                self._owned_client.cancel_event = self.cancel_event
                self.client = self._owned_client
        return self.client

    def _stop(self) -> None:
        self.cancel_event.set()
        cancel = getattr(self.client, "cancel", None)
        if cancel is not None:
            cancel()

    def _prepare(
        self,
        task: BaseEvaluationTask,
        writer: ResultWriter,
    ) -> tuple[list[TaskCase], dict[int, dict[str, Any]]]:
        if self.source_dir:
            with RunLock(self.source_dir):
                manifest, cases, source = load_snapshot(self.source_dir)
                if manifest["identity"]["task"] != self.config.run.task:
                    raise ValueError("Offline grading task does not match the saved run")
                archive = RunArchive(writer.paths.root, cases)
                archive.create(self.config, identity(self.config, task, cases), "grade", str(self.source_dir))
                copy_generations(source, archive)
                task.prepare_assets(cases, writer.paths.root, source=self.source_dir)
            self.archive = archive
            return cases, {}
        cases = task.load_cases()
        if self.config.run.limit is not None:
            cases = cases[: self.config.run.limit]
        if not cases:
            raise ValueError(f"No cases found in dataset: {self.config.dataset.path}")
        current_identity = identity(self.config, task, cases)
        if self.resume_dir:
            manifest, saved_cases, archive = load_snapshot(self.resume_dir)
            if manifest["operation"] not in {"run", "generate"}:
                raise ValueError("Use grade again to regenerate an offline grading run")
            if manifest["identity"] != current_identity:
                raise ValueError(
                    "Resume fingerprint mismatch: dataset, prompts, model, grading code or settings changed"
                )
            self.operation = manifest["operation"]
            checkpoints = archive.checkpoints()
            task.prepare_assets(saved_cases, writer.paths.root, source=self.resume_dir)
            previous_path = self.resume_dir / "summary.json"
            archived_usage, archived_attempts = archive.usage()
            previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else {}
            storage_errors = [item for item in checkpoints.values() if item["status"] == Status.STORAGE_ERROR]
            for key in self._unpersisted_usage:
                self._unpersisted_usage[key] = max(
                    previous.get("token_usage", {}).get(key, 0) - archived_usage[key],
                    sum(item.get(key, 0) for item in storage_errors),
                    0,
                )
            self._unpersisted_attempts = max(
                (previous.get("api_attempts") or 0) - archived_attempts,
                sum(item.get("api_attempts", 0) for item in storage_errors),
                0,
            )
            self.archive = archive
            return saved_cases, checkpoints
        archive = RunArchive(writer.paths.root, cases)
        archive.create(self.config, current_identity, self.operation)
        task.prepare_assets(cases, writer.paths.root)
        self.archive = archive
        return cases, {}

    def run(self) -> tuple[RunSummary, ResultWriter]:
        cases: list[TaskCase] = []
        results: dict[int, dict[str, Any]] = {}
        totals: Counter[str] = Counter()
        started_at = time.monotonic()
        run_status = RunStatus.RUNNING
        error: str | None = None
        pool: concurrent.futures.ThreadPoolExecutor | None = None
        pending: dict[concurrent.futures.Future[TaskResult], int] = {}

        with ResultWriter(self.config, root=self.resume_dir) as writer:
            task = self.build_task()
            preparation_error = None
            try:
                cases, checkpoints = self._prepare(task, writer)
            except Exception as exc:
                if self.resume_dir or self.source_dir:
                    # Refuse before changing any existing run artifacts.
                    raise
                checkpoints = {}
                preparation_error = exc
            writer.start(list(checkpoints.values()) if self.resume_dir else None)
            terminal = (*GRADED_STATUSES, Status.GENERATED)
            results.update({index: result for index, result in checkpoints.items() if result["status"] in terminal})
            totals.update(item["status"] for item in results.values())

            def record(index: int, result: TaskResult) -> None:
                if index in results:
                    return
                result.case_index = index
                payload = result.to_dict()
                writer.write_result(payload)
                results[index] = payload
                totals[result.status] += 1
                writer.write_state(
                    RunStatus.RUNNING,
                    total_cases=len(cases),
                    completed_cases=sum(totals.values()) - totals[Status.CANCELLED],
                    cancelled_cases=totals[Status.CANCELLED],
                    operation=self.operation,
                )

            try:
                if preparation_error is not None:
                    raise preparation_error
                indices = [index for index in range(len(cases)) if index not in results]
                if task.requires_execution and self.operation != "generate" and indices:
                    task.check_ready()
                print(
                    f"Starting {self.operation}: task={self.config.run.task} model={self.config.model.model_name}\n"
                    f"  Cases: {len(cases)}; retained: {len(results)}; workers: {self.config.run.workers}\n"
                    f"  Output: {writer.paths.root}",
                    flush=True,
                )
                pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.config.run.workers)
                next_position = 0
                with tqdm(
                    total=len(cases), initial=len(results), desc=self.config.run.task, unit="case", dynamic_ncols=True
                ) as progress:
                    while pending or next_position < len(indices):
                        try:
                            if self.cancel_event.is_set() and run_status == RunStatus.RUNNING:
                                run_status, error = RunStatus.INTERRUPTED, "Evaluation interrupted"
                                self._stop()
                            while (
                                not self.cancel_event.is_set()
                                and next_position < len(indices)
                                and len(pending) < self.config.run.workers
                            ):
                                index = indices[next_position]
                                pending[pool.submit(self._evaluate_case, task, cases[index], index)] = index
                                next_position += 1
                            if not pending:
                                break
                            done, _ = concurrent.futures.wait(
                                pending,
                                timeout=0.1,
                                return_when=concurrent.futures.FIRST_COMPLETED,
                            )
                            for future in done:
                                index = pending[future]
                                result = future.result()
                                record(index, result)
                                del pending[future]
                                progress.update(1)
                                if result.status in (
                                    Status.API_ERROR_FATAL,
                                    Status.EXECUTION_ERROR,
                                    Status.STORAGE_ERROR,
                                ):
                                    if run_status == RunStatus.RUNNING:
                                        run_status, error = RunStatus.ABORTED, result.error or result.status
                                    self._stop()
                                progress.set_postfix_str(
                                    f"passed={totals[Status.PASSED]} failed={totals[Status.FAILED]}",
                                    refresh=False,
                                )
                        except KeyboardInterrupt:
                            run_status, error = RunStatus.INTERRUPTED, "Evaluation interrupted by user"
                            self._stop()
            except KeyboardInterrupt:
                run_status, error = RunStatus.INTERRUPTED, "Evaluation interrupted by user"
                self._stop()
            except Exception as exc:
                run_status, error = RunStatus.FAILED, str(exc)
                self._stop()
            finally:
                if pool is not None:
                    pool.shutdown(wait=True, cancel_futures=True)
                    for future, index in pending.items():
                        if not future.cancelled():
                            record(index, future.result())
                if self._owned_client is not None:
                    self._owned_client.client.close()

            for index, case in enumerate(cases):
                if index not in results:
                    record(index, TaskResult(case.case_id, Status.CANCELLED, 0.0, error=error or "Not scheduled"))
            if run_status == RunStatus.RUNNING:
                if self.cancel_event.is_set():
                    run_status, error = RunStatus.INTERRUPTED, "Evaluation interrupted"
                elif any(totals[status] for status in ERROR_STATUSES) or totals[Status.CANCELLED]:
                    run_status, error = RunStatus.FAILED, "Some cases could not be evaluated"
                else:
                    run_status = RunStatus.COMPLETED
            wall = time.monotonic() - started_at
            graded = sum(totals[status] for status in GRADED_STATUSES)
            completed = sum(totals.values()) - totals[Status.CANCELLED]
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            new_usage = dict(usage)
            attempts = new_attempts = 0
            if self.archive is not None and self.operation != "grade":
                usage, attempts = self.archive.usage()
                new_usage, new_attempts = self.archive.usage(new_only=True)
            payloads = list(results.values())
            incomplete_usage_cases = 0
            if self.archive is not None and self.operation != "grade":
                incomplete_usage_cases = len(
                    {r["case_index"] for r in self.archive.generations if not r["generation"]["usage_complete"]}
                )
                storage_errors = [item for item in payloads if item["status"] == Status.STORAGE_ERROR]
                for key in usage:
                    current = sum(item.get(key, 0) for item in storage_errors)
                    usage[key] += self._unpersisted_usage[key] + current
                    new_usage[key] += current
                current_attempts = sum(item.get("api_attempts", 0) for item in storage_errors)
                attempts += self._unpersisted_attempts + current_attempts
                new_attempts += current_attempts
            summary = RunSummary(
                task=self.config.run.task,
                model=self.config.model.model_name,
                total_cases=len(cases),
                completed_cases=completed,
                pass_rate=totals[Status.PASSED] / graded if graded else 0.0,
                wall_clock_seconds=wall,
                wall_clock_human=format_seconds(wall),
                average_case_seconds=sum(item["duration_seconds"] for item in payloads) / completed
                if completed
                else 0.0,
                throughput_tokens_per_second=new_usage["total_tokens"] / wall if wall else 0.0,
                status_counts={key: value for key, value in totals.items() if value},
                token_usage=usage,
                run_status=run_status,
                error=error,
                cancelled_cases=totals[Status.CANCELLED],
                operation=self.operation,
                api_attempts=attempts,
                new_api_attempts=new_attempts,
                new_token_usage=new_usage,
                source_run=str(self.source_dir) if self.source_dir else None,
                incomplete_usage_cases=incomplete_usage_cases,
                test_coverage=task.coverage_info(cases),
            )
            writer.finalize_results(payloads)
            writer.write_report(render_markdown_report(summary, self.config, payloads))
            writer.write_summary(summary)
            return summary, writer

    def _evaluate_case(self, task: BaseEvaluationTask, case: TaskCase, index: int) -> TaskResult:
        started = time.time()
        generation: GenerationResult | None = None
        if self.cancel_event.is_set():
            return TaskResult(case.case_id, Status.CANCELLED, 0.0, error="Evaluation cancelled")
        try:
            if error := task.validate_case(case):
                return TaskResult(case.case_id, Status.INTERNAL_ERROR, 0.0, error=error)
            assert self.archive is not None
            saved = self.archive.latest.get(index)
            if saved and not saved["generation"]["error"] and not saved["generation"]["cancelled"]:
                generation = GenerationResult(**saved["generation"])
            elif self.operation == "grade":
                return TaskResult(
                    case.case_id,
                    Status.MISSING_GENERATION,
                    0.0,
                    error="No complete saved model answer; offline grading never calls the API",
                )
            else:
                messages = task.build_messages(case)
                generation_started = time.monotonic()
                generation = self._get_client().generate(messages)
                try:
                    saved = self.archive.save_generation(
                        index,
                        messages,
                        generation,
                        request_parameters(self.config),
                        time.monotonic() - generation_started,
                    )
                except OSError as exc:
                    return task._usage_result(
                        case, started, generation, Status.STORAGE_ERROR, error=f"Cannot persist raw answer: {exc}"
                    )
            assert saved is not None
            if self.operation == "grade":
                generation = replace(generation, usage={}, attempts=0)
            if generation.error:
                result = task._api_error_result(case, started, generation)
            elif self.operation == "generate":
                result = task._usage_result(case, started, generation, Status.GENERATED, actual="answer saved")
            else:
                result = task.grade_case(case, generation, started)
            result.metadata["generation_id"] = saved["generation_id"]
            return result
        except Exception as exc:
            if generation is not None:
                return task._usage_result(case, started, generation, Status.INTERNAL_ERROR, error=str(exc))
            return TaskResult(case.case_id, Status.INTERNAL_ERROR, time.time() - started, error=str(exc))
