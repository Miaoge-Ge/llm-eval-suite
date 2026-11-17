"""Private JSON-lines RPC for the C++ service's bounded official-grader pool.

This process never receives an API key and never creates a model client. Model
requests, sessions and job scheduling belong to the native service.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import threading
import time
import zipfile
from collections import Counter, OrderedDict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from llm_eval.core.artifacts import RunArchive, identity, load_snapshot, write_json
from llm_eval.core.clients import GenerationResult, request_parameters
from llm_eval.core.reporting import ResultWriter, RunSummary, render_markdown_report
from llm_eval.core.settings import (
    DEFAULT_DATASET_PATHS,
    NUMPY_IMAGE,
    PLUS_TASKS,
    DatasetConfig,
    ExecutionConfig,
    FrameworkConfig,
    ModelConfig,
    RunConfig,
    config_from_snapshot,
)
from llm_eval.core.status import ERROR_STATUSES, GRADED_STATUSES, Status
from llm_eval.core.utils import format_seconds, repo_root
from llm_eval.tasks import TASK_REGISTRY, BaseEvaluationTask, TaskCase, TaskResult

_cache: OrderedDict[str, tuple[FrameworkConfig, BaseEvaluationTask, list[TaskCase]]] = OrderedDict()


@contextlib.contextmanager
def cancellation(task: BaseEvaluationTask, root: Path):
    stopped = threading.Event()

    def watch() -> None:
        while not stopped.wait(0.1):
            if (root / ".cancelled").exists():
                task.executor.cancel_event.set()
                return

    task.executor.cancel_event.clear()
    if (root / ".cancelled").exists():
        task.executor.cancel_event.set()
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        yield
    finally:
        stopped.set()
        watcher.join()


def checked_root(value: str) -> Path:
    root = Path(value).resolve()
    state = Path(os.environ["LLM_EVAL_STATE_DIR"]).resolve()
    if not root.is_relative_to(state / "jobs") or root == state / "jobs":
        raise ValueError("Run must stay in the service job directory")
    return root


def context(root: Path) -> tuple[FrameworkConfig, BaseEvaluationTask, list[TaskCase]]:
    key = str(root)
    if key not in _cache:
        manifest, cases, _ = load_snapshot(root)
        config = config_from_snapshot(manifest["config"], root)
        _cache[key] = (config, TASK_REGISTRY[config.run.task](config), cases)
        if len(_cache) > 4:
            _cache.popitem(last=False)
    _cache.move_to_end(key)
    return _cache[key]


def prepare(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    task_name = "api_experience" if spec["mode"] == "api" else spec["task"]
    limit = spec.get("limit")
    if task_name not in TASK_REGISTRY or (task_name in {"quick40", "api_experience"} and limit is not None):
        raise ValueError("Invalid task or fixed-suite limit")
    config = FrameworkConfig(
        model=ModelConfig("native-no-api-key", spec["base_url"], spec["model"], spec.get("timeout_seconds", 120)),
        dataset=DatasetConfig(task_name, repo_root() / DEFAULT_DATASET_PATHS[task_name]),
        run=RunConfig(
            task=task_name,
            workers=spec["workers"],
            output_dir=root.parent.parent,
            execution=ExecutionConfig(image=NUMPY_IMAGE if task_name in PLUS_TASKS else "python:3.12-slim"),
            max_output_tokens=spec["max_output_tokens"],
            thinking_enabled=spec["thinking_enabled"],
            limit=limit,
        ),
        model_config_path=root / "native-request",
    )
    task = TASK_REGISTRY[task_name](config)
    cases = task.load_cases()
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise ValueError("Dataset has no cases")
    for case in cases:
        if error := task.validate_case(case):
            raise ValueError(error)
    if task.requires_execution:
        with cancellation(task, root):
            task.check_ready()
    root.mkdir(parents=True, exist_ok=True)
    archive = RunArchive(root, cases)
    archive.create(config, identity(config, task, cases), "run")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["native_runtime"] = {
        "language": "C++20",
        "sources_sha256": {
            str(path.relative_to(repo_root())): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((repo_root() / "server").glob("*"))
            if path.is_file() and path.suffix in {".cpp", ".hpp", ".txt"}
        },
    }
    write_json(manifest_path, manifest)
    write_json(root / "config.json", config.to_public_dict())
    task.prepare_assets(cases, root)
    requests = []
    for case in cases:
        options = task.generation_options(case)
        body = {**request_parameters(config), **options.get("parameters", {}), "messages": task.build_messages(case)}
        body.update(body.pop("extra_body", {}))
        requests.append(
            {
                "case_id": case.case_id,
                "body": body,
                "capture_timing": options.get("capture_timing", False),
                "max_attempts": options.get("max_attempts", 3),
            }
        )
    return {"requests": requests, "total_cases": len(cases)}


def grade(root: Path, data: dict[str, Any]) -> dict[str, Any]:
    config, task, cases = context(root)
    index = data["index"]
    case = cases[index]
    generation = GenerationResult(**data["generation"])
    archive = RunArchive(root, cases)
    archive.load_generations()
    request = data["request"]
    # API keys and authentication headers are never part of this protocol.
    saved = archive.latest.get(index) or archive.save_generation(
        index, request["messages"], generation, request, data["duration_seconds"]
    )
    started = time.time() - data["duration_seconds"]
    if generation.error:
        result = task._api_error_result(case, started, generation)
    else:
        with cancellation(task, root):
            result = task.grade_case(case, generation, started)
    result.case_index = index
    result.metadata.update(task.case_metadata(case))
    result.metadata["generation_id"] = saved["generation_id"]
    row = result.to_dict()
    write_json(root / "cases" / f"{index:08d}.json", row)
    return row


def finish(root: Path, data: dict[str, Any]) -> dict[str, Any]:
    config, task, cases = context(root)
    # Recover completed HTTP responses even when a scorer died before archiving.
    for pending in (root / "native_pending").glob("*.json"):
        record = json.loads(pending.read_text(encoding="utf-8"))
        recovered = RunArchive(root, cases)
        recovered.load_generations()
        index = record["index"]
        if index not in recovered.latest:
            request = record["request"]
            recovered.save_generation(
                index,
                request["messages"],
                GenerationResult(**record["generation"]),
                request,
                record["duration_seconds"],
            )
    _, _, archive = load_snapshot(root)
    rows = archive.checkpoints()
    error = data.get("error")
    for index, case in enumerate(cases):
        if index not in rows:
            saved = archive.latest.get(index)
            result = TaskResult(
                case.case_id,
                Status.INTERNAL_ERROR if saved else Status.CANCELLED,
                0,
                error=error or ("Scoring did not complete" if saved else "Not scheduled"),
                case_index=index,
            )
            if saved:
                result.metadata["generation_id"] = saved["generation_id"]
            result.metadata.update(task.case_metadata(case))
            rows[index] = result.to_dict()
            write_json(root / "cases" / f"{index:08d}.json", rows[index])
    ordered = [rows[index] for index in sorted(rows)]
    counts = Counter(row["status"] for row in ordered)
    usage, attempts = archive.usage()
    status = data.get("status", "COMPLETED")
    if status == "COMPLETED" and (any(counts[item] for item in ERROR_STATUSES) or counts[Status.CANCELLED]):
        status, error = "FAILED", "Some cases could not be evaluated"
    graded = sum(counts[item] for item in GRADED_STATUSES)
    completed = len(cases) - counts[Status.CANCELLED]
    wall = data["wall_clock_seconds"]
    summary = RunSummary(
        task=config.run.task,
        model=config.model.model_name,
        total_cases=len(cases),
        completed_cases=completed,
        pass_rate=counts[Status.PASSED] / graded if graded else 0,
        wall_clock_seconds=wall,
        wall_clock_human=format_seconds(wall),
        average_case_seconds=sum(row["duration_seconds"] for row in ordered) / completed if completed else 0,
        throughput_tokens_per_second=usage["total_tokens"] / wall if wall else 0,
        status_counts=dict(counts),
        token_usage=usage,
        run_status=status,
        error=error,
        cancelled_cases=counts[Status.CANCELLED],
        api_attempts=attempts,
        new_api_attempts=attempts,
        new_token_usage=usage,
        incomplete_usage_cases=sum(not row["generation"]["usage_complete"] for row in archive.generations),
        test_coverage=task.coverage_info(cases),
        metrics=task.summarize(ordered),
    )
    if config.run.task == "api_experience":
        summary = replace(summary, pass_rate=summary.metrics["overall"]["first_attempt_success_rate"] or 0)
    writer = ResultWriter(config, root=root)
    writer.finalize_results(ordered)
    writer.write_report(render_markdown_report(summary, config, ordered))
    writer.write_summary(summary)
    _cache.pop(str(root), None)
    return {"summary": asdict(summary), "cases": ordered}


def export(root: Path) -> dict[str, Any]:
    destination = root / "artifacts.zip"
    temporary = root / ".artifacts.zip.tmp"
    if not destination.exists():
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in root.rglob("*"):
                    if path.is_file() and path.suffix in {".json", ".jsonl", ".md", ".gz"}:
                        kind = zipfile.ZIP_STORED if path.suffix == ".gz" else zipfile.ZIP_DEFLATED
                        archive.write(path, path.relative_to(root), compress_type=kind)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return {"path": str(destination), "bytes": destination.stat().st_size}


def main() -> None:
    for line in sys.stdin:
        try:
            data = json.loads(line)
            root = checked_root(data["root"])
            os.environ["LLM_EVAL_JOB_ID"] = root.relative_to(Path(os.environ["LLM_EVAL_STATE_DIR"]) / "jobs").parts[0]
            with contextlib.redirect_stdout(sys.stderr):
                action = data["action"]
                if action == "prepare":
                    result = prepare(root, data["spec"])
                elif action == "grade":
                    result = grade(root, data)
                elif action == "finish":
                    result = finish(root, data)
                elif action == "export":
                    result = export(root)
                else:
                    raise ValueError("Unknown grader operation")
            response = {"ok": True, "result": result}
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
