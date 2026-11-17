"""A bounded evaluation queue with public, restart-safe job records."""

from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr

from llm_eval.core.artifacts import write_json
from llm_eval.core.runner import EvaluationRunner
from llm_eval.core.settings import (
    DEFAULT_DATASET_PATHS,
    NUMPY_IMAGE,
    PLUS_TASKS,
    DatasetConfig,
    ExecutionConfig,
    FrameworkConfig,
    ModelConfig,
    RunConfig,
)
from llm_eval.core.utils import repo_root


class JobRequest(BaseModel):
    mode: Literal["benchmark", "api"]
    task: str = "quick40"
    base_url: str = "https://api.deepseek.com/v1"
    model: str = Field(default="deepseek-flash", min_length=1, max_length=200)
    api_key: SecretStr = SecretStr("")
    workers: int = Field(default=10, ge=1, le=10, strict=True)
    max_output_tokens: int = Field(default=16384, ge=1, le=65536, strict=True)
    thinking_enabled: bool = True
    limit: int | None = Field(default=None, ge=1, le=5000, strict=True)


class JobManager:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evaluation")
        self.lock = threading.RLock()
        self.records: dict[str, dict[str, Any]] = {}
        self.runners: dict[str, EvaluationRunner] = {}
        self.futures: dict[str, Future[None]] = {}
        for path in self.root.glob("*/job.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if record["status"] in {"QUEUED", "RUNNING", "CANCELLING"}:
                    record.update(status="INTERRUPTED", error="服务重启，任务已中断；原始结果仍保留。")
                    write_json(path, record)
                self.records[record["id"]] = record
            except (OSError, ValueError, KeyError):
                continue

    def submit(self, request: JobRequest, api_key: str) -> dict[str, Any]:
        task = "api_experience" if request.mode == "api" else request.task
        if task not in DEFAULT_DATASET_PATHS or (request.mode == "benchmark" and task == "api_experience"):
            raise ValueError("请选择有效的 Benchmark 题库。")
        if task in {"quick40", "api_experience"} and request.limit is not None:
            raise ValueError("固定题包必须完整运行，不能限制题量。")
        with self.lock:
            if sum(row["status"] in {"QUEUED", "RUNNING", "CANCELLING"} for row in self.records.values()) >= 5:
                raise ValueError("等待中的任务已达上限，请等待完成后再提交。")
            job_id = uuid.uuid4().hex
            directory = self.root / job_id
            config = FrameworkConfig(
                model=ModelConfig(api_key, request.base_url.rstrip("/"), request.model, timeout_seconds=120),
                dataset=DatasetConfig(task, repo_root() / DEFAULT_DATASET_PATHS[task]),
                run=RunConfig(
                    task=task,
                    workers=request.workers,
                    output_dir=directory / "runs",
                    max_output_tokens=request.max_output_tokens,
                    thinking_enabled=request.thinking_enabled,
                    limit=request.limit,
                    execution=ExecutionConfig(image=NUMPY_IMAGE if task in PLUS_TASKS else "python:3.12-slim"),
                ),
                model_config_path=directory / "web-request",
            )
            record = {
                "id": job_id,
                "status": "QUEUED",
                "created_at": time.time(),
                "mode": request.mode,
                "task": task,
                "model": request.model,
                "workers": request.workers,
                "max_output_tokens": request.max_output_tokens,
                "thinking_enabled": request.thinking_enabled,
                "error": None,
            }
            self.records[job_id] = record
            write_json(directory / "job.json", record)
            runner = EvaluationRunner(config)
            self.runners[job_id] = runner
            self.futures[job_id] = self.pool.submit(self._run, job_id, runner)
            return dict(record)

    def _run(self, job_id: str, runner: EvaluationRunner) -> None:
        with self.lock:
            self._update(job_id, status="RUNNING", started_at=time.time())
        try:
            summary, _ = runner.run()
            with self.lock:
                self._update(
                    job_id,
                    status=summary.run_status,
                    summary=asdict(summary),
                    error=summary.error,
                    finished_at=time.time(),
                )
        except Exception as exc:
            message = str(exc).replace(runner.config.model.api_key, "[REDACTED]")
            with self.lock:
                self._update(job_id, status="FAILED", error=message, finished_at=time.time())
        finally:
            with self.lock:
                self.runners.pop(job_id, None)
                self.futures.pop(job_id, None)

    def _update(self, job_id: str, **fields: Any) -> None:
        self.records[job_id].update(fields)
        write_json(self.root / job_id / "job.json", self.records[job_id])

    def get(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            if job_id not in self.records:
                raise KeyError(job_id)
            record = dict(self.records[job_id])
        record["cases"] = []
        for path in (self.root / job_id / "runs").glob("*/*/cases/*.json"):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                record["cases"].append(row)
            except (OSError, ValueError):
                continue
        record["cases"].sort(key=lambda row: row.get("case_index", 0))
        for path in (self.root / job_id / "runs").glob("*/*/status.json"):
            with suppress(OSError, ValueError):
                record["progress"] = json.loads(path.read_text(encoding="utf-8"))
        return record

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            return [
                dict(row) for row in sorted(self.records.values(), key=lambda row: row["created_at"], reverse=True)[:50]
            ]

    def cancel(self, job_id: str) -> None:
        with self.lock:
            if job_id not in self.records:
                raise KeyError(job_id)
            future = self.futures.get(job_id)
            if future and future.cancel():
                self._update(job_id, status="INTERRUPTED", error="任务在开始前被取消。", finished_at=time.time())
                self.futures.pop(job_id, None)
                self.runners.pop(job_id, None)
            elif runner := self.runners.get(job_id):
                runner.cancel_event.set()
                self._update(job_id, status="CANCELLING")

    def run_directory(self, job_id: str) -> Path | None:
        self.get(job_id)
        return next((self.root / job_id / "runs").glob("*/*/manifest.json"), None)

    def close(self) -> None:
        with self.lock:
            for job_id in list(self.runners):
                self.cancel(job_id)
        self.pool.shutdown(wait=False, cancel_futures=True)
