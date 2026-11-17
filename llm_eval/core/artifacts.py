"""Durable generation records, case checkpoints, run identity, and writer locking."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, BinaryIO

from llm_eval.core.clients import GenerationResult
from llm_eval.core.settings import FrameworkConfig
from llm_eval.core.status import ERROR_STATUSES, GRADED_STATUSES, Status
from llm_eval.tasks import BaseEvaluationTask, TaskCase

SCHEMA_VERSION = 1


def encode_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep sibling temporary names short: nested regrades can approach Windows'
    # path-length limit even when the destination itself is writable.
    temporary = path.with_name(f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(5):
            try:
                temporary.replace(path)
                break
            except PermissionError as exc:
                # Windows readers can briefly deny replacement of an existing file.
                # Retry only the atomic rename, preserving the previous file on failure.
                if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 4:
                    raise
                time.sleep(0.02 * 2**attempt)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    write_atomic(path, encode_json(value) + "\n")


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid saved JSON at {path}: {exc}") from exc


def _validate_generation(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("Generation must be an object")
    generation = GenerationResult(**value)
    if not isinstance(generation.content, str) or not isinstance(generation.usage, dict):
        raise ValueError("Invalid generation content or usage")
    counts = [generation.attempts, *generation.usage.values()]
    if any(type(count) is not int or count < 0 for count in counts):
        raise ValueError("Usage and attempt counts must be nonnegative integers")
    if generation.error is not None and not isinstance(generation.error, str):
        raise ValueError("Invalid generation error")
    if any(type(flag) is not bool for flag in (generation.fatal, generation.cancelled, generation.usage_complete)):
        raise ValueError("Invalid generation flags")
    if (generation.fatal or generation.cancelled) and not generation.error:
        raise ValueError("Failed generation is missing its error")


class RunLock:
    """OS-owned lock: crashes release it without stale lock-file deletion."""

    def __init__(self, root: Path) -> None:
        self.path = root / ".run.lock"
        self.handle: BinaryIO | None = None

    def __enter__(self) -> RunLock:
        handle = self.path.open("a+b")
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise ValueError(f"Run directory is already in use: {self.path.parent}") from exc
        self.handle = handle
        return self

    def __exit__(self, *args: Any) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def implementation_hash() -> str:
    root = Path(__file__).parent.parent
    sources = {str(path.relative_to(root)): digest(path.read_bytes()) for path in sorted(root.rglob("*.py"))}
    return digest(encode_json(sources).encode())


def identity(config: FrameworkConfig, task: BaseEvaluationTask, cases: list[TaskCase]) -> dict[str, Any]:
    model = asdict(config.model)
    model.pop("api_key")
    prompts = []
    for case in cases:
        try:
            prompts.append(task.build_messages(case) if not task.validate_case(case) else None)
        except (KeyError, ValueError, TypeError):
            prompts.append(None)
    lockfile = Path(__file__).resolve().parents[2] / "uv.lock"
    return {
        "python_version": sys.version,
        "dependency_lock_sha256": digest(lockfile.read_bytes()) if lockfile.exists() else None,
        "model": model,
        "task": config.run.task,
        "thinking_enabled": config.run.thinking_enabled,
        "reasoning_effort": config.run.reasoning_effort,
        "generation_options": [task.generation_options(case) for case in cases],
        "max_output_tokens": config.run.max_output_tokens,
        "output_token_parameter": config.run.output_token_parameter,
        "temperature": config.run.temperature,
        "seed": config.run.seed,
        "stall_threshold_seconds": config.run.stall_threshold_seconds,
        "experience_slo": [config.run.slo_ttft_seconds, config.run.slo_tps, config.run.slo_max_gap_seconds],
        "execution": asdict(config.run.execution),
        "execution_timeout_seconds": config.run.execution_timeout_seconds,
        "dataset_sha256": digest(config.dataset.path.read_bytes()),
        "cases_sha256": digest(encode_json([asdict(case) for case in cases]).encode()),
        "prompts_sha256": digest(encode_json(prompts).encode()),
        "implementation_sha256": implementation_hash(),
    }


class RunArchive:
    def __init__(self, root: Path, cases: list[TaskCase]) -> None:
        self.root = root
        self.cases = cases
        self._lock = threading.Lock()
        self.generations: list[dict[str, Any]] = []
        self.latest: dict[int, dict[str, Any]] = {}
        self.initial_ids: set[str] = set()

    def create(
        self, config: FrameworkConfig, run_identity: dict[str, Any], operation: str, source: str | None = None
    ) -> None:
        snapshot = "".join(encode_json(asdict(case)) + "\n" for case in self.cases)
        write_atomic(self.root / "dataset.jsonl", snapshot)
        write_json(
            self.root / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "operation": operation,
                "identity": run_identity,
                "fingerprint": digest(encode_json(run_identity).encode()),
                "snapshot_sha256": digest(snapshot.encode()),
                "config": config.to_public_dict(),
                "source_run": source,
                "created_at": time.time(),
            },
        )

    def load_generations(self) -> None:
        records = []
        seen = set()
        for path in sorted((self.root / "generations").glob("*/*.json")):
            try:
                record = read_object(path)
                index = record["case_index"]
                if type(index) is not int or not 0 <= index < len(self.cases):
                    raise ValueError("Invalid generation case index")
                if path.parent.name != f"{index:08d}" or not path.stem.isdecimal() or int(path.stem) < 1:
                    raise ValueError("Generation filename does not match its case or sequence")
                if record["case_id"] != self.cases[index].case_id:
                    raise ValueError("Generation case mismatch")
                if record["schema_version"] != SCHEMA_VERSION:
                    raise ValueError("Unsupported generation version")
                if record["payload_sha256"] != digest(encode_json(self.cases[index].payload).encode()):
                    raise ValueError("Generation payload mismatch")
                identifier = record["generation_id"]
                if not isinstance(identifier, str) or not identifier or identifier in seen:
                    raise ValueError("Invalid or duplicate generation identifier")
                _validate_generation(record["generation"])
                seen.add(identifier)
                records.append(record)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid generation record at {path}: {exc}") from exc
        self.generations = records
        self.latest = {record["case_index"]: record for record in records}
        self.initial_ids = {record["generation_id"] for record in self.generations}

    def save_generation(
        self,
        index: int,
        messages: list[dict[str, str]],
        generation: GenerationResult,
        request: dict[str, Any],
        duration: float,
    ) -> dict[str, Any]:
        case = self.cases[index]
        record = {
            "schema_version": SCHEMA_VERSION,
            "generation_id": uuid.uuid4().hex,
            "case_index": index,
            "case_id": case.case_id,
            "payload_sha256": digest(encode_json(case.payload).encode()),
            "messages": messages,
            "request_parameters": request,
            "generation": asdict(generation),
            "duration_seconds": duration,
            "recorded_at": time.time(),
        }
        with self._lock:
            folder = self.root / "generations" / f"{index:08d}"
            sequence = max((int(path.stem) for path in folder.glob("*.json")), default=0) + 1
            write_json(folder / f"{sequence:08d}.json", record)
            self.generations.append(record)
            self.latest[index] = record
        return record

    def checkpoints(self) -> dict[int, dict[str, Any]]:
        results = {}
        known = {record["generation_id"]: record for record in self.generations}
        terminal = (*GRADED_STATUSES, Status.GENERATED)
        for path in sorted((self.root / "cases").glob("*.json")):
            try:
                payload = read_object(path)
                index = payload["case_index"]
                if type(index) is not int or not 0 <= index < len(self.cases) or index in results:
                    raise ValueError("Invalid or duplicate case index")
                if payload["case_id"] != self.cases[index].case_id:
                    raise ValueError("Checkpoint case mismatch")
                if payload["status"] not in (*terminal, *ERROR_STATUSES, Status.CANCELLED):
                    raise ValueError("Invalid checkpoint status")
                duration = payload["duration_seconds"]
                if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
                    raise ValueError("Invalid checkpoint duration")
                if payload["status"] in terminal:
                    original = known.get(payload["metadata"]["generation_id"])
                    if original is None:
                        raise ValueError("Completed checkpoint is missing its original generation record")
                    if original["case_index"] != index:
                        raise ValueError("Completed checkpoint refers to another case's generation")
                    if original["generation"]["error"] or not original["generation"]["content"].strip():
                        raise ValueError("Completed checkpoint refers to an incomplete generation")
                results[index] = payload
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid result checkpoint at {path}: {exc}") from exc
        return results

    def usage(self, *, new_only: bool = False) -> tuple[dict[str, int], int]:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        attempts = 0
        for record in self.generations:
            if new_only and record["generation_id"] in self.initial_ids:
                continue
            generation = record["generation"]
            for key in usage:
                usage[key] += generation["usage"].get(key, 0)
            attempts += generation["attempts"]
        return usage, attempts


def load_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise ValueError("This run has no manifest/raw-answer archive; legacy runs cannot be resumed or regraded")
    manifest = read_object(path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported run manifest version")
    if not isinstance(manifest.get("identity"), dict) or not isinstance(manifest.get("config"), dict):
        raise ValueError("Run manifest is missing its identity or configuration")
    if manifest.get("fingerprint") != digest(encode_json(manifest["identity"]).encode()):
        raise ValueError("Run manifest fingerprint is corrupt")
    if manifest.get("operation") not in ("run", "generate", "grade") or not manifest.get("snapshot_sha256"):
        raise ValueError("Run manifest is missing its operation or dataset hash")
    return manifest


def load_snapshot(root: Path) -> tuple[dict[str, Any], list[TaskCase], RunArchive]:
    manifest = load_manifest(root)
    data = (root / "dataset.jsonl").read_bytes()
    if digest(data) != manifest["snapshot_sha256"]:
        raise ValueError("Saved dataset snapshot has changed")
    cases = [TaskCase(**json.loads(line)) for line in data.decode("utf-8").splitlines() if line.strip()]
    archive = RunArchive(root, cases)
    archive.load_generations()
    return manifest, cases, archive


def copy_generations(source: RunArchive, destination: RunArchive) -> None:
    for path in sorted((source.root / "generations").glob("*/*.json")):
        target = destination.root / path.relative_to(source.root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    destination.load_generations()
