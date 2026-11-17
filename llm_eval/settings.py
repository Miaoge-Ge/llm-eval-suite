from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .utils import parse_bool, repo_root, resolve_env_placeholders, slugify

DEFAULT_DATASET_PATHS = {
    "humaneval": Path("datasets/HumanEval.jsonl"),
    "humanevalplus": Path("datasets/HumanEvalPlus.jsonl"),
    "mbpp": Path("datasets/MBPP.jsonl"),
    "mbppplus": Path("datasets/MBPPPlus.jsonl"),
    "gsm": Path("datasets/GSM8K.jsonl"),
    "aime2025": Path("datasets/AIME2025.jsonl"),
    "aime2026": Path("datasets/AIME2026.jsonl"),
    "gpqa": Path("datasets/GPQA.jsonl"),
    "mmlu_pro": Path("datasets/MMLUPro.jsonl"),
    "ifeval": Path("datasets/IFEval.jsonl"),
    "livecodebench": Path("datasets/LiveCodeBench.jsonl"),
    "quick40": Path("llm_eval/suites/quick40-v1.json"),
    "api_experience": Path("llm_eval/suites/api-experience-v1.json"),
}

NUMPY_VERSION = "2.2.6"
NUMPY_IMAGE = "llm-eval-python:3.12-numpy2.2.6"
PLUS_TASKS = ("humanevalplus", "mbppplus", "quick40")


@dataclass(frozen=True)
class ModelConfig:
    api_key: str
    base_url: str
    model_name: str
    timeout_seconds: int = 120


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: Path


@dataclass(frozen=True)
class ExecutionConfig:
    backend: str = "docker"
    image: str = "python:3.12-slim"
    memory_mb: int = 256
    cpus: float = 1.0
    pids_limit: int = 64
    max_output_bytes: int = 1_048_576
    tmpfs_mb: int = 32

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or self.backend not in {"docker", "local"}:
            raise ValueError("execution.backend must be docker or local")
        if (
            not isinstance(self.image, str)
            or not self.image
            or self.image.startswith("-")
            or any(c.isspace() for c in self.image)
        ):
            raise ValueError("execution.image must be a nonempty Docker image reference")
        for name in ("memory_mb", "pids_limit", "max_output_bytes", "tmpfs_mb"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"execution.{name} must be a positive integer")
        if self.memory_mb < 16:
            raise ValueError("execution.memory_mb must be at least 16")
        if (
            isinstance(self.cpus, bool)
            or not isinstance(self.cpus, (int, float))
            or not math.isfinite(self.cpus)
            or self.cpus <= 0
        ):
            raise ValueError("execution.cpus must be finite and positive")


@dataclass(frozen=True)
class RunConfig:
    task: str
    workers: int = 10
    output_dir: Path = Path("results")
    execution_timeout_seconds: int = 20
    thinking_enabled: bool = False
    reasoning_effort: str | None = None
    limit: int | None = None
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    max_output_tokens: int | None = None
    output_token_parameter: str = "max_tokens"
    temperature: float | None = None
    seed: int | None = None
    stall_threshold_seconds: float = 1.0
    slo_ttft_seconds: float | None = None
    slo_tps: float | None = None
    slo_max_gap_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.output_token_parameter not in ("max_tokens", "max_completion_tokens"):
            raise ValueError("output_token_parameter must be max_tokens or max_completion_tokens")
        if self.stall_threshold_seconds is None:
            raise ValueError("stall_threshold_seconds must be finite and positive")
        if self.max_output_tokens is not None and (
            type(self.max_output_tokens) is not int or self.max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if self.temperature is not None and (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or not 0 <= self.temperature <= 2
        ):
            raise ValueError("temperature must be finite and between 0 and 2")
        for name in ("stall_threshold_seconds", "slo_ttft_seconds", "slo_tps", "slo_max_gap_seconds"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")

    @property
    def reasoning_display(self) -> str:
        return self.reasoning_effort or "default" if self.thinking_enabled else "disabled"


@dataclass(frozen=True)
class FrameworkConfig:
    model: ModelConfig
    dataset: DatasetConfig
    run: RunConfig
    model_config_path: Path

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset"]["path"] = str(self.dataset.path)
        payload["run"]["output_dir"] = str(self.run.output_dir)
        payload["model"]["api_key"] = "***"
        payload["model_config_path"] = str(self.model_config_path)
        return payload


def load_framework_config(
    model_config_path: str | Path | None = None,
    task: str | None = None,
    workers: int | None = None,
    limit: int | None = None,
) -> FrameworkConfig:
    root = repo_root()
    model_file = _resolve_file_path(model_config_path, root / "configs" / "model.yaml")
    model_data = _load_yaml(model_file)
    execution_data = model_data.get("execution", {})
    if not isinstance(execution_data, dict):
        raise ValueError("execution must be a mapping")
    task_name = _require_text(task or "humaneval", "task")
    if task_name in PLUS_TASKS:
        execution_data = {"image": NUMPY_IMAGE, **execution_data}
    try:
        execution = ExecutionConfig(**execution_data)
    except TypeError as exc:
        raise ValueError(f"Invalid execution configuration: {exc}") from exc

    dataset_path = _resolve_dataset_path(task_name)
    model = ModelConfig(
        api_key=_require_text(model_data.get("api_key"), "api_key"),
        base_url=_require_text(model_data.get("base_url"), "base_url"),
        model_name=_require_text(model_data.get("model_name"), "model_name"),
        timeout_seconds=int(model_data.get("timeout_seconds", 120)),
    )

    run = RunConfig(
        task=task_name,
        workers=1
        if task_name == "api_experience"
        else max(
            1, int(workers if workers is not None else (1 if task_name == "quick40" else model_data.get("workers", 10)))
        ),
        output_dir=(root / "results" / slugify(model.model_name)).resolve(),
        execution_timeout_seconds=max(1, int(model_data.get("execution_timeout_seconds", 20))),
        thinking_enabled=parse_bool(model_data.get("thinking_enabled", False), default=False),
        reasoning_effort=_optional_text(model_data.get("reasoning_effort")),
        limit=max(1, int(limit)) if limit is not None else None,
        execution=execution,
        max_output_tokens=model_data.get("max_output_tokens", 4096 if task_name == "quick40" else None),
        output_token_parameter=model_data.get("output_token_parameter", "max_tokens"),
        temperature=model_data.get("temperature"),
        seed=model_data.get("seed"),
        stall_threshold_seconds=model_data.get("stall_threshold_seconds", 1.0),
        slo_ttft_seconds=model_data.get("slo_ttft_seconds"),
        slo_tps=model_data.get("slo_tps"),
        slo_max_gap_seconds=model_data.get("slo_max_gap_seconds"),
    )

    return FrameworkConfig(
        model=model,
        dataset=DatasetConfig(name=task_name, path=dataset_path),
        run=run,
        model_config_path=model_file,
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle) or {}
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected top-level mapping in {path}")
    return resolve_env_placeholders(parsed)


def _resolve_file_path(path_value: str | Path | None, fallback: Path) -> Path:
    path = Path(path_value) if path_value else fallback
    return path if path.is_absolute() else repo_root() / path


def _resolve_dataset_path(task_name: str) -> Path:
    relative_path = DEFAULT_DATASET_PATHS.get(task_name)
    if not relative_path:
        available = ", ".join(sorted(DEFAULT_DATASET_PATHS))
        raise ValueError(f"Unsupported task '{task_name}'. Available tasks: {available}")
    return (repo_root() / relative_path).resolve()


def _require_text(value: Any, field_name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"Missing required configuration field: {field_name}")
    if text.startswith("${") and text.endswith("}"):
        raise ValueError(f"Unresolved environment variable for configuration field: {field_name}")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def config_from_snapshot(payload: dict[str, Any], root: Path) -> FrameworkConfig:
    """Reconstruct an offline configuration without loading secrets or environment placeholders."""
    model = dict(payload["model"])
    model["api_key"] = "offline-no-api-key"
    run = dict(payload["run"])
    run["output_dir"] = root / "regrades"
    run["execution"] = ExecutionConfig(**run.get("execution", {}))
    return FrameworkConfig(
        model=ModelConfig(**model),
        dataset=DatasetConfig(name=payload["dataset"]["name"], path=root / "dataset.jsonl"),
        run=RunConfig(**run),
        model_config_path=root / "config.json",
    )
