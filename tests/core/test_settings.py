import pytest

from llm_eval.core.settings import ExecutionConfig, load_framework_config
from llm_eval.core.utils import repo_root


def test_load_framework_config_resolves_env(tmp_path, monkeypatch):
    model_config = tmp_path / "model.yaml"

    monkeypatch.setenv("TEST_API_KEY", "secret-token")
    model_config.write_text(
        "\n".join(
            [
                "api_key: ${TEST_API_KEY}",
                "base_url: https://example.invalid/v1",
                "model_name: demo-model",
                "workers: 10",
                "thinking_enabled: true",
                "reasoning_effort: high",
            ]
        ),
        encoding="utf-8",
    )
    config = load_framework_config(model_config, task="humaneval")

    assert config.model.api_key == "secret-token"
    assert config.model.model_name == "demo-model"
    assert config.run.workers == 10
    assert config.run.thinking_enabled is True
    assert config.run.reasoning_effort == "high"
    assert config.dataset.path == (repo_root() / "datasets" / "benchmark" / "HumanEval.jsonl").resolve()
    assert config.run.output_dir == (repo_root() / "results" / "demo-model").resolve()


@pytest.mark.parametrize(
    "overrides",
    [
        {"backend": "unknown"},
        {"memory_mb": 0},
        {"memory_mb": 8},
        {"memory_mb": True},
        {"pids_limit": -1},
        {"max_output_bytes": 0},
        {"cpus": 0},
        {"cpus": float("nan")},
        {"cpus": float("inf")},
        {"tmpfs_mb": -1},
        {"image": "--privileged"},
        {"image": ""},
        {"image": 123},
        {"cpus": "fast"},
        {"backend": []},
    ],
)
def test_execution_limits_reject_invalid_configuration(overrides):
    with pytest.raises(ValueError):
        ExecutionConfig(**overrides)


def test_execution_defaults_are_isolated():
    assert ExecutionConfig().backend == "docker"


def test_execution_config_is_loaded_and_saved_in_snapshot(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text(
        "api_key: dummy\nbase_url: https://example.invalid/v1\nmodel_name: demo\n"
        "execution:\n  backend: local\n  memory_mb: 128\n  max_output_bytes: 4096\n",
        encoding="utf-8",
    )
    config = load_framework_config(path, task="gsm")
    assert config.run.execution.backend == "local"
    assert config.run.execution.memory_mb == 128
    assert config.to_public_dict()["run"]["execution"]["max_output_bytes"] == 4096


def test_output_dir_slugifies_model_name(tmp_path):
    model_config = tmp_path / "model.yaml"
    model_config.write_text(
        "\n".join(
            [
                "api_key: dummy",
                "base_url: https://example.invalid/v1",
                'model_name: "org/model:beta"',
            ]
        ),
        encoding="utf-8",
    )
    config = load_framework_config(model_config, task="gsm")

    assert config.model.model_name == "org/model:beta"
    assert config.run.output_dir == (repo_root() / "results" / "org-model-beta").resolve()


def test_load_framework_config_parses_disabled_thinking(tmp_path, monkeypatch):
    model_config = tmp_path / "model.yaml"

    monkeypatch.setenv("TEST_API_KEY", "secret-token")
    model_config.write_text(
        "\n".join(
            [
                "api_key: ${TEST_API_KEY}",
                "base_url: https://example.invalid/v1",
                "model_name: demo-model",
                "thinking_enabled: disabled",
                "reasoning_effort: high",
            ]
        ),
        encoding="utf-8",
    )
    config = load_framework_config(model_config, task="humaneval")

    assert config.run.thinking_enabled is False
    assert config.run.reasoning_effort == "high"
