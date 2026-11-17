from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_eval.cli import main
from llm_eval.reporting import RunSummary


def test_list_tasks_prints_registry(capsys):
    exit_code = main(["run", "--list-tasks"])
    out = capsys.readouterr().out
    assert exit_code == 0
    for name in ("humaneval", "gsm", "mbpp", "gpqa"):
        assert name in out


def test_unknown_task_returns_error(tmp_path, capsys):
    config = tmp_path / "model.yaml"
    config.write_text(
        "\n".join(
            [
                "api_key: dummy",
                "base_url: https://example.invalid/v1",
                "model_name: demo-model",
            ]
        ),
        encoding="utf-8",
    )
    exit_code = main(["run", "--config", str(config), "--task", "does-not-exist"])
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "does-not-exist" in err


@pytest.mark.parametrize(
    ("status", "exit_code"), [("COMPLETED", 0), ("FAILED", 2), ("ABORTED", 2), ("INTERRUPTED", 130)]
)
def test_cli_exit_code_reflects_run_status(fake_config, monkeypatch, status, exit_code):
    summary = RunSummary("gsm", "demo", 1, 1, 0.0, 0.1, "00:00:00", 0.1, 0, {}, {}, run_status=status)
    writer = SimpleNamespace(paths=SimpleNamespace(report_md=Path("report.md")))
    monkeypatch.setattr("llm_eval.cli.load_framework_config", lambda **kwargs: fake_config)
    monkeypatch.setattr(
        "llm_eval.cli.EvaluationRunner", lambda config, **kwargs: SimpleNamespace(run=lambda: (summary, writer))
    )
    assert main([]) == exit_code
