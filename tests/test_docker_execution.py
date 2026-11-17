import json
import subprocess

import pytest

from llm_eval.execution import ExecutionEngine, ExecutionUnavailable
from llm_eval.settings import ExecutionConfig


def test_missing_docker_fails_closed(monkeypatch):
    monkeypatch.setattr("llm_eval.execution.shutil.which", lambda _: None)
    engine = ExecutionEngine()
    with pytest.raises(ExecutionUnavailable, match="Docker is required"):
        engine.check_ready()
    assert engine.run("print('must not run on host')", 5).status == "EXECUTION_ERROR"


@pytest.mark.parametrize("feature", ["OSType", "MemoryLimit", "SwapLimit", "PidsLimit", "CpuCfsQuota"])
def test_docker_requires_enforceable_limits(monkeypatch, feature):
    monkeypatch.setattr("llm_eval.execution.shutil.which", lambda _: "docker")
    capabilities = {"OSType": "linux", "MemoryLimit": True, "SwapLimit": True, "PidsLimit": True, "CpuCfsQuota": True}
    capabilities[feature] = False
    monkeypatch.setattr(
        "llm_eval.execution._docker_control", lambda _: subprocess.CompletedProcess([], 0, json.dumps(capabilities), "")
    )
    with pytest.raises(ExecutionUnavailable):
        ExecutionEngine().check_ready()


def test_container_command_has_isolation_and_resource_limits():
    engine = ExecutionEngine(ExecutionConfig(memory_mb=128, cpus=0.5, pids_limit=8, tmpfs_mb=16))
    args = engine._docker_command("test-container", "worker")
    for option, value in {
        "--network": "none",
        "--user": "65534:65534",
        "--cap-drop": "ALL",
        "--security-opt": "no-new-privileges",
        "--memory": "128m",
        "--memory-swap": "128m",
        "--cpus": "0.5",
        "--pids-limit": "8",
        "--pull": "never",
    }.items():
        assert args[args.index(option) + 1] == value
    assert "--read-only" in args
    assert "--mount" not in args and "--volume" not in args
    assert not any("API_KEY" in arg for arg in args)
    assert "/work:rw,nosuid,nodev,noexec,size=16m,mode=1777" in args


@pytest.mark.parametrize("status", [None, "TIMEOUT", "CANCELLED", "OUTPUT_LIMIT"])
def test_container_removed_after_every_execution_outcome(monkeypatch, status):
    engine = ExecutionEngine()
    engine._ready = True
    calls = []

    def monitor(command, payload, *args, **kwargs):
        envelope = json.loads(payload)
        return status, b"", envelope["receipt"].encode(), 0

    def control(args):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, '{"OOMKilled": false}', "")

    monkeypatch.setattr(engine, "_monitor", monitor)
    monkeypatch.setattr("llm_eval.execution._docker_control", control)
    result = engine.run("pass", 1)
    assert result.status == (status or "PASSED")
    assert calls[-1][:2] == ["rm", "--force"]
    assert calls[-1][2].startswith("llm-eval-")


def test_container_cleanup_failure_is_infrastructure_error(monkeypatch):
    engine = ExecutionEngine()
    engine._ready = True
    monkeypatch.setattr(engine, "_monitor", lambda *a, **kw: ("TIMEOUT", b"", b"", -9))
    monkeypatch.setattr(
        "llm_eval.execution._docker_control", lambda args: subprocess.CompletedProcess(args, 1, "", "daemon offline")
    )
    result = engine.run("pass", 1)
    assert result.status == "EXECUTION_ERROR"
    assert "cleanup failed" in result.detail
