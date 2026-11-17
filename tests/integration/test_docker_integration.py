"""Real Linux-container tests. Opt in with LLM_EVAL_DOCKER_TESTS=1 after pulling the image."""

import os
import subprocess
from dataclasses import replace

import pytest

from llm_eval.core.execution import ExecutionEngine
from llm_eval.core.settings import ExecutionConfig

pytestmark = pytest.mark.skipif(
    os.getenv("LLM_EVAL_DOCKER_TESTS") != "1",
    reason="Set LLM_EVAL_DOCKER_TESTS=1 to run real Docker Linux-container tests",
)


@pytest.fixture
def engine(monkeypatch):
    executor = ExecutionEngine(ExecutionConfig(memory_mb=64, pids_limit=8, max_output_bytes=4096))
    executor.check_ready()  # Explicitly requested integration runs must fail, not skip, if Docker is broken.
    names = []
    command = executor._docker_command

    def capture(name, worker):
        names.append(name)
        return command(name, worker)

    monkeypatch.setattr(executor, "_docker_command", capture)
    yield executor
    for name in names:
        result = subprocess.run(["docker", "inspect", name], capture_output=True, text=True, timeout=15)
        assert result.returncode != 0, f"Execution container was not removed: {name}"


def test_docker_isolates_environment_network_and_host_files(engine, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    secret = tmp_path / "host-private.txt"
    secret.write_text("private", encoding="utf-8")
    code = (
        "import os, socket\nfrom pathlib import Path\n"
        "assert 'OPENAI_API_KEY' not in os.environ\n"
        f"assert not Path({str(secret)!r}).exists()\n"
        "assert os.getuid() != 0\n"
        "sock = socket.socket(); sock.settimeout(1)\n"
        "assert sock.connect_ex(('1.1.1.1', 443)) != 0\n"
        "Path('/work/writable').write_text('ok')\n"
        "try:\n    Path('/root-write').write_text('no')\n"
        "except OSError:\n    pass\nelse:\n    raise AssertionError('root writable')\n"
    )
    result = engine.run(code, 10)
    assert result.status == "PASSED", result


@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("assert True", "PASSED"),
        ("import os; os._exit(0)", "FAILED"),
        ("while True: pass", "TIMEOUT"),
        ("data = bytearray(256 * 1024 * 1024)", "RESOURCE_LIMIT"),
        ("while True: print('x' * 4096)", "OUTPUT_LIMIT"),
    ],
)
def test_docker_limits_and_completion(engine, code, status):
    # Docker Desktop startup can take several seconds. Only the intentional
    # infinite loop tests a short deadline; other cases isolate their own limit.
    result = engine.run(code, 2 if status == "TIMEOUT" else 10)
    assert result.status == status, result


def test_docker_pid_limit(engine):
    engine.config = replace(engine.config, memory_mb=256)
    code = (
        "import subprocess, sys\nchildren = []\n"
        "try:\n"
        "    for _ in range(16):\n"
        "        children.append(subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)']))\n"
        "except OSError:\n    pass\n"
        "else:\n    raise AssertionError('PID limit was not enforced')\n"
    )
    result = engine.run(code, 5)
    assert result.status == "PASSED", result


def test_plus_real_numpy_runs_under_container_limits():
    from llm_eval.core.settings import NUMPY_IMAGE

    executor = ExecutionEngine(ExecutionConfig(image=NUMPY_IMAGE))
    executor.check_numpy()
    result = executor.run(
        "import numpy as np\n"
        "assert np.allclose(np.array([1, 2]), [[1, 2], [1, 2]])\n"
        "assert np.allclose([np.inf, -np.inf], [np.inf, -np.inf])\n"
        "assert np.isclose([1, 2], [1, 3]).tolist() == [True, False]\n"
        "assert np.float32(16777217) == 16777216\n",
        10,
    )
    assert result.status == "PASSED", result


def test_plus_plain_image_missing_numpy_fails_as_infrastructure():
    from llm_eval.core.execution import ExecutionUnavailable

    with pytest.raises(ExecutionUnavailable, match="require real NumPy"):
        ExecutionEngine(ExecutionConfig()).check_numpy()


@pytest.mark.parametrize(("code", "status"), [("print(6)", "FAILED"), ("print(int(input()) * 2)", "PASSED")])
def test_livecodebench_private_tests_in_real_containers(fake_config, code, status):
    from llm_eval.benchmark.livecodebench import write_private_tests
    from llm_eval.core.clients import GenerationResult
    from llm_eval.tasks import LiveCodeBenchTask, TaskCase

    descriptor = write_private_tests(fake_config.dataset.path.parent, [{"input": "10\n", "output": "20\n"}])
    config = replace(fake_config, run=replace(fake_config.run, execution=ExecutionConfig()))
    task = LiveCodeBenchTask(config)
    case = TaskCase(
        "lcb", {"problem": "Double", "tests": [{"input": "3\n", "output": "6\n"}], "private_tests": descriptor}
    )
    result = task.grade_case(case, GenerationResult(code, {}))
    assert result.status == status, result.error
    assert result.metadata["tests_executed"] == 2
