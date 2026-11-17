import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from llm_eval.execution import ExecutionEngine
from llm_eval.settings import ExecutionConfig


def local(**limits):
    return ExecutionEngine(ExecutionConfig(backend="local", **limits))


def test_local_worker_has_clean_environment_and_temporary_cwd(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-do-not-inherit")
    monkeypatch.setenv("CUSTOM_PRIVATE_TOKEN", "another-test-secret")
    monkeypatch.setenv("PYTHONPATH", "test-invalid-path")
    code = (
        "import os\n"
        "assert 'OPENAI_API_KEY' not in os.environ\n"
        "assert 'CUSTOM_PRIVATE_TOKEN' not in os.environ\n"
        "assert 'PYTHONPATH' not in os.environ\n"
        "print(os.getcwd())\n"
    )
    result = local().run(code, 5)
    assert result.status == "PASSED", result.detail
    directory = result.stdout.strip()
    assert "llm-eval-" in directory
    assert not os.path.exists(directory)


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_output_flood_is_bounded_and_stopped(stream):
    result = local(max_output_bytes=4096).run(
        f"import sys\nwhile True: sys.{stream}.write('x' * 4096); sys.{stream}.flush()",
        5,
    )
    assert result.status == "OUTPUT_LIMIT"
    assert len(result.stdout.encode()) <= 4096
    assert len(result.detail) <= 4096


def test_memory_limit_is_enforced():
    result = local(memory_mb=64).run("data = bytearray(256 * 1024 * 1024)", 5)
    assert result.status == "RESOURCE_LIMIT", result


@pytest.mark.parametrize(
    "read", ["input()", "sys.stdin.buffer.read().decode()", "os.read(0, 1024).decode()", "open(0).read()"]
)
def test_stdin_is_available_through_python_and_raw_file_descriptor(read):
    result = local().run(f"import sys, os\nprint(({read}).strip())", 5, "42\n", assertions=False)
    assert result.status == "PASSED", result
    assert result.stdout.strip() == "42"


@pytest.mark.parametrize("mode", ["normal", "timeout", "cancel"])
def test_cleanup_terminates_descendant_processes(tmp_path, mode):
    child_marker = tmp_path / "child-survived"
    started = tmp_path / "started"
    child_code = f"import time; from pathlib import Path; time.sleep(1.5); Path({str(child_marker)!r}).touch()"
    code = (
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        f"Path({str(started)!r}).touch()\n" + ("time.sleep(10)\n" if mode != "normal" else "")
    )
    engine = local()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(engine.run, code, 1 if mode == "timeout" else 5)
        if mode == "cancel":
            deadline = time.monotonic() + 3
            while not started.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert started.exists()
            engine.cancel_event.set()
        result = future.result(timeout=7)
    assert result.status == {"normal": "PASSED", "timeout": "TIMEOUT", "cancel": "CANCELLED"}[mode], result
    time.sleep(1.6)
    assert not child_marker.exists(), "A descendant escaped cleanup"


def test_cancel_before_execution_does_not_start_process(monkeypatch):
    engine = local()
    engine.cancel_event.set()
    monkeypatch.setattr("llm_eval.execution.subprocess.Popen", lambda *a, **kw: pytest.fail("Unexpected process"))
    assert engine.run("assert False", 5).status == "CANCELLED"


@pytest.mark.skipif(os.name != "nt", reason="Aggregate local PID limit uses Windows job objects")
def test_windows_job_process_limit():
    code = (
        "import subprocess, sys\n"
        "try:\n"
        "    subprocess.Popen([sys.executable, '-c', 'pass'])\n"
        "except OSError:\n"
        "    print('blocked')\n"
        "else:\n"
        "    raise AssertionError('child should be blocked')\n"
    )
    # The Windows venv redirector and its Python interpreter occupy two slots.
    result = local(pids_limit=2).run(code, 5)
    assert result.status == "PASSED", result
    assert result.stdout.strip() == "blocked"
