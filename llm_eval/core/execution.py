"""Bounded code execution. Docker is the default; local is for trusted code only."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llm_eval.core.settings import NUMPY_VERSION, ExecutionConfig
from llm_eval.core.status import Status


class ExecutionUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    stdout: str = ""
    detail: str = ""


def _local_environment(directory: str) -> dict[str, str]:
    # Never inherit API keys, proxy credentials, PYTHONPATH or user site config.
    env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR") if key in os.environ}
    env.update(
        {
            "PATH": os.defpath,
            "HOME": directory,
            "USERPROFILE": directory,
            "TMP": directory,
            "TEMP": directory,
            "TMPDIR": directory,
            "LANG": "C.UTF-8",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    return env


def _docker_control(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["docker", *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ExecutionUnavailable(f"Docker control command failed: {exc}") from exc


class ExecutionEngine:
    def __init__(self, config: ExecutionConfig | None = None, cancel_event: threading.Event | None = None) -> None:
        self.config = config or ExecutionConfig()
        self.cancel_event = cancel_event or threading.Event()
        self._ready = False
        self._ready_lock = threading.Lock()
        self._numpy_ready = False
        self._numpy_lock = threading.Lock()

    def check_numpy(self) -> None:
        """Probe the actual isolated runtime, before spending model requests."""
        with self._numpy_lock:
            if self._numpy_ready:
                return
            result = self.run(
                f"import numpy as np\nassert np.__version__ == {NUMPY_VERSION!r}, np.__version__\n"
                "assert np.array([1, 2]).sum() == 3\n",
                15,
            )
            if result.status != Status.PASSED:
                raise ExecutionUnavailable(
                    f"Plus tasks require real NumPy {NUMPY_VERSION} in the execution runtime; "
                    f"dependency probe returned {result.status}: {result.detail}"
                )
            self._numpy_ready = True

    def check_ready(self) -> None:
        with self._ready_lock:
            if self._ready:
                return
            if (
                self.config.backend == "local"
                and os.name == "nt"
                and self.config.pids_limit < 2
                and sys.executable != getattr(sys, "_base_executable", sys.executable)
            ):
                raise ExecutionUnavailable("Local Windows venv execution needs at least two process slots")
            if self.config.backend == "docker":
                if not shutil.which("docker"):
                    raise ExecutionUnavailable(
                        "Docker is required for code tasks; install/start Docker and prepare its image"
                    )
                info = _docker_control(["info", "--format", "{{json .}}"])
                if info.returncode:
                    raise ExecutionUnavailable(f"Docker is unavailable: {info.stderr.strip()}")
                try:
                    capabilities = json.loads(info.stdout)
                except ValueError as exc:
                    raise ExecutionUnavailable("Invalid Docker daemon response") from exc
                if capabilities.get("OSType") != "linux":
                    raise ExecutionUnavailable("Code execution requires Docker Linux containers")
                for feature in ("MemoryLimit", "SwapLimit", "PidsLimit", "CpuCfsQuota"):
                    if not capabilities.get(feature):
                        raise ExecutionUnavailable(f"Docker daemon cannot enforce {feature}")
                image = _docker_control(["image", "inspect", self.config.image])
                if image.returncode:
                    raise ExecutionUnavailable(f"Docker image unavailable locally: {self.config.image}; pull it first")
            self._ready = True

    def _docker_command(self, name: str, worker: str) -> list[str]:
        cfg = self.config
        return [
            "docker",
            "create",
            "--name",
            name,
            "--pull",
            "never",
            "--interactive",
            "--network",
            "none",
            "--read-only",
            "--user",
            "65534:65534",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--ipc",
            "none",
            "--pids-limit",
            str(cfg.pids_limit),
            "--memory",
            f"{cfg.memory_mb}m",
            "--memory-swap",
            f"{cfg.memory_mb}m",
            "--cpus",
            str(cfg.cpus),
            "--ulimit",
            "nofile=64:64",
            "--ulimit",
            "core=0:0",
            "--log-driver",
            "none",
            "--tmpfs",
            f"/work:rw,nosuid,nodev,noexec,size={cfg.tmpfs_mb}m,mode=1777",
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,noexec,size={cfg.tmpfs_mb}m,mode=1777",
            "--workdir",
            "/work",
            "--env",
            "HOME=/work",
            "--env",
            "TMPDIR=/tmp",
            "--env",
            "OPENBLAS_NUM_THREADS=1",
            "--env",
            "OMP_NUM_THREADS=1",
            "--env",
            "MKL_NUM_THREADS=1",
            "--entrypoint",
            "python",
            self.config.image,
            "-I",
            "-u",
            "-X",
            "utf8",
            "-c",
            worker,
        ]

    def run(self, code: str, timeout: int, stdin: str = "", *, assertions: bool = True) -> ExecutionResult:
        if self.cancel_event.is_set():
            return ExecutionResult(Status.CANCELLED, detail="Execution cancelled")
        try:
            self.check_ready()
        except ExecutionUnavailable as exc:
            return ExecutionResult(Status.EXECUTION_ERROR, detail=str(exc))
        receipt = f"\nLLM_EVAL_COMPLETED_{uuid.uuid4().hex}\n"
        envelope = (
            json.dumps(
                {
                    "code": code,
                    "stdin": stdin,
                    "assertions": assertions,
                    "receipt": receipt,
                    "timeout": timeout,
                    "limits": asdict(self.config),
                }
            ).encode("utf-8")
            + b"\n"
        )
        worker = Path(__file__).with_name("_worker.py").read_text(encoding="utf-8")
        name = f"llm-eval-{uuid.uuid4().hex}" if self.config.backend == "docker" else None
        with tempfile.TemporaryDirectory(prefix="llm-eval-") as directory:
            command = (
                self._docker_command(name, worker)
                if name
                else [
                    sys.executable,
                    "-I",
                    "-u",
                    "-X",
                    "utf8",
                    "-c",
                    worker,
                ]
            )
            cleanup_error = ""
            oom = False
            try:
                if name:
                    created = _docker_control(command[1:])
                    if created.returncode:
                        raise ExecutionUnavailable(f"Container creation failed: {created.stderr.strip()}")
                    # Create first so the cleanup target exists before launching
                    # the cancellable CLI attachment.
                    command = ["docker", "start", "--attach", "--interactive", name]
                status, stdout, stderr, returncode = self._monitor(
                    command,
                    envelope,
                    directory,
                    timeout,
                    len(receipt.encode()),
                    local=name is None,
                )
                if name and status is None:
                    inspection = _docker_control(["inspect", "--format", "{{json .State}}", name])
                    if inspection.returncode:
                        raise ExecutionUnavailable(f"Cannot inspect execution container: {inspection.stderr.strip()}")
                    state = json.loads(inspection.stdout)
                    if state.get("Error"):
                        raise ExecutionUnavailable(f"Container startup failed: {state['Error']}")
                    if state.get("Running"):
                        raise ExecutionUnavailable("Docker attachment ended while the container was still running")
                    returncode = int(state.get("ExitCode", returncode))
                    oom = bool(state.get("OOMKilled"))
            except (OSError, ValueError, ExecutionUnavailable) as exc:
                status, stdout, stderr, returncode = Status.EXECUTION_ERROR, b"", str(exc).encode(), -1
            finally:
                if name:
                    # Killing the Docker CLI alone leaves containers alive.
                    try:
                        removed = _docker_control(["rm", "--force", name])
                        if removed.returncode and "No such container" not in removed.stderr:
                            cleanup_error = f"Container cleanup failed for {name}: {removed.stderr.strip()}"
                    except ExecutionUnavailable as exc:
                        cleanup_error = f"Container cleanup failed for {name}: {exc}"
            if cleanup_error:
                return ExecutionResult(Status.EXECUTION_ERROR, detail=cleanup_error)
            receipt_bytes = receipt.encode()
            completed = stderr.endswith(receipt_bytes)
            if completed:
                stderr = stderr[: -len(receipt_bytes)]
            if len(stdout) + len(stderr) > self.config.max_output_bytes:
                status = status or Status.OUTPUT_LIMIT
                stdout = stdout[: self.config.max_output_bytes]
                stderr = stderr[: self.config.max_output_bytes - len(stdout)]
            output = stdout.decode("utf-8", errors="replace")
            detail = stderr.decode("utf-8", errors="replace").strip()
            if status:
                return ExecutionResult(status, output, detail or f"Execution stopped: {status}")
            if oom or returncode in (-9, -24, 137, 152) or "MemoryError" in detail:
                return ExecutionResult(Status.RESOURCE_LIMIT, output, detail or "Execution resource limit exceeded")
            if name and returncode in (125, 126, 127):
                return ExecutionResult(Status.EXECUTION_ERROR, output, detail or "Container could not execute Python")
            if returncode != 0:
                return ExecutionResult(Status.FAILED, output, detail or f"Process exited with code {returncode}")
            if assertions and not completed:
                return ExecutionResult(Status.FAILED, output, "Test program exited before completion")
            return ExecutionResult(Status.PASSED, output)

    def _monitor(
        self,
        command: list[str],
        payload: bytes,
        directory: str,
        timeout: int,
        receipt_size: int,
        *,
        local: bool,
    ) -> tuple[str | None, bytes, bytes, int]:
        process: subprocess.Popen[bytes] | None = None
        job: Any = None
        lock = threading.Lock()
        overflow = threading.Event()
        buffers = [bytearray(), bytearray()]
        cap = self.config.max_output_bytes + receipt_size
        threads: list[threading.Thread] = []

        def drain(pipe: Any, target: bytearray) -> None:
            try:
                while chunk := pipe.read1(4096):
                    with lock:
                        remaining = cap - sum(map(len, buffers))
                        target.extend(chunk[: max(0, remaining)])
                        if len(chunk) > remaining:
                            overflow.set()
            except (OSError, ValueError):
                pass

        def feed(pipe: Any) -> None:
            try:
                pipe.write(payload)
                pipe.flush()
            except (OSError, ValueError):
                pass
            finally:
                pipe.close()

        status = None
        creation_flags = 0
        if sys.platform == "win32":
            creation_flags = subprocess.CREATE_NO_WINDOW
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=directory,
                env=_local_environment(directory) if local else None,
                start_new_session=os.name != "nt",
                creationflags=creation_flags,
            )
            if os.name == "nt":
                from llm_eval.core._windows_job import WindowsJob

                job = WindowsJob(process.pid, self.config.memory_mb if local else None, self.config.pids_limit)
            # The worker waits for stdin: limits are attached before code runs.
            threads = [
                threading.Thread(target=drain, args=(process.stdout, buffers[0]), daemon=True),
                threading.Thread(target=drain, args=(process.stderr, buffers[1]), daemon=True),
                threading.Thread(target=feed, args=(process.stdin,), daemon=True),
            ]
            for thread in threads:
                thread.start()
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                if self.cancel_event.is_set():
                    status = Status.CANCELLED
                    break
                if overflow.is_set():
                    status = Status.OUTPUT_LIMIT
                    break
                if time.monotonic() >= deadline:
                    status = Status.TIMEOUT
                    break
                self.cancel_event.wait(0.01)
        finally:
            # Clean descendants even when the parent exits successfully.
            if job is not None:
                job.close()
            if process is not None:
                if sys.platform != "win32":
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                if process.poll() is None:
                    process.kill()
                process.wait()
                for thread in threads:
                    thread.join(timeout=3)
                for pipe in (process.stdin, process.stdout, process.stderr):
                    if pipe:
                        pipe.close()
        if overflow.is_set() and status is None:
            status = Status.OUTPUT_LIMIT
        assert process is not None
        return status, bytes(buffers[0]), bytes(buffers[1]), process.returncode


def execute_python(code: str, timeout_seconds: int, config: ExecutionConfig | None = None) -> tuple[str, str]:
    result = ExecutionEngine(config).run(code, timeout_seconds)
    return result.status, result.stdout.strip() if result.status == Status.PASSED else result.detail


def run_python_with_stdin(
    code: str,
    stdin_text: str,
    timeout_seconds: int,
    config: ExecutionConfig | None = None,
) -> tuple[str, str, str]:
    result = ExecutionEngine(config).run(code, timeout_seconds, stdin_text, assertions=False)
    return ("OK" if result.status == Status.PASSED else result.status), result.stdout, result.detail


def normalize_stdout(text: str) -> str:
    """Strip trailing whitespace per line and trailing blank lines."""
    lines = [line.rstrip() for line in text.strip().split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)
