"""Exercise the compiled Linux service through real HTTP and scorer processes."""

import hashlib
import hmac
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from llm_eval.core.utils import repo_root

pytestmark = pytest.mark.skipif(not os.getenv("LLM_EVAL_NATIVE_BINARY"), reason="Requires compiled native service")


@pytest.fixture
def service(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    state = tmp_path / "state"
    for digit in ("1", "2"):
        folder = state / "jobs" / (digit * 32)
        folder.mkdir(parents=True)
        (folder / "job.json").write_text(
            json.dumps({"id": digit * 32, "owner_id": digit * 32, "status": "COMPLETED", "created_at": time.time()})
        )
    interrupted = state / "jobs" / ("3" * 32)
    interrupted.mkdir()
    (interrupted / "job.json").write_text(
        json.dumps({"owner_id": "admin", "status": "RUNNING", "created_at": time.time()})
    )
    password = "native-test-password"
    environment = dict(
        os.environ,
        LLM_EVAL_ROOT=str(repo_root()),
        LLM_EVAL_STATE_DIR=str(state),
        LLM_EVAL_PYTHON=sys.executable,
        LLM_EVAL_PORT=str(port),
        LLM_EVAL_ADMIN_PASSWORD=password,
        LLM_EVAL_SESSION_SECRET="test-secret-" * 4,
        LLM_EVAL_EXECUTION_CAPACITY="2",
        LLM_EVAL_REQUEST_CAPACITY="4",
    )
    log = (tmp_path / "server.log").open("w+")
    process = subprocess.Popen([os.environ["LLM_EVAL_NATIVE_BINARY"]], env=environment, stdout=log, stderr=log)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                if httpx.get(base + "/api/health").status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if process.poll() is not None:
                log.seek(0)
                pytest.fail(log.read())
            time.sleep(0.1)
        else:
            pytest.fail("Native service did not start")
        with httpx.Client(base_url=base, timeout=60) as admin:
            assert admin.post("/api/login", json={"password": password}).status_code == 200
            yield base, admin, state, process
    finally:
        process.terminate()
        process.wait(timeout=40)
        log.close()


@pytest.fixture
def provider():
    class Mock(BaseHTTPRequestHandler):
        count = 0
        requests = []
        mode = "failures"
        answers = {}
        default_answer = "hello world"
        lock = threading.Lock()

        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with Mock.lock:
                Mock.count += 1
                Mock.requests.append(body)
                number = Mock.count
            if (Mock.mode == "failures" and number % 3 == 0) or (Mock.mode == "retry" and number == 1):
                self.send_response(429)
                self.end_headers()
                self.wfile.write(b'{"error":"limited"}')
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            if Mock.mode == "slow":
                time.sleep(2)
            answer = Mock.answers.get(json.dumps(body["messages"], sort_keys=True), Mock.default_answer)
            chunks = [
                {"choices": [{"index": 0, "delta": {"reasoning_content": "thinking"}}]},
                {"choices": [{"index": 0, "delta": {"content": answer[:1]}}]},
                {"choices": [{"index": 0, "delta": {"content": answer[1:]}}]},
            ]
            if Mock.mode == "reasoning_only":
                chunks = [
                    {"choices": [{"index": 0, "delta": {"reasoning_content": "thinking"}, "finish_reason": "length"}]}
                ]
            elif Mock.mode != "failures" or number % 3 != 1:
                chunks += [
                    {
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                    }
                ]
            for chunk in chunks:
                try:
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
                time.sleep(0.005)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Mock)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", Mock
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def wait_job(client, job_id, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        response.raise_for_status()
        row = response.json()
        if row["status"] not in {"QUEUED", "RUNNING", "CANCELLING"}:
            return row
        time.sleep(0.2)
    pytest.fail("Native job did not finish")


def submit(admin, url, **options):
    response = admin.post(
        "/api/jobs", json=dict(base_url=url, api_key="test-only-key", model="mock", mode="api", workers=10, **options)
    )
    assert response.status_code == 202, response.text
    return response.json()["id"]


def test_anonymous_sessions_security_and_admin_isolation(service, provider):
    base, admin, _, _ = service
    url, _ = provider
    with httpx.Client(base_url=base) as first, httpx.Client(base_url=base) as second:
        assert first.get("/api/jobs").status_code == 401
        for client in (first, second):
            response = client.post("/api/session")
            assert response.status_code == 200
            assert "HttpOnly" in response.headers["set-cookie"]
            assert "SameSite=Strict" in response.headers["set-cookie"]
            assert client.get("/api/config").json()["server_key_available"] is False
        assert first.cookies.get("llm_eval_guest") != second.cookies.get("llm_eval_guest")
        assert first.post("/api/jobs", json={}).status_code == 422
        assert first.post("/api/jobs", json={"base_url": url, "api_key": "test"}).status_code == 422
        assert first.post("/api/session", headers={"Origin": "https://evil.example"}).status_code == 403
        assert first.get("/api/admin/capacity").status_code == 404
        job_id = submit(admin, url)
        for client in (first, second):
            assert client.get("/api/jobs").json() == []
            for suffix in ("", "/report", "/artifacts"):
                assert client.get(f"/api/jobs/{job_id}{suffix}").status_code == 404
            assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 404
        forged = first.cookies.get("llm_eval_guest")[:-1] + "x"
        assert httpx.get(base + "/api/jobs", headers={"Cookie": f"llm_eval_guest={forged}"}).status_code == 401
        wait_job(admin, job_id)


def test_restored_guest_ownership_and_interrupted_restart(service):
    base, admin, _, process = service
    from llm_eval.core.settings import DEFAULT_DATASET_PATHS

    for task in admin.get("/api/config").json()["tasks"]:
        path = repo_root() / DEFAULT_DATASET_PATHS[task["id"]]
        if path.is_file():
            with path.open(encoding="utf-8") as dataset:
                assert task["total_cases"] == sum(bool(line.strip()) for line in dataset)
        else:
            assert task["total_cases"] is None and not task["available"]
    tokens = []
    for digit in ("1", "2"):
        payload = f"{int(time.time()) + 3600}.{digit * 32}.{'a' * 32}"
        signature = hmac.new(("test-secret-" * 4).encode(), payload.encode(), hashlib.sha256).hexdigest()
        tokens.append(payload + "." + signature)
    for own, other, token in (("1", "2", tokens[0]), ("2", "1", tokens[1])):
        with httpx.Client(base_url=base, headers={"Cookie": f"llm_eval_guest={token}"}) as guest:
            assert [row["id"] for row in guest.get("/api/jobs").json()] == [own * 32]
            assert guest.get(f"/api/jobs/{own * 32}").status_code == 200
            assert guest.get(f"/api/jobs/{other * 32}").status_code == 404
            assert guest.post(f"/api/jobs/{other * 32}/cancel").status_code == 404
    assert admin.get(f"/api/jobs/{'3' * 32}").json()["status"] == "INTERRUPTED"
    children = Path(f"/proc/{process.pid}/task/{process.pid}/children").read_text().split()
    assert len(children) == 2
    for child in children:
        environment = Path(f"/proc/{child}/environ").read_bytes()
        for forbidden in (b"DEEPSEEK_API_KEY=", b"LLM_EVAL_ADMIN_PASSWORD=", b"LLM_EVAL_SESSION_SECRET="):
            assert forbidden not in environment


def test_one_hundred_connections_do_not_pin_idle_http_workers(service):
    from concurrent.futures import ThreadPoolExecutor
    from http.client import HTTPConnection
    from urllib.parse import urlsplit

    base, _, _, _ = service
    target = urlsplit(base)
    barrier = threading.Barrier(100)

    def visit(_):
        connection = HTTPConnection(target.hostname, target.port, timeout=8)
        try:
            connection.connect()
            barrier.wait(timeout=8)
            connection.request("GET", "/api/health")
            response = connection.getresponse()
            assert response.status == 200
            assert json.loads(response.read())["runtime"] == "cpp"
            barrier.wait(timeout=8)
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=100) as pool:
        list(pool.map(visit, range(100)))


def test_sixty_failures_are_not_hidden_by_retries(service, provider):
    _, admin, state, _ = service
    url, mock = provider
    job_id = submit(admin, url)
    row = wait_job(admin, job_id)
    assert "summary" in row, row
    assert mock.count == row["summary"]["api_attempts"] == 60
    assert len(row["cases"]) == 60
    assert len(row["summary"]["metrics"]["scenarios"]) == 6
    usage = row["summary"]["metrics"]["overall"]["token_usage"]
    assert usage["total_tokens"]["tokens"] == row["summary"]["token_usage"]["total_tokens"]
    assert usage["total_tokens"]["missing_requests"] > 0
    assert "## Token usage" in admin.get(f"/api/jobs/{job_id}/report").text
    assert sum("response_format" in body for body in mock.requests) == 10
    records = list((state / "jobs" / job_id).glob("runs/*/*/generations/*/*.json"))
    assert len(records) == 60
    for path in records:
        generation = json.loads(path.read_text())["generation"]
        assert generation["attempts"] == 1
        assert len(generation["attempt_records"]) == 1
    response = admin.get(f"/api/jobs/{job_id}")
    assert admin.get(f"/api/jobs/{job_id}", headers={"If-None-Match": response.headers["etag"]}).status_code == 304
    assert admin.get(f"/api/jobs/{job_id}/artifacts").content[:2] == b"PK"
    capacity = admin.get("/api/admin/capacity").json()
    assert capacity["requests"]["peak"] <= 4
    assert capacity["graders"]["peak"] <= 2


def test_reasoning_only_truncation_has_no_retries_or_invented_tps(service, provider):
    _, admin, _, _ = service
    url, mock = provider
    mock.mode = "reasoning_only"
    row = wait_job(admin, submit(admin, url))
    assert row["summary"]["api_attempts"] == mock.count == 60
    for case in row["cases"]:
        metrics = case["metadata"]["experience"]
        assert metrics["truncated"] is True
        assert metrics["first_body_seconds"] is None
        assert metrics["output_tps_estimate"] is None


def test_cancel_and_per_owner_capacity(service, provider):
    _, admin, _, _ = service
    url, mock = provider
    mock.mode = "slow"
    ids = [submit(admin, url), submit(admin, url)]
    response = admin.post("/api/jobs", json={"mode": "api", "base_url": url, "api_key": "test"})
    assert response.status_code == 422
    for job_id in ids:
        assert admin.post(f"/api/jobs/{job_id}/cancel").status_code == 200
    for job_id in ids:
        assert wait_job(admin, job_id)["status"] == "INTERRUPTED"
    assert mock.count < 60


def test_benchmark_retry_keeps_both_attempts(service, provider):
    _, admin, _, _ = service
    url, mock = provider
    mock.mode = "retry"
    response = admin.post(
        "/api/jobs",
        json={"mode": "benchmark", "task": "gsm", "limit": 1, "base_url": url, "api_key": "test", "workers": 1},
    )
    assert response.status_code == 202
    row = wait_job(admin, response.json()["id"])
    assert row["status"] == "COMPLETED", row
    assert row["summary"]["api_attempts"] == mock.count == 2


def test_parallel_jobs_observe_global_request_and_grader_limits(service, provider):
    _, admin, _, _ = service
    url, mock = provider
    mock.mode = "reference"
    ids = [submit(admin, url), submit(admin, url)]
    for identifier in ids:
        assert wait_job(admin, identifier)["summary"]["api_attempts"] == 60
    capacity = admin.get("/api/admin/capacity").json()
    assert mock.count == 120
    assert capacity["requests"]["peak"] <= 4
    assert capacity["graders"]["peak"] <= 2


def test_hard_restart_preserves_in_flight_request_accounting(service, provider, tmp_path):
    base, admin, state, process = service
    url, mock = provider
    mock.mode = "slow"
    identifier = submit(admin, url)
    for _ in range(100):
        if mock.count:
            break
        time.sleep(0.02)
    assert mock.count > 0
    environment = dict(
        item.decode().split("=", 1) for item in Path(f"/proc/{process.pid}/environ").read_bytes().split(b"\0") if item
    )
    process.kill()
    process.wait(timeout=10)
    pending = list((state / "jobs" / identifier).glob("runs/*/*/native_pending/*.json"))
    assert len(pending) >= mock.count
    with (tmp_path / "restarted.log").open("w") as log:
        restarted = subprocess.Popen(process.args, env=environment, stdout=log, stderr=log)
        try:
            with httpx.Client(base_url=base, timeout=10) as client:
                for _ in range(100):
                    try:
                        response = client.post("/api/login", json={"password": "native-test-password"})
                        if response.status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.05)
                row = client.get(f"/api/jobs/{identifier}").json()
                assert row["status"] == "INTERRUPTED", row
                assert row["summary"]["api_attempts"] == len(pending)
                assert row["summary"]["incomplete_usage_cases"] == len(pending)
                assert row["summary"]["metrics"]["overall"]["attempted"] == len(pending)
        finally:
            restarted.terminate()
            restarted.wait(timeout=40)


@pytest.mark.skipif(
    os.getenv("LLM_EVAL_DOCKER_TESTS") != "1"
    or not (repo_root() / "datasets/benchmark/LiveCodeBench.private").is_dir(),
    reason="Requires Docker and private LiveCodeBench assets",
)
def test_native_livecodebench_private_tests(service, provider):
    _, admin, _, _ = service
    url, mock = provider
    mock.mode = "reference"
    mock.default_answer = (
        "import sys\nrows=sys.stdin.read().split()\n"
        "for s in rows[1:]:\n print('YES' if sum(a!=b for a,b in zip(s,'abc'))<=2 else 'NO')"
    )
    response = admin.post(
        "/api/jobs",
        json={
            "mode": "benchmark",
            "task": "livecodebench",
            "limit": 1,
            "base_url": url,
            "api_key": "test",
            "workers": 1,
        },
    )
    assert response.status_code == 202, response.text
    row = wait_job(admin, response.json()["id"])
    assert row["status"] == "COMPLETED", row
    assert row["summary"]["status_counts"] == {"PASSED": 1}
    assert row["summary"]["api_attempts"] == mock.count == 1


@pytest.mark.skipif(os.getenv("LLM_EVAL_DOCKER_TESTS") != "1", reason="Requires real Docker")
@pytest.mark.parametrize("task_name", ["humanevalplus", "mbppplus"])
def test_native_full_plus_tests_and_offline_regrade(service, provider, fake_config, task_name):
    from dataclasses import replace

    from llm_eval.cli import main
    from llm_eval.core.settings import DEFAULT_DATASET_PATHS, NUMPY_IMAGE, DatasetConfig, ExecutionConfig
    from llm_eval.tasks import TASK_REGISTRY

    _, admin, state, _ = service
    url, mock = provider
    config = replace(
        fake_config,
        dataset=DatasetConfig(task_name, repo_root() / DEFAULT_DATASET_PATHS[task_name]),
        run=replace(fake_config.run, task=task_name, execution=ExecutionConfig(image=NUMPY_IMAGE)),
    )
    task = TASK_REGISTRY[task_name](config)
    case = task.load_cases()[0]
    if task_name == "humanevalplus":
        answer = case.payload["prompt"] + case.payload["canonical_solution"]
    else:
        assert case.payload["entry_point"] == "similar_elements"
        answer = "def similar_elements(a, b):\n    return tuple(set(a) & set(b))"
    # The complete source test is passed through to the Docker grader.
    assert len(case.payload["test"]) > 1000
    mock.mode = "reference"
    mock.answers = {json.dumps(task.build_messages(case), sort_keys=True): answer}
    response = admin.post(
        "/api/jobs",
        json={
            "base_url": url,
            "api_key": "test-only-key",
            "model": "mock",
            "mode": "benchmark",
            "task": task_name,
            "limit": 1,
            "workers": 10,
        },
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    row = wait_job(admin, job_id, 300)
    assert row["status"] == "COMPLETED", row
    assert row["summary"]["status_counts"] == {"PASSED": 1}
    assert mock.count == 1
    root = state / "jobs" / job_id / "runs" / task_name / "native"
    assert main(["grade", "--run-dir", str(root), "--output-dir", str(root.parent / "regrade")]) == 0
    assert mock.count == 1


def test_reasoning_presets_reach_provider_and_saved_reports(service, provider):
    _, admin, state, _ = service
    url, mock = provider
    mock.mode = "reference"
    presets = admin.get("/api/config").json()["reasoning_presets"]
    assert len(presets["models"]) == 7
    assert all(t["id"] != "quick40" for t in admin.get("/api/config").json()["tasks"])
    for preset in presets["models"]:
        for effort in preset["efforts"]:
            response = admin.post(
                "/api/jobs",
                json={
                    "mode": "benchmark",
                    "task": "gsm",
                    "limit": 1,
                    "base_url": url,
                    "api_key": "test",
                    "model": preset["id"],
                    "reasoning_effort": effort,
                },
            )
            assert response.status_code == 202, response.text
            identifier = response.json()["id"]
            row = wait_job(admin, identifier)
            assert row["status"] == "COMPLETED", row
            assert row["reasoning_effort"] == effort
            assert row["thinking_enabled"] is (effort != "none")
            body = mock.requests[-1]
            assert body["reasoning_effort"] == effort and "thinking" not in body
            run = state / "jobs" / identifier / "runs" / "gsm" / "native"
            saved = json.loads((run / "config.json").read_text())
            assert saved["run"]["reasoning_effort"] == effort
            assert f"| Reasoning effort | `{effort}` |" in admin.get(f"/api/jobs/{identifier}/report").text
    before = mock.count
    for model, effort in (("glm-5.3-flash", "none"), ("qwen3.8-flash", "max"), ("custom", 1), ("custom", "invalid")):
        assert (
            admin.post(
                "/api/jobs",
                json={"mode": "api", "base_url": url, "api_key": "test", "model": model, "reasoning_effort": effort},
            ).status_code
            == 422
        )
    assert (
        admin.post(
            "/api/jobs", json={"mode": "benchmark", "task": "quick40", "base_url": url, "api_key": "test"}
        ).status_code
        == 422
    )
    assert mock.count == before
