import json
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from llm_eval.core.clients import GenerationResult
from llm_eval.web.app import WebSettings, create_app
from llm_eval.web.jobs import JobManager, JobRequest


@pytest.fixture
def settings(tmp_path):
    return WebSettings(tmp_path, "a-long-test-password", "session-secret-for-tests-" * 2, "server-secret-api-key")


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings), base_url="https://testserver") as client:
        yield client


def login(client):
    response = client.post("/api/login", json={"password": "a-long-test-password"})
    assert response.status_code == 200
    assert "HttpOnly" in response.headers["set-cookie"] and "Secure" in response.headers["set-cookie"]


def test_authentication_protects_jobs_and_downloads(client):
    assert client.get("/api/health").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/api/jobs").status_code == 401
    assert client.get("/api/jobs/missing/artifacts").status_code == 401
    assert client.post("/api/login", json={"password": "wrong"}).status_code == 401
    login(client)
    assert client.get("/api/jobs").json() == []
    assert client.get("/api/config").json()["model_verification"] == "planned"
    tasks = client.get("/api/config").json()["tasks"]
    assert next(task for task in tasks if task["id"] == "humaneval")["total_cases"] == 164
    assert "server-secret-api-key" not in client.get("/api/config").text
    client.post("/api/logout")
    assert client.get("/api/jobs").status_code == 401


def test_cross_origin_mutations_are_rejected(client):
    login(client)
    response = client.post("/api/logout", headers={"Origin": "https://another.example"})
    assert response.status_code == 403
    assert client.get("/api/jobs").status_code == 200


def test_default_key_is_not_sent_to_other_hosts_and_errors_hide_input(client):
    login(client)
    request = {"mode": "api", "base_url": "https://other.example/v1"}
    assert client.post("/api/jobs", json=request).status_code == 422
    request.update(api_key="do-not-echo-this", workers=0)
    response = client.post("/api/jobs", json=request)
    assert response.status_code == 422 and "do-not-echo-this" not in response.text
    assert client.post("/api/jobs", json={"mode": "api", "limit": 1}).status_code == 422
    assert client.post("/api/jobs", json={"mode": "benchmark", "task": "api_experience"}).status_code == 422
    assert client.get("/api/jobs/nonexistent/report").status_code == 404


def test_job_runs_through_real_runner_and_never_persists_key(settings, monkeypatch):
    class AnswerClient:
        def __init__(self, config):
            self.client = SimpleNamespace(close=lambda: None)

        def generate(self, messages):
            return GenerationResult("#### $18", {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8})

    monkeypatch.setattr("llm_eval.core.runner.OpenAICompatibleClient", AnswerClient)
    with TestClient(create_app(settings), base_url="https://testserver") as client:
        login(client)
        created = client.post("/api/jobs", json={"mode": "benchmark", "task": "gsm", "limit": 1}).json()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            job = client.get(f"/api/jobs/{created['id']}").json()
            if job["status"] not in {"QUEUED", "RUNNING"}:
                break
            time.sleep(0.02)
        assert job["status"] == "COMPLETED", job
        assert job["summary"]["status_counts"] == {"PASSED": 1}
        assert len(job["cases"]) == 1
        assert client.get(f"/api/jobs/{created['id']}/report").status_code == 200
        assert client.get(f"/api/jobs/{created['id']}/artifacts").content.startswith(b"PK")
    for path in settings.state_dir.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".md"}:
            assert settings.default_api_key not in path.read_text(encoding="utf-8")


def test_restart_marks_unfinished_jobs_without_losing_records(tmp_path):
    directory = tmp_path / "existing"
    directory.mkdir()
    path = directory / "job.json"
    path.write_text(json.dumps({"id": "existing", "status": "RUNNING", "created_at": 1}), encoding="utf-8")
    manager = JobManager(tmp_path)
    try:
        assert manager.get("existing")["status"] == "INTERRUPTED"
        assert path.exists()
    finally:
        manager.close()


def test_pending_job_can_be_cancelled_without_api_call(tmp_path, monkeypatch):
    import threading

    entered, release = threading.Event(), threading.Event()
    calls = []

    class WaitingRunner:
        def __init__(self, config):
            self.config = config
            self.cancel_event = threading.Event()

        def run(self):
            from llm_eval.core.reporting import RunSummary

            calls.append(self.config)
            entered.set()
            assert release.wait(5)
            summary = RunSummary("gsm", "test", 0, 0, 0, 0, "00:00:00", 0, 0, {}, {}, run_status="COMPLETED")
            return summary, None

    monkeypatch.setattr("llm_eval.web.jobs.EvaluationRunner", WaitingRunner)
    manager = JobManager(tmp_path)
    try:
        request = JobRequest(mode="benchmark", task="gsm", limit=1)
        manager.submit(request, "secret")
        assert entered.wait(5)
        second = manager.submit(request, "secret")
        manager.cancel(second["id"])
        assert manager.get(second["id"])["status"] == "INTERRUPTED"
        assert len(calls) == 1
    finally:
        release.set()
        manager.close()
