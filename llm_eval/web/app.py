"""Single-user web service. Credentials stay in process memory or server environment."""

from __future__ import annotations

import hashlib
import hmac
import io
import os
import secrets
import time
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, SecretStr

from llm_eval.core.settings import DEFAULT_DATASET_PATHS
from llm_eval.core.utils import repo_root
from llm_eval.web.jobs import JobManager, JobRequest


@dataclass(frozen=True)
class WebSettings:
    state_dir: Path
    admin_password: str
    session_secret: str
    default_api_key: str = ""
    default_base_url: str = "https://api.deepseek.com/v1"
    default_model: str = "deepseek-flash"

    @classmethod
    def from_env(cls) -> WebSettings:
        password = os.getenv("LLM_EVAL_ADMIN_PASSWORD", "")
        secret = os.getenv("LLM_EVAL_SESSION_SECRET", "")
        if len(password) < 12 or len(secret) < 32:
            raise RuntimeError("Set LLM_EVAL_ADMIN_PASSWORD (12+ characters) and LLM_EVAL_SESSION_SECRET (32+).")
        return cls(
            Path(os.getenv("LLM_EVAL_STATE_DIR", str(repo_root() / "results" / "web"))),
            password,
            secret,
            os.getenv("DEEPSEEK_API_KEY", ""),
            os.getenv("LLM_EVAL_DEFAULT_BASE_URL", "https://api.deepseek.com/v1"),
            os.getenv("LLM_EVAL_DEFAULT_MODEL", "deepseek-flash"),
        )


class LoginRequest(BaseModel):
    password: SecretStr


def create_app(settings: WebSettings | None = None, manager: JobManager | None = None) -> FastAPI:
    settings = settings or WebSettings.from_env()
    jobs = manager or JobManager(settings.state_dir / "jobs")
    static = Path(__file__).with_name("static")
    failed_logins: dict[str, list[float]] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        jobs.close()

    app = FastAPI(title="LLM Eval Suite", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.jobs = jobs

    @app.middleware("http")
    async def protect_request(request: Request, call_next):
        if request.method in {"POST", "PUT", "DELETE", "PATCH"}:
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "不允许跨站操作。"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        # Pydantic's default response includes submitted values, including credentials.
        return JSONResponse({"detail": "提交参数无效，请检查模型、并发数和输出预算。"}, status_code=422)

    def signature(payload: str) -> str:
        return hmac.new(settings.session_secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

    def authenticated(request: Request) -> None:
        token = request.cookies.get("llm_eval_session", "")
        try:
            payload, signed = token.rsplit(".", 1)
            expires = int(payload.split(".", 1)[0])
            valid = expires > time.time() and hmac.compare_digest(signed, signature(payload))
        except (ValueError, TypeError):
            valid = False
        if not valid:
            raise HTTPException(401, "请先登录。")

    @app.post("/api/login")
    def login(payload: LoginRequest, request: Request, response: Response):
        address = request.client.host if request.client else "unknown"
        now = time.time()
        history = [stamp for stamp in failed_logins.get(address, []) if now - stamp < 600]
        if len(history) >= 8:
            raise HTTPException(429, "登录尝试过多，请十分钟后再试。")
        actual = hashlib.sha256(payload.password.get_secret_value().encode()).digest()
        expected = hashlib.sha256(settings.admin_password.encode()).digest()
        if not hmac.compare_digest(actual, expected):
            failed_logins[address] = [*history, now]
            raise HTTPException(401, "登录密码不正确。")
        failed_logins.pop(address, None)
        token = f"{int(now) + 43200}.{secrets.token_hex(16)}"
        response.set_cookie(
            "llm_eval_session",
            token + "." + signature(token),
            max_age=43200,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="strict",
        )
        return {"ok": True}

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    api = APIRouter(prefix="/api", dependencies=[Depends(authenticated)])

    @api.post("/logout")
    def logout(response: Response):
        response.delete_cookie("llm_eval_session")
        return {"ok": True}

    @api.get("/config")
    def configuration():
        labels = {
            "quick40": "Quick40 · 固定 40 题",
            "humaneval": "HumanEval · 代码生成",
            "humanevalplus": "HumanEval+ · 完整扩展测试",
            "mbpp": "MBPP · Python 编程",
            "mbppplus": "MBPP+ · 完整扩展测试",
            "gsm": "GSM8K · 数学推理",
            "aime2025": "AIME 2025 · 数学竞赛",
            "aime2026": "AIME 2026 · 数学竞赛",
            "gpqa": "GPQA · 专业知识",
            "mmlu_pro": "MMLU-Pro · 多学科知识",
            "ifeval": "IFEval · 指令遵循",
            "livecodebench": "LiveCodeBench · 编程竞赛",
        }
        tasks = []
        for task, label in labels.items():
            path = repo_root() / DEFAULT_DATASET_PATHS[task]
            ready = path.is_file()
            if task == "livecodebench":
                ready = ready and (path.parent / "LiveCodeBench.private").is_dir()
            tasks.append({"id": task, "label": label, "available": ready})
        return {
            "tasks": tasks,
            "base_url": settings.default_base_url,
            "model": settings.default_model,
            "server_key_available": bool(settings.default_api_key),
            "max_workers": 10,
            "model_verification": "planned",
        }

    @api.get("/jobs")
    def list_jobs():
        return jobs.list()

    @api.post("/jobs", status_code=202)
    def create_job(payload: JobRequest):
        endpoint = urlsplit(payload.base_url)
        if endpoint.scheme not in {"http", "https"} or not endpoint.hostname or endpoint.username or endpoint.password:
            raise HTTPException(422, "API 地址必须是没有内嵌凭据的 HTTP(S) URL。")
        api_key = payload.api_key.get_secret_value()
        if not api_key and payload.base_url.rstrip("/") == settings.default_base_url.rstrip("/"):
            api_key = settings.default_api_key
        if not api_key:
            raise HTTPException(422, "请输入此 API 地址对应的密钥。")
        try:
            return jobs.submit(payload, api_key)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    def lookup(job_id: str) -> dict[str, Any]:
        try:
            return jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(404, "任务不存在。") from exc

    @api.get("/jobs/{job_id}")
    def get_job(job_id: str):
        return lookup(job_id)

    @api.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        lookup(job_id)
        jobs.cancel(job_id)
        return {"ok": True}

    @api.get("/jobs/{job_id}/report", response_class=PlainTextResponse)
    def report(job_id: str):
        lookup(job_id)
        manifest = jobs.run_directory(job_id)
        path = manifest.parent / "report.md" if manifest else None
        if path is None or not path.is_file():
            raise HTTPException(404, "报告尚未生成。")
        return PlainTextResponse(path.read_text(encoding="utf-8"))

    @api.get("/jobs/{job_id}/artifacts")
    def artifacts(job_id: str):
        lookup(job_id)
        manifest = jobs.run_directory(job_id)
        if manifest is None:
            raise HTTPException(404, "任务尚未生成记录。")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in manifest.parent.rglob("*"):
                if path.is_file() and path.suffix in {".json", ".jsonl", ".md", ".gz"}:
                    archive.write(path, path.relative_to(manifest.parent))
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{job_id}.zip"'},
        )

    app.include_router(api)

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/static/{filename}")
    def asset(filename: str):
        if filename not in {"app.js", "styles.css"}:
            raise HTTPException(404)
        return FileResponse(static / filename)

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(
        "llm_eval.web.app:create_app",
        factory=True,
        host=os.getenv("LLM_EVAL_HOST", "127.0.0.1"),
        port=int(os.getenv("LLM_EVAL_PORT", "8080")),
        proxy_headers=True,
        forwarded_allow_ips="127.0.0.1,::1",
    )
