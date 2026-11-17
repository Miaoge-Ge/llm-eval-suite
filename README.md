# LLM Eval Suite

[中文文档](README_CN.md) · [CLI reference](docs/reference.md) · [Web deployment](docs/deployment.md)

Evaluate model capability and API service quality through OpenAI-compatible endpoints, from the CLI or an authenticated web workbench. Every run preserves answers, configuration, timings and grading evidence.

| Function | Status | Measures |
| --- | --- | --- |
| Benchmark | Available | Code, math, knowledge and instruction following; full datasets or fixed Quick40 |
| API testing | Available | First-attempt success, streaming latency, interruptions, truncation and observable throughput |
| Model verification | Planned | Future comparisons against trusted references; no identity verdicts implemented |

## Quick start

Use Python 3.10+ and [uv](https://docs.astral.sh/uv/). Code benchmarks require Docker running Linux containers.

```bash
git clone https://github.com/Miaoge-Ge/llm-eval-suite.git
cd llm-eval-suite
uv sync --frozen --extra web
cp configs/model.example.yaml configs/model.yaml
# Set OPENAI_API_KEY and edit the endpoint/model in configs/model.yaml.
docker pull python:3.12-slim
docker build -t llm-eval-python:3.12-numpy2.2.6 -f docker/Dockerfile.numpy docker
uv run llm-eval benchmark --task quick40 --config configs/model.yaml --workers 10
uv run llm-eval api --config configs/model.yaml --workers 10
```

PowerShell: use `Copy-Item` and `$env:OPENAI_API_KEY = 'your-key'`. Keep real keys out of committed YAML. Original commands `run`, `api-test` and `quick-test` remain supported.

Full tasks: `humaneval`, `humanevalplus`, `mbpp`, `mbppplus`, `gsm`, `aime2025`, `aime2026`, `gpqa`, `mmlu_pro`, `ifeval`, `livecodebench`. LiveCodeBench requires separately downloaded private tests; see the [reference](docs/reference.md).

## Web workbench

```bash
export LLM_EVAL_ADMIN_PASSWORD='choose-a-long-private-password'
export LLM_EVAL_SESSION_SECRET="$(python -c 'import secrets; print(secrets.token_hex(32))')"
# Optional: DEEPSEEK_API_KEY supplies the default DeepSeek endpoint only.
uv run --extra web llm-eval-web
```

Open `http://127.0.0.1:8080`. Configure tests, view progress, cancel tasks, browse history, open reports and export ZIP records. One job runs at a time, with up to 10 workers and five active/queued jobs. Custom API keys remain in process memory. The default server key belongs in a protected environment file. Public deployments use HTTPS; [systemd and Caddy instructions](docs/deployment.md) are included.

## Measurement rules

- **Quick40:** 10 original cases each for code, math, knowledge and instruction following. Source IDs, hashes and selection reasons are recorded. Fixed suites reject `--limit`; selected HumanEval+/MBPP+ cases retain every test.
- **API:** 30 streaming requests across short answer, long answer and long context scenarios. One attempt per case, no automatic retry or resume. Original concurrency is preserved during offline grading.
- **Timing:** first meaningful content includes reasoning; first body content does not. Per-chunk timestamps are preserved. TPS is estimated only when token usage and a valid body window are available; otherwise it is missing. Truncation is reported separately from transport success.
- **Scoring:** API errors are separate from wrong answers. Reports include scoring coverage and errors alongside accuracy. Reasoning-only responses ending at the token limit are retained as truncated and not retried.
- **Recovery:** benchmark runs support resume; both fixed suites support offline regrading without API access. Regrading creates new results and preserves the source run.

```bash
uv run llm-eval grade --run-dir results/<model>/<task>/<run>
uv run llm-eval report --run-dir results/<model>/<task>/<run>
uv run llm-eval resume --run-dir results/<model>/<task>/<run> --config configs/model.yaml
```

## Directory layout

```text
llm_eval/
  benchmark/          # tasks, grading, Quick40, benchmark reports
  api_testing/        # API suite, streaming metrics, quality reports
  model_verification/ # roadmap only
  core/               # shared clients, runner, execution, storage
  web/                # FastAPI service and browser UI
  cli.py              # command entry points
  tasks.py            # public task registry
datasets/benchmark/   # original benchmark data
scripts/benchmark/    # dataset import / fixed suite construction
scripts/api_testing/  # API suite construction
tests/                # core, benchmark, API, integration, web
configs/              # secret-free templates and DeepSeek CI profile
deploy/               # service and reverse-proxy templates
docs/                 # CLI and deployment reference
```

## CI and real API testing

Ordinary CI runs Linux/Windows regression tests, Ruff, mypy and real Docker integration tests without model credentials. Manually dispatch **DeepSeek live evaluation** with repository secret `DEEPSEEK_API_KEY` to run 30 API requests followed by all 40 benchmark cases, both with **10 workers**.

The DeepSeek profile explicitly enables thinking and uses a **16,384-token shared reasoning/body budget**. The first live run exhausted smaller budgets during reasoning; that failed run remains in Actions. Prompts and target answer lengths are unchanged. Comparisons must use the same settings; timings reflect the runner's network.

```bash
uv sync --frozen --extra dev --extra web
uv run ruff check .
uv run ruff format --check .
uv run mypy llm_eval
uv run pytest
# After preparing Docker images:
LLM_EVAL_DOCKER_TESTS=1 uv run pytest tests/integration
```

Benchmark scores and model self-identification cannot establish provenance. The planned verification module will report comparison evidence and uncertainty.
