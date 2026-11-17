# LLM Eval Suite

[中文文档](README_CN.md) · [CLI reference](docs/reference.md) · [Web deployment](docs/deployment.md) · [CI](https://github.com/Miaoge-Ge/llm-eval-suite/actions/workflows/ci.yml)

Evaluate model capability and API service quality through OpenAI-compatible endpoints, from the CLI or a registration-free web workbench. Every run preserves answers, configuration, timings and grading evidence.

| Function | Status | Measures |
| --- | --- | --- |
| Benchmark | Available | Code, math, knowledge and instruction following; standard datasets with optional case limits |
| API testing | Available | 60 fixed requests covering service quality, streaming latency, output checks and token usage |
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
uv run llm-eval benchmark --task gsm --limit 10 --config configs/model.yaml --workers 10
uv run llm-eval api --config configs/model.yaml --workers 10
```

PowerShell: use `Copy-Item` and `$env:OPENAI_API_KEY = 'your-key'`. Keep real keys out of committed YAML. Original commands `run` and `api-test` remain supported.

Full tasks: `humaneval`, `humanevalplus`, `mbpp`, `mbppplus`, `gsm`, `aime2025`, `aime2026`, `gpqa`, `mmlu_pro`, `ifeval`, `livecodebench`. LiveCodeBench requires separately downloaded private tests; see the [reference](docs/reference.md).

## Benchmark datasets

The following counts describe the files currently bundled with this repository, not necessarily the complete upstream datasets. The web selector displays the installed dataset size and updates the planned case count when you change the dataset or case limit.

| Dataset | CLI task | Bundled cases |
| --- | --- | ---: |
| HumanEval | `humaneval` | 164 |
| HumanEval+ | `humanevalplus` | 164 |
| MBPP | `mbpp` | 974 |
| MBPP+ | `mbppplus` | 378 |
| GSM8K | `gsm` | 1,319 |
| AIME 2025 | `aime2025` | 30 |
| AIME 2026 | `aime2026` | 30 |
| GPQA | `gpqa` | 198 |
| MMLU-Pro | `mmlu_pro` | 420 |
| IFEval | `ifeval` | 541 |
| LiveCodeBench | `livecodebench` | 219 |

Leaving the case limit blank runs the entire installed dataset. A limit larger than the dataset runs only the available cases. Counts refer to questions; every selected Plus question still runs its full expanded tests. LiveCodeBench needs its additional private test assets before it becomes available. Quick40 and the `quick-test` command have been removed.

## Web workbench

[Native architecture](docs/native-service.md) · [Concurrency and migration validation](docs/validation-native-2026-11-17.md)

The production web service uses C++20 on Linux for HTTP, signed anonymous sessions, scheduling and model HTTP/SSE requests. Existing Python benchmark graders and isolated Python code execution preserve the scoring rules. Build dependencies and HTTPS setup are in the [deployment guide](docs/deployment.md).

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
# Required for protected maintenance APIs; the web UI has no admin login page.
export LLM_EVAL_ADMIN_PASSWORD='choose-a-long-private-password'
export LLM_EVAL_SESSION_SECRET="$(python -c 'import secrets; print(secrets.token_hex(32))')"
# Optional: DEEPSEEK_API_KEY is available only to authenticated maintenance API clients.
uv run --extra web llm-eval-web
```

Open `http://127.0.0.1:8080`. Visitors enter directly without registering or logging in and supply their own model API key. The admin button and login page have been removed; connection failures offer a reconnect action. A signed, random browser cookie identifies the workspace for 30 days; every list, detail, report, download and cancellation checks ownership. IP addresses are only used for rate limiting. Tabs in the same browser share a workspace; clearing cookies or switching browsers loses access to previous anonymous jobs. There is no account recovery.

Defaults: 20 running jobs, 40 model requests, four scoring processes, 200 active/queued jobs, and two active/queued jobs per visitor. Each job requests at most 10 workers. Shared resources may cause queueing; request scheduling delay is reported separately from API latency. A 4-core/4-GB host is not a promise of 100 independent CPU-heavy evaluations at once. Custom API keys remain in memory and never enter scoring processes. Public deployments require HTTPS.

The web form offers model-specific reasoning levels and sends the selected `reasoning_effort` unchanged to the OpenAI-compatible gateway. `none` requests disabled reasoning. Presets describe the configured model list, not verified gateway capabilities. See [API60 and reasoning settings](docs/api-testing.md).

## API60 and token usage

| Scenario | Requests | Checks |
| --- | ---: | --- |
| Short answers | 10 | Response latency and completion |
| Long answers | 10 | Sustained output, stalls and throughput |
| Long context | 10 | Response under approximately 4K reference input tokens |
| JSON output | 10 | JSON mode support, object structure and values |
| Multi-turn context | 10 | Recall of corrected context |
| Multilingual echo | 10 | Exact Chinese, English, accented-character and emoji output |

Requests are interleaved across scenarios. Failures are retained with no automatic retries or fallback; output checks are reported separately from transport success. The results view includes per-scenario success, first-body P50/P95, truncation and output checks.

The results view and exported report show **input, output and total tokens**. The web counters update as responses arrive, including failed or truncated requests with known usage. Output includes reasoning tokens when the provider reports them there; reasoning and cached tokens are not added twice. Missing usage remains unknown, and partial totals show how many requests lack usage. Total tokens are derived from input + output only when both are known. These are observed token counts, not a billing estimate.

## Measurement rules

- **Benchmark:** select a standard dataset; Plus cases retain their full expanded tests.
- **API60:** ten requests each for short answers, long answers, long context, JSON output, multi-turn context, and multilingual echo. One attempt per case, no retries or resume. Output checks are separate from stream success; original concurrency is preserved during offline grading.
- **Timing:** first meaningful content includes reasoning; first body content does not. Per-chunk timestamps are preserved. TPS is estimated only when token usage and a valid body window are available; otherwise it is missing. Truncation is reported separately from transport success.
- **Scoring:** API errors are separate from wrong answers. Reports include scoring coverage and errors alongside accuracy. Reasoning-only responses ending at the token limit are retained as truncated and not retried.
- **Recovery:** benchmark runs support resume; benchmark and API runs support offline regrading without API access. Regrading creates new results and preserves the source run.

```bash
uv run llm-eval grade --run-dir results/<model>/<task>/<run>
uv run llm-eval report --run-dir results/<model>/<task>/<run>
uv run llm-eval resume --run-dir results/<model>/<task>/<run> --config configs/model.yaml
```

## Directory layout

```text
llm_eval/
  benchmark/          # tasks, grading, benchmark reports
  api_testing/        # API suite, streaming metrics, quality reports
  model_verification/ # roadmap only
  core/               # shared clients, runner, execution, storage
  web/                # browser UI, native launcher; legacy baseline retained
  native_bridge.py    # official scoring and archive RPC (no model credentials)
  cli.py              # command entry points
  tasks.py            # public task registry
server/               # C++20 HTTP, auth, queue, libcurl SSE and scorer pool
datasets/benchmark/   # original benchmark data
scripts/benchmark/    # standard dataset import
scripts/api_testing/  # API suite construction
scripts/native/       # real-model validation through the C++ service
tests/                # core, benchmark, API, integration, web
configs/              # secret-free templates and DeepSeek CI profile
deploy/               # service and reverse-proxy templates
docs/                 # CLI and deployment reference
```

## CI and real API testing

[Historical migration validation](docs/validation-native-2026-11-17.md) describes the previous release. See [API60 and reasoning settings](docs/api-testing.md) for the current rules.

Ordinary CI runs Linux/Windows regression tests, browser UI logic checks, Ruff, mypy and real Docker integration tests, a Release C++ build, and native HTTP failure/isolation tests without model credentials. Manually dispatch **DeepSeek live evaluation** with repository secret `DEEPSEEK_API_KEY` to run the C++ production service for 60 API requests followed by a selected standard dataset (GSM8K, 10 cases by default), both with **10 workers**.

The DeepSeek profile explicitly enables thinking and uses a **16,384-token shared reasoning/body budget**. Compare runs with the same suite, concurrency, reasoning settings and output budget; timings reflect the runner's network.

The [2026-11-17 live run](https://github.com/Miaoge-Ge/llm-eval-suite/actions/runs/44046356695) completed **60/60 API requests**, with **30/30 output checks**, no truncation and no extra attempts. It reported **44,782 input + 39,432 output = 84,214 total tokens**. This is one session using `deepseek-flash`, not validation of every reasoning preset or a long-term availability guarantee.

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
