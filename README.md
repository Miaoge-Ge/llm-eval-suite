# LLM Eval Framework

[中文文档](README_CN.md)

A clean, engineering-first evaluation framework for LLM benchmarks. It uses `uv`-managed environments, a single minimal model config, and produces one self-contained Markdown report per run against any OpenAI-compatible endpoint.

## Highlights

- `uv` as the default environment and dependency workflow
- Installable package (`hatchling` build backend) with a `llm-eval` console script
- Single model config file, with `${ENV_VAR}` placeholders for secrets
- 10 built-in tasks across code generation, math reasoning, and multiple-choice knowledge
- OpenAI-compatible client with streaming, retries, and usage accounting
- One self-contained Markdown report per run (config + metrics + per-case results)
- Per-subject / per-domain accuracy breakdown for all knowledge tasks
- Optional thinking mode with configurable `reasoning_effort`
- Live progress bar with pass/fail counts during evaluation
- Quality tooling wired in: `ruff` (lint + format), `mypy` (type checking), `pytest`

## Project structure

```text
llm_eval/
  cli.py
  clients.py
  execution.py         # subprocess execution of generated code
  extraction.py        # answer extraction (code blocks, \boxed{}, letters, numbers)
  reporting.py         # run persistence (JSONL + config snapshot) and Markdown rendering
  runner.py
  settings.py
  status.py            # canonical case-status constants
  tasks.py
  utils.py
  ifeval/              # vendored Google IFEval checkers + scoring glue
configs/
  model.example.yaml   # committed template (no secrets)
datasets/
  HumanEval.jsonl
  HumanEvalPlus.jsonl
  MBPP.jsonl
  MBPPPlus.jsonl
  GSM8K.jsonl
  AIME2025.jsonl
  AIME2026.jsonl
  GPQA.jsonl
  MMLUPro.jsonl
  IFEval.jsonl
  LiveCodeBench.jsonl
scripts/
  fetch_datasets.py    # re-download / regenerate datasets from Hugging Face
tests/
pyproject.toml
```

## Install

Recommended:

```bash
uv sync
```

Install with dev dependencies (`pytest`, `ruff`, `mypy` and type stubs):

```bash
uv sync --extra dev
```

## Model config

Copy the committed template to a local config (the `configs/` directory is git-ignored apart from the example, so your real keys never get committed):

```bash
cp configs/model.example.yaml configs/model.yaml
```

[configs/model.example.yaml](configs/model.example.yaml):

```yaml
base_url: ${OPENAI_BASE_URL:-https://api.openai.com/v1}
api_key: ${OPENAI_API_KEY}
model_name: gpt-4.1-mini
workers: 10
timeout_seconds: 120
execution_timeout_seconds: 20
thinking_enabled: false
reasoning_effort:
```

Prefer `${ENV_VAR}` placeholders for secrets so the API key is never written to disk. Recognized fields:

| Field | Purpose |
| --- | --- |
| `base_url` | OpenAI-compatible endpoint |
| `api_key` | API key (use `${OPENAI_API_KEY}`) |
| `model_name` | Model id sent to the endpoint; also names the output directory |
| `workers` | Concurrent requests (default `10`) |
| `timeout_seconds` | Per-request HTTP timeout (default `120`) |
| `execution_timeout_seconds` | Cap for running generated code during grading (default `20`) |
| `thinking_enabled` | `true`/`false` (also accepts `enabled`/`disabled`) |
| `reasoning_effort` | `low`/`medium`/`high`/`max`, sent only when thinking is enabled |

Each run writes to its own timestamped directory under `results/<model_name>/<task>/` (the model name is slugified, so ids like `org/model:beta` are safe on every platform).

## CLI

```bash
# run a benchmark
uv run llm-eval run --config configs/model.yaml --task gpqa

# equivalent module form
uv run python -m llm_eval run --config configs/model.yaml --task gsm

# smoke test: only the first 5 cases, with 2 workers
uv run llm-eval run --config configs/model.yaml --task gsm --limit 5 --workers 2

# list all available tasks
uv run llm-eval run --list-tasks
```

`--config` defaults to `configs/model.yaml` and `--task` defaults to `humaneval`. `--workers` overrides the config value; `--limit N` runs only the first N cases — handy for smoke-testing a new endpoint before a full run.

### Run every benchmark

```bash
# Code generation
uv run llm-eval run --config configs/model.yaml --task humaneval
uv run llm-eval run --config configs/model.yaml --task humanevalplus
uv run llm-eval run --config configs/model.yaml --task mbpp
uv run llm-eval run --config configs/model.yaml --task mbppplus
uv run llm-eval run --config configs/model.yaml --task livecodebench

# Math reasoning
uv run llm-eval run --config configs/model.yaml --task gsm
uv run llm-eval run --config configs/model.yaml --task aime2025
uv run llm-eval run --config configs/model.yaml --task aime2026

# Multiple-choice knowledge
uv run llm-eval run --config configs/model.yaml --task gpqa
uv run llm-eval run --config configs/model.yaml --task mmlu_pro

# Instruction following
uv run llm-eval run --config configs/model.yaml --task ifeval
```

When a run starts, the framework prints the selected task, model, dataset path, worker count, thinking mode, and output directory, then shows a live progress bar (passed / failed / last HTTP status) while cases are processed.

## Thinking mode

For providers that support it, the client sends `reasoning_effort` and `extra_body={"thinking": {"type": "enabled"}}` when `thinking_enabled` is on. Equivalent SDK call:

```python
from openai import OpenAI

client = OpenAI(api_key="...", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
)
```

## Supported tasks

### Code generation

| Task | Dataset | Cases | Grading |
| --- | --- | --- | --- |
| `humaneval` | HumanEval | 164 | Execute generated code against unit tests |
| `humanevalplus` | HumanEval+ | 164 | Extended tests with real NumPy 2.2.6 |
| `mbpp` | MBPP | 974 | Execute generated code against assertion tests |
| `mbppplus` | MBPP+ | 378 | EvalPlus extended tests with real NumPy 2.2.6 |
| `livecodebench` | LiveCodeBench stdin subset | 219 | 604 public and 2,502 private tests |

### Math reasoning

| Task | Dataset | Cases | Grading |
| --- | --- | --- | --- |
| `gsm` | GSM8K | 1319 | Exact match on the final `#### <number>` answer |
| `aime2025` | AIME 2025 (I+II) | 30 | Integer match on the `\boxed{}` answer |
| `aime2026` | AIME 2026 (I+II) | 30 | Integer match on the `\boxed{}` answer |

### Multiple-choice knowledge

The MCQ task asks the model to output `\boxed{A/B/C/D}` and grades by exact letter match. The report includes a per-domain accuracy breakdown.

| Task | Dataset | Cases | Coverage |
| --- | --- | --- | --- |
| `gpqa` | GPQA-Diamond | 198 | PhD-level science (Physics, Chemistry, Biology) |
| `mmlu_pro` | MMLU-Pro | 420 | 14 subjects, 10-way choices (30 sampled per category) |

### Instruction following

| Task | Dataset | Cases | Grading |
| --- | --- | --- | --- |
| `ifeval` | IFEval | 541 | Programmatic verification of each instruction (prompt-level strict accuracy) |

`livecodebench` keeps the 219 stdin/stdout problems (AtCoder/Codeforces) from the pinned `test.jsonl` release; it does not include functional/LeetCode problems or later releases. Reports and CLI output explicitly state the subset and public/private test coverage. Legacy public-only snapshots remain usable, with an incomplete-coverage warning; their scores must not be presented as full-test scores. Output comparison follows stripped lines, then exact decimal-token equality, without float tolerance or token reordering.

Private tests are compressed per-problem assets under `datasets/LiveCodeBench.private/` (about 818 MiB, excluded from Git). Prepare them on a fresh checkout:

```bash
uv run --with huggingface-hub python scripts/fetch_datasets.py livecodebench
# Or use an already downloaded upstream file:
uv run python scripts/fetch_datasets.py livecodebench --source /path/to/test.jsonl
```

The importer pins dataset revision `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`, rejects missing/incompatible private tests, and writes the index atomically. It decodes upstream private data with a data-only unpickler and a 512 MiB decompression bound per problem. Runs verify SHA-256 and snapshot selected assets, so offline grading survives removal of the original dataset. Tests are read one at a time; selected assets consume disk space in each run. Missing/corrupt assets fail before model requests. `ifeval` vendors Google's instruction checkers under `llm_eval/ifeval/` and depends on `langdetect`. Other datasets can be regenerated with `uv run --with datasets python scripts/fetch_datasets.py <name>`.

### Plus runtime and grading

Plus tasks use real, pinned NumPy, including arrays, broadcasting, infinities, and dtype behavior. Build the runtime once:

```bash
docker build -t llm-eval-python:3.12-numpy2.2.6 -f docker/Dockerfile.numpy docker
```

Without an explicit `execution.image`, Plus selects this image; other code tasks use `python:3.12-slim`. Custom images must supply NumPy 2.2.6. A dependency probe executes inside the configured sandbox before generation; missing/wrong NumPy fails as infrastructure, not a wrong answer. BLAS threads are capped at one. Local development uses the same pinned project dependency. Old Plus answers can be regraded with `grade --run-dir PATH --execution-image llm-eval-python:3.12-numpy2.2.6`.

The exported Plus tests enforce type and sequence-length checks before tolerant `allclose` comparisons, preventing accidental broadcasting from accepting malformed answers. The broken exported HumanEval/32 root oracle is corrected to check the absolute polynomial residual using the scalar candidate result. Input corpora and tolerances are unchanged. Execution time/memory limits remain configurable and differ from EvalPlus's reference-dependent timing policy; a correct but slow or memory-heavy implementation can still reach the configured limits. Windows local floating-point results may differ from the bundled Linux-generated expectations; Docker Linux is the default grading environment.

## Output

Each run writes to its own timestamped directory:

```text
results/<model_name>/<task>/<YYYYMMDD-HHMMSS>/
  manifest.json  # versioned run identity and reproducibility fingerprint
  dataset.jsonl  # selected cases, including grading inputs
  generations/   # immutable raw answers and request / retry metadata
  cases/         # atomic per-case result checkpoints
  results.jsonl   # incremental results; rebuilt from checkpoints on resume
  config.json     # snapshot of the effective config (API key masked)
  report.md       # the human-readable report
  status.json     # atomically updated lifecycle state
  summary.json    # final machine-readable metrics
```

Raw answers are atomically saved before grading, preserving whitespace, reasoning content, partial failed responses, request messages, effective parameters, per-attempt metadata, and known usage. API keys are masked in configuration snapshots and omitted from request parameters. Case checkpoints are authoritative; resume can rebuild a truncated `results.jsonl`. `sessions/` records each session's configuration, and `history/` preserves the prior JSONL when resuming. New runs use new directories.

### Generate, resume, and regrade offline

```bash
# Save answers without running generated code or grading
uv run llm-eval generate --config configs/model.yaml --task humaneval --limit 10
# Continue the original operation in its existing directory
uv run llm-eval resume --config configs/model.yaml --run-dir results/<model>/<task>/<run>
# Grade saved answers with the current grader, without model access or an API key
uv run llm-eval grade --run-dir results/<model>/<task>/<run>
# Rebuild a completed run's Markdown report without grading or model access
uv run llm-eval report --run-dir results/<model>/<task>/<run>
```

`run --resume PATH` and `generate --resume PATH` also resume the saved operation. Finished cases are retained; saved complete answers awaiting grading are reused without another request. API failures can be retried, with historical attempts and usage retained. Reports distinguish cumulative cost from this session's new requests and tokens. `GENERATED` means an answer was saved, not that it passed; generation-only runs have zero grading coverage.

Resume requires matching dataset, selected cases, prompts, model parameters, execution settings, framework code, Python version, and dependency lock. Workers and API keys may change. A mismatch is rejected before changing the existing run; start a new run or use `grade` after changing the grader. Pin Docker images by digest when reproducibility matters; a mutable image tag cannot guarantee an identical environment.

Offline `grade` uses the saved dataset snapshot and answers, writes a separate run under `<source>/regrades/` (or `--output-dir`), and records its source. It never contacts the model, and its API attempts / token cost are zero. Code grading defaults to Docker even if the source used local execution; `--execution-backend local` explicitly opts into trusted local execution. Missing or incomplete answers become `MISSING_GENERATION`, not fresh requests. `STORAGE_ERROR` aborts before grading an answer that could not be saved. A crashed grading run can be regraded again from its source. Legacy runs without the raw-answer archive cannot recover answers that were never recorded. Report rebuilding requires a final summary; resume or regrade an interrupted run first.

`report.md` contains:

- **Overview** — task, model, dataset, workers, thinking mode
- **Metrics** — pass rate, total/completed/graded cases, wall clock, throughput, prompt/completion/total tokens (graded = passed + failed + timeout; API errors are excluded from the pass-rate denominator)
- **Status counts** — how many cases passed, failed, errored, etc.
- **Accuracy by domain** — per-subject breakdown (knowledge tasks only)
- **Results** — a per-case table with status, time, tokens, and a detail column

Code tasks default to Docker Linux containers. Preflight checks the runtime, resource-limit capabilities, and locally available image before model requests. Missing prerequisites fail closed; math, knowledge, and IFEval tasks do not require Docker.

### Execution isolation and limits

On Windows, use Docker Desktop's Linux container engine. Prepare the image once with `docker pull python:3.12-slim`. Configure execution in the model YAML:

```yaml
execution:
  backend: docker
  # image: python:3.12-slim  # optional; Plus defaults to the NumPy image
  memory_mb: 256
  cpus: 1.0
  pids_limit: 64
  max_output_bytes: 1048576
  tmpfs_mb: 32
```

Each execution uses a separate container with no host mounts, no model secrets, no network, a non-root user, a read-only root filesystem, and dropped capabilities. Memory and PID limits apply to the entire container; `cpus` controls its CPU quota. `/work` and `/tmp` each receive a tmpfs of `tmpfs_mb`. stdout and stderr share the bounded output budget. `execution_timeout_seconds` caps the attached execution's wall time; Docker control operations have separate 15-second timeouts.

Containers and local descendants are cleaned up after success, failure, timeout, or cancellation. Cleanup failures become `EXECUTION_ERROR` and abort the run, retaining the container name in the error. Images are never automatically pulled; use a digest to pin the execution environment. Containers are not VMs, and the completion receipt is not a defense against deliberate grader tampering.

Explicit `backend: local` is for trusted development code only. It uses a temporary working directory, an environment allowlist, bounded output, deadlines, and process-tree cleanup. Windows Job Objects enforce aggregate memory/PID limits; POSIX uses process groups and per-process resource limits. Local mode does not isolate the host filesystem or network, provide Docker CPU/disk quotas, or provide aggregate POSIX memory/PID limits. Windows venv launchers typically occupy two process slots.

### Run state and exit codes

`status.json` starts as `RUNNING` and ends as `COMPLETED`, `FAILED`, `ABORTED`, or `INTERRUPTED`. Interrupted and failed runs retain results, `CANCELLED` records for unevaluated cases, a report, and `summary.json`. At most `workers` cases are in flight; fatal API or execution errors stop further scheduling.

| Exit code | Meaning |
| --- | --- |
| 0 | Evaluation completed; wrong answers, timeouts, and resource-limit failures are still valid graded outcomes |
| 1 | Invalid configuration |
| 2 | Incomplete evaluation or API / execution / framework errors |
| 130 | User interruption |

Ctrl+C stops scheduling, execution, and retries, attempts to close active model streams, and preserves known usage. HTTP requests awaiting a response may still wait for `timeout_seconds`. Force-killing the host process or stopping Docker can prevent cleanup. Reports show graded accuracy, grading coverage, cancellations, and infrastructure error rate. `OUTPUT_LIMIT` and `RESOURCE_LIMIT` are graded failures; `EXECUTION_ERROR` and `CANCELLED` are excluded from the accuracy denominator.

### Grading and retry rules

- AIME accepts a 1–3 digit nonnegative integer in the last `\boxed{}` (including `070`), or an integer on the final line when no box is present. Decimals are not rounded and expressions are not reduced to numeric fragments.
- Multiple-choice boxes must match a complete answer such as `B`, `(B)`, or `Answer: B`. Without a box, the last explicit answer declaration or a standalone final-line option is used. Invalid final answers do not revive earlier answers.
- GSM8K compares exact finite decimal values without floating-point rounding or tolerance. It uses the last `####` marker; an invalid final marker does not fall back. Without a marker, only a numeric final line or a complete `answer is` / `answer:` / `total is` / `total:` declaration is accepted. Fractions, expressions, and numbers merely mentioned in reasoning are not extracted as answers. Invalid reference answers are framework errors, not model failures.
- Python extraction removes common indentation before trimming whitespace, preserving nested blocks in function-body completions.
- IFEval initializes language-detection profiles before starting workers and fixes the detector seed to 0, making language scoring repeatable across threads and runs.
- Assertion test programs must finish normally. Early `sys.exit(0)` / `os._exit(0)` calls do not pass. Empty test sets produce `INTERNAL_ERROR` before any model request. Completion checking is not a tamper-resistant sandbox; stdin/stdout problems continue to be judged by their output.
- Each case makes at most 3 API attempts, including the initial request; SDK retries are disabled. Connection/transport errors, HTTP 408/409/429/5xx, and empty completions may be retried. HTTP 401/403 are immediately fatal; other HTTP client errors are not retried.
- Usage already returned by failed attempts is included in totals. JSONL records include `api_attempts` and `usage_complete`; reports summarize attempts and cases with incomplete usage. Missing provider usage is not estimated.
- Streams must provide a terminal `finish_reason` for choice 0. Clean EOF or `[DONE]` alone does not prove that the answer is complete: these incomplete attempts are retained and retried within the same budget, never graded as successful generations. A reported `length` limit remains a model outcome and is not retried. Additional choices are never concatenated into the answer.
- Resume and report rebuilding reject malformed records, negative/noninteger usage, duplicate generation IDs, unknown checkpoint statuses, and completed checkpoints linked to missing, incomplete, or other cases' answers before rewriting existing results.

## Development

Tests do not call live model APIs. Real container tests are opt-in after preparing Docker and its image: set `LLM_EVAL_DOCKER_TESTS=1`, then run `uv run pytest tests/test_docker_integration.py tests/test_lightweight_suites.py`. They are skipped otherwise; mocked Docker tests are not a substitute for container validation.

The lightweight suite tests cover fixed selection, retry-free accounting for 30 API requests, and all 40 reference answers graded in real Docker, including resume and offline regrading. API failures are injected through a mock HTTP transport.

Windows sandbox temporary paths can exceed the filesystem path limit and produce `FileNotFoundError`. Use a short project-local test directory, for example `uv run pytest --basetemp .pytest_cache/run-tmp`. Reserve that directory for test scratch files: pytest clears it.

All tooling is configured in `pyproject.toml` and runs through `uv`:

```bash
uv run ruff check .    # lint
uv run ruff format .   # auto-format
uv run mypy            # type check
uv run pytest          # tests
```
