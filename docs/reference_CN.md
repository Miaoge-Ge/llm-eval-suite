# LLM Eval Suite — reference

[English Documentation](reference.md)

一个面向工程使用的 LLM 评测框架。默认用 `uv` 管理环境，配置极简（单个模型配置文件），每次运行只产出一个自包含的 Markdown 报告，支持任意 OpenAI-compatible 接口。

## 特点

- 默认使用 `uv` 管理依赖与运行环境
- 可安装的 Python 包（`hatchling` 构建后端），提供 `llm-eval` 命令行入口
- 单一模型配置文件，密钥用 `${ENV_VAR}` 占位符注入
- 内置 10 个评测任务，涵盖代码生成、数学推理、多选知识题三大类
- OpenAI-compatible 客户端，支持流式输出、自动重试与 token 用量统计
- 每次运行只产出一个自包含 Markdown 报告（配置 + 指标 + 逐条结果）
- 知识题任务自动生成分学科 / 分领域准确率明细
- 支持 `thinking` 与可配置的 `reasoning_effort`
- 评测过程中显示实时进度条（通过数 / 失败数 / 最近 HTTP 状态）
- 内置质量工具链：`ruff`（检查 + 格式化）、`mypy`（类型检查）、`pytest`

## 目录结构

```text
llm_eval/
  benchmark/          # benchmark tasks, grading, reports
  api_testing/        # streaming quality metrics, reports, fixed 60 requests
  model_verification/ # planned; no identity verdicts yet
  core/               # clients, runners, execution, persistence
  web/                # authenticated web workbench
  cli.py
  tasks.py            # public task registry
configs/              # secret-free examples
datasets/benchmark/  # original benchmark datasets
scripts/benchmark/    # dataset import and fixed suite construction
scripts/api_testing/  # API suite construction
tests/                # mirrors feature directories
deploy/               # systemd and Caddy templates
docs/                 # reference and deployment guide
```

## 安装

推荐方式：

```bash
uv sync
```

安装开发依赖（`pytest`、`ruff`、`mypy` 及类型存根）：

```bash
uv sync --extra dev --extra web
```

## 模型配置

把已提交的模板复制为本地配置（`configs/` 目录除示例外都被 git 忽略，因此真实密钥不会被提交）：

```bash
cp configs/model.example.yaml configs/model.yaml
```

[configs/model.example.yaml](../configs/model.example.yaml)：

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

建议用 `${ENV_VAR}` 占位符注入密钥，避免把 API Key 明文写入文件。可识别的字段：

| 字段 | 含义 |
| --- | --- |
| `base_url` | OpenAI-compatible 接口地址 |
| `api_key` | API 密钥（建议用 `${OPENAI_API_KEY}`） |
| `model_name` | 发送给接口的模型 id，同时用于命名输出目录 |
| `workers` | 并发请求数（默认 `10`） |
| `timeout_seconds` | 单次请求的 HTTP 超时（默认 `120`） |
| `execution_timeout_seconds` | 打分时运行生成代码的时间上限（默认 `20`） |
| `thinking_enabled` | `true`/`false`（也接受 `enabled`/`disabled`） |
| `reasoning_effort` | `low`/`medium`/`high`/`max`，仅在开启思考模式时发送 |

每次运行写入 `results/<model_name>/<task>/` 下独立的时间戳目录（模型名会做 slug 处理,`org/model:beta` 这类 id 在任何平台都安全)。

## CLI 用法

```bash
# 运行某个评测任务
uv run llm-eval benchmark --config configs/model.yaml --task gpqa

# 等价的模块形式
uv run python -m llm_eval run --config configs/model.yaml --task gsm

# 冒烟测试：只跑前 5 个用例，并发 2
uv run llm-eval benchmark --config configs/model.yaml --task gsm --limit 5 --workers 2

# 列出所有可用任务
uv run llm-eval benchmark --list-tasks
```

`--config` 默认 `configs/model.yaml`，`--task` 默认 `humaneval`。`--workers` 覆盖配置文件中的并发数；`--limit N` 只跑前 N 个用例，适合在全量评测前先验证新接口。

### 运行全部评测

```bash
# 代码生成
uv run llm-eval benchmark --config configs/model.yaml --task humaneval
uv run llm-eval benchmark --config configs/model.yaml --task humanevalplus
uv run llm-eval benchmark --config configs/model.yaml --task mbpp
uv run llm-eval benchmark --config configs/model.yaml --task mbppplus
uv run llm-eval benchmark --config configs/model.yaml --task livecodebench

# 数学推理
uv run llm-eval benchmark --config configs/model.yaml --task gsm
uv run llm-eval benchmark --config configs/model.yaml --task aime2025
uv run llm-eval benchmark --config configs/model.yaml --task aime2026

# 多选知识题
uv run llm-eval benchmark --config configs/model.yaml --task gpqa
uv run llm-eval benchmark --config configs/model.yaml --task mmlu_pro

# 指令遵循
uv run llm-eval benchmark --config configs/model.yaml --task ifeval
```

命令启动后会立即打印：当前任务、模型、数据集路径、并发数、思考模式、输出目录，随后在处理用例时显示实时进度条（通过数 / 失败数 / 最近一次 HTTP 状态码）。

## 思考模式

对于支持该能力的服务端，当 `thinking_enabled` 开启时，客户端会发送 `reasoning_effort` 和 `extra_body={"thinking": {"type": "enabled"}}`。对应的 SDK 调用形式：

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

## 当前支持任务

### 代码生成

| 任务 | 数据集 | 题数 | 打分方式 |
| --- | --- | --- | --- |
| `humaneval` | HumanEval | 164 | 执行生成代码并跑单元测试 |
| `humanevalplus` | HumanEval+ | 164 | 扩展测试，使用真实 NumPy 2.2.6 |
| `mbpp` | MBPP | 974 | 执行生成代码并跑断言测试 |
| `mbppplus` | MBPP+ | 378 | EvalPlus 扩展测试，使用真实 NumPy 2.2.6 |
| `livecodebench` | LiveCodeBench stdin 子集 | 219 | 604 个公开测试与 2,502 个私有测试 |

### 数学推理

| 任务 | 数据集 | 题数 | 打分方式 |
| --- | --- | --- | --- |
| `gsm` | GSM8K | 1319 | 精确匹配 `#### <数字>` 末尾答案 |
| `aime2025` | AIME 2025 (I+II) | 30 | 对 `\boxed{}` 内答案做整数匹配 |
| `aime2026` | AIME 2026 (I+II) | 30 | 对 `\boxed{}` 内答案做整数匹配 |

### 多选知识题

多选任务要求模型输出 `\boxed{A/B/C/D}`，按字母精确匹配打分。报告中会自动生成分领域准确率明细。

| 任务 | 数据集 | 题数 | 覆盖范围 |
| --- | --- | --- | --- |
| `gpqa` | GPQA-Diamond | 198 | 博士级科学题（物理、化学、生物） |
| `mmlu_pro` | MMLU-Pro | 420 | 14 个学科，10 选项（每类采样 30 题） |

### 指令遵循

| 任务 | 数据集 | 题数 | 打分方式 |
| --- | --- | --- | --- |
| `ifeval` | IFEval | 541 | 对每条指令做程序化校验（prompt 级 strict 准确率） |

`livecodebench` 保留固定 `test.jsonl` 版本中的 219 道 stdin/stdout 题（AtCoder/Codeforces），不包含 functional/LeetCode 题型或后续版本。报告和 CLI 会明确显示评测子集及公开/私有测试覆盖。旧版仅含公开样例的快照仍可评分，但会标明覆盖不足，不能将其通过率作为完整测试成绩。输出比较先按行去除首尾空白，再做精确十进制 token 比较，不使用浮点容差、不忽略顺序。

私有测试按题压缩存放在 `datasets/benchmark/LiveCodeBench.private/`（约 818 MiB，不纳入 Git）。全新检出后先准备：

```bash
uv run --with huggingface-hub python scripts/benchmark/fetch_datasets.py livecodebench
# 也可以使用已下载的上游原始文件：
uv run python scripts/benchmark/fetch_datasets.py livecodebench --source /path/to/test.jsonl
```

导入器固定数据版本 `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`，拒绝私有测试缺失或题型不兼容的输入，最后原子更新索引。上游压缩数据通过禁止可执行对象的反序列化器读取，单题解压上限为 512 MiB。运行前校验 SHA-256，并将选中题目的测试文件复制到运行目录；离线评分不依赖原数据目录。测试逐条读取，每次运行会占用相应快照磁盘空间；文件缺失或损坏时，在请求模型前失败。`ifeval` 使用 `llm_eval/benchmark/ifeval/` 中的 Google 指令校验器和 `langdetect`。其他数据集可用 `uv run --with datasets python scripts/benchmark/fetch_datasets.py <name>` 重新生成。

### Plus 执行环境与评分

Plus 任务改用固定版本的真实 NumPy，支持数组、广播、无穷值和 dtype 行为。首次运行前构建镜像：

```bash
docker build -t llm-eval-python:3.12-numpy2.2.6 -f docker/Dockerfile.numpy docker
```

未显式配置 `execution.image` 时，Plus 自动选择上述镜像，其他代码任务仍用 `python:3.12-slim`。自定义镜像也必须安装 NumPy 2.2.6；框架在生成前进入实际沙箱探测依赖，缺失或版本不符记为基础设施错误，不计为模型答错。BLAS 线程数固定为 1，本地开发使用项目中同版本依赖。旧 Plus 回答可用 `grade --run-dir PATH --execution-image llm-eval-python:3.12-numpy2.2.6` 重新评分。

Plus 导出的测试程序在浮点容差比较前补齐类型和序列长度检查，避免 NumPy 广播把错误结构的答案判为通过；HumanEval/32 原先错误的求根断言已修正为使用标量结果计算多项式残差的绝对值。题目输入和容差不变。执行时限、内存仍按配置约束，与 EvalPlus 按参考耗时设置预算的策略不同；较慢或占用内存较大的正确实现仍可能达到限制。Windows 本地浮点结果可能与数据集中按 Linux 生成的期望值存在差异，默认使用 Docker Linux 评分。

## 输出产物

每次运行写入独立的时间戳目录：

```text
results/<model_name>/<task>/<YYYYMMDD-HHMMSS>/
  manifest.json  # 版本化运行清单与复现指纹
  dataset.jsonl  # 本次选中用例的快照，包含评分输入
  generations/   # 不覆盖的原始回答及请求、重试记录
  cases/         # 原子写入的逐用例检查点
  results.jsonl   # 增量结果；续跑时由检查点重建
  config.json     # 本次运行的有效配置快照（API Key 已脱敏）
  report.md       # 人类可读的评测报告
  status.json     # 原子更新的运行状态、完成数和停止原因
  summary.json    # 最终结构化指标，含覆盖率和基础设施错误率
```

原始回答先原子落盘，再进入评分，保留原始空白、推理内容、失败请求的部分回答、请求消息、实际参数、每次尝试的元数据和已知用量。配置快照中的 API key 会脱敏，请求参数不包含密钥。逐用例检查点是恢复依据，续跑可重建被截断的 `results.jsonl`。`sessions/` 保存各次会话配置，`history/` 在续跑前备份旧 JSONL；新运行使用独立目录。

### 只生成、断点续跑与离线重新评分

```bash
# 只保存回答，不执行生成代码、不评分
uv run llm-eval generate --config configs/model.yaml --task humaneval --limit 10
# 在原目录继续原来的操作
uv run llm-eval resume --config configs/model.yaml --run-dir results/<model>/<task>/<run>
# 使用当前评分器重新评分，无需模型接口或 API key
uv run llm-eval grade --run-dir results/<model>/<task>/<run>
# 根据已结束运行的检查点和汇总重新生成 Markdown 报告
uv run llm-eval report --run-dir results/<model>/<task>/<run>
```

也可使用 `run --resume PATH` 或 `generate --resume PATH`，恢复的操作以原运行清单为准。已结束的评分不重复执行；回答已保存但评分未完成时直接复用回答，不重复请求模型。API 错误可以重试，历史请求次数和用量持续累计；报告分别列出累计消耗与本次会话新增消耗。`GENERATED` 仅表示回答已保存，不表示通过，只生成模式的评分覆盖率为零。

续跑会校验数据集、选中用例、提示词、模型参数、执行配置、框架代码、Python 版本和依赖锁文件；允许修改并发数和 API key。指纹不一致时，在修改原运行产物前拒绝续跑。修改评分器后请用 `grade`，或启动新运行。需要严格复现时请把 Docker 镜像固定到 digest，可变标签无法保证执行环境完全一致。

离线 `grade` 使用保存的数据集快照和回答，在 `<原目录>/regrades/` 下创建独立结果（可用 `--output-dir` 指定），并记录来源；不会访问模型接口，API 次数和 token 消耗记为零。代码评分默认使用 Docker，即使原运行采用本地执行；只有显式传入 `--execution-backend local` 才启用可信代码的本地执行。缺失或不完整的回答记为 `MISSING_GENERATION`，不会自动补调模型。原始回答保存失败记为 `STORAGE_ERROR`，中止运行且不对该回答评分。离线评分中断后，可从原始来源再次执行 `grade`。旧版本未保存的回答无法补恢复。`report` 要求已有最终汇总；运行中断时先续跑或重新评分。

`report.md` 包含：

- **Overview** — 任务、模型、数据集、并发数、思考模式
- **Metrics** — 通过率、总用例 / 完成 / 参与打分用例数、总用时、吞吐、prompt/completion/total tokens（参与打分 = 通过 + 失败 + 超时；API 报错不计入通过率分母）
- **Status counts** — 各状态（通过 / 失败 / 报错等）计数
- **Accuracy by domain** — 分学科准确率明细（仅知识题任务）
- **Results** — 逐条结果表：状态、用时、tokens、详情列

代码任务默认使用 Docker Linux 容器；启动前检查运行时、资源限制能力和本地镜像，失败时停止，不会自动降级到本机执行。数学、知识题和 IFEval 不依赖 Docker。

### 执行隔离与资源限制

Windows 上使用 Docker Desktop 的 Linux 容器模式，并先准备镜像：

```powershell
docker pull python:3.12-slim
```

模型配置可增加以下内容，完整示例见 `configs/model.example.yaml`：

```yaml
execution:
  backend: docker
  # image: python:3.12-slim  # 可选；Plus 默认使用含 NumPy 的镜像
  memory_mb: 256
  cpus: 1.0
  pids_limit: 64
  max_output_bytes: 1048576
  tmpfs_mb: 32
```

每次执行使用独立容器，不挂载宿主目录，不传入模型密钥，禁用网络，以非 root 用户运行，根文件系统只读并移除额外权限。内存与进程数按容器整体限制；`cpus` 是 CPU 配额，`execution_timeout_seconds` 是单次执行的墙钟时间上限。`/work`、`/tmp` 各使用一个大小为 `tmpfs_mb` 的内存文件系统；标准输出和标准错误共享 `max_output_bytes` 上限。达到限制后停止执行，并限制保存在内存和结果中的输出量。

容器创建、检查、清理等控制操作各另设 15 秒超时。超时、取消、正常结束及错误退出都会清理容器或本地进程树；容器清理失败会记录为 `EXECUTION_ERROR` 并中止运行，错误中保留容器名称。镜像不会自动下载；可用固定 digest 代替 tag 以锁定执行环境。Docker 容器不是虚拟机，完成标记也不提供针对恶意候选代码的评分防篡改保证。

`backend: local` 仅供可信代码的开发测试。它使用独立临时目录、环境变量白名单、输出与超时限制，并清理进程树；Windows 使用 Job Object 限制整体内存和进程数，POSIX 使用进程组与进程资源限制。本地模式不隔离网络或宿主文件系统，也不提供 Docker 的整体 CPU/磁盘限制；POSIX 本地模式没有整体内存/PID 配额。Windows 虚拟环境启动器通常占用两个进程名额。

### 运行状态与退出码

`status.json` 运行时为 `RUNNING`，结束后为 `COMPLETED`、`FAILED`、`ABORTED` 或 `INTERRUPTED`。中断和失败也会保存已完成结果、未执行项的 `CANCELLED` 记录及最终报告。调度队列最多保留 `workers` 个在途任务；认证或执行后端致命错误发生后停止提交新任务。

| CLI 退出码 | 含义 |
| --- | --- |
| 0 | 评测正常完成；模型答错、超时或资源超限仍可属于正常完成的评测 |
| 1 | 配置错误 |
| 2 | 评测未完整完成，或存在 API / 执行环境 / 框架错误 |
| 130 | 用户中断 |

按 Ctrl+C 后停止调度、取消代码执行和重试，尝试关闭正在读取的模型响应，并保留已知用量。尚未返回响应的 HTTP 请求仍可能等待 `timeout_seconds`；强制结束宿主进程或关闭 Docker 服务无法保证清理完成。报告同时展示参与评分的准确率、评分覆盖率、取消数和基础设施错误率，避免把部分结果当成完整成绩。`OUTPUT_LIMIT`、`RESOURCE_LIMIT` 计为已评分失败；`EXECUTION_ERROR`、`CANCELLED` 不计入准确率分母。

### 评分与重试规则

- AIME 接受最后一个 `\boxed{}` 中的 1–3 位非负整数（如 `070`），未使用盒子时只接受最后一行的整数；不对小数取整，也不从表达式中截取数字。
- 选择题对盒内答案做完整匹配，接受 `B`、`(B)`、`Answer: B` 等明确格式；无盒子时采用最后一次明确答案声明或末行独立选项。无效的最后答案不会回退到之前的答案。
- GSM8K 使用精确有限十进制比较，不再使用浮点舍入或容差。优先取最后一个 `####` 后的值，最终标记无效时不回退。没有标记时，仅接受最后一行的完整数值，或完整的 `answer is` / `answer:` / `total is` / `total:` 声明；不会从分数、表达式或推理中的数字片段提取答案。标准答案无效记为框架错误，不计为模型答错。
- Python 代码提取先去除公共缩进，再清理首尾空白，保留函数体补全中的嵌套层级。
- IFEval 在启动工作线程前加载语言检测配置，并固定检测器种子为 0，保证语言评分在多线程和多次运行之间可重复。
- 代码测试程序必须正常执行到末尾；`sys.exit(0)`、`os._exit(0)` 提前退出不算通过。空测试集记为 `INTERNAL_ERROR`，且不发送模型请求。完成检查不构成防篡改沙箱；stdin/stdout 题仍按实际输出判分。
- 每个用例最多调用接口 3 次（含首次调用），关闭 SDK 内层重试。连接/传输错误、HTTP 408/409/429/5xx 和空回答可重试；401/403 立即返回致命错误，其他 HTTP 客户端错误不重试。
- 失败尝试已返回的 token 用量仍计入总量。`results.jsonl` 增加 `api_attempts` 和 `usage_complete`，报告汇总请求次数及用量不完整的用例数；服务端未返回的用量不作估算。
- 流式响应必须包含第 0 个回答的终止 `finish_reason`。仅遇到正常 EOF 或 `[DONE]` 不代表回答完整：此类部分回答会保存，并在同一重试预算内重试，不进入成功生成后的评分流程。明确返回 `length` 则属于模型输出限制，不额外重试；不会拼接其他候选回答。
- 续跑和报告重建会在改写已有结果前拒绝损坏的记录、负数或非整数用量、重复回答 ID、未知检查点状态，以及关联到缺失、不完整或其他题目回答的已完成检查点。

## 开发

### GitHub Actions

`CI` 在 push、PR 和手动触发时运行 Linux/Windows 回归、Ruff、mypy，以及真实 Docker 集成测试。普通 CI 不调用模型 API，Plus 容器测试使用参考答案。

`DeepSeek live evaluation` 仅手动触发，使用 `configs/deepseek.ci.yaml` 和仓库 Actions Secret `DEEPSEEK_API_KEY`：

```bash
gh workflow run deepseek-live.yml --ref main
```

它通过 C++ 服务执行 API60（六个场景各 10 题、10 并发、禁止重试），再测试所选标准题库。工作流的 benchmark_task 和 benchmark_limit 可配置，默认 GSM8K 10 题。模型为 deepseek-flash，启用默认思考强度，输出预算 16384 token。

工作流保留真实评分结果；模型答错不作为 CI 基础设施失败，API、运行及存储错误会导致工作流失败。即使计时步骤失败，也会继续尝试能力测试。报告展示在运行 Summary，原始响应、逐段计时和 JSON/Markdown 报告作为 artifact 保留 14 天。真实测试不会在 PR 上自动运行。

也可以在本地设置 `DEEPSEEK_API_KEY` 后执行：

```bash
uv run llm-eval api-test --config configs/deepseek.ci.yaml
uv run llm-eval benchmark --task gsm --limit 10 --config configs/deepseek.ci.yaml
```

### 本地验证

所有工具均在 `pyproject.toml` 中配置，并通过 `uv` 运行：

```bash
uv run ruff check .    # 代码检查
uv run ruff format .   # 自动格式化
uv run mypy            # 类型检查
uv run pytest          # 运行测试
```

默认测试不请求真实模型接口。真实容器集成测试需准备 Docker 和镜像后显式启用：

```powershell
$env:LLM_EVAL_DOCKER_TESTS = "1"
uv run pytest tests/integration/test_docker_integration.py tests/integration/test_lightweight_suites.py
```

未启用时容器集成测试显示为跳过；模拟 Docker 调用的测试不能替代真实容器验证。

`test_lightweight_suites.py` 覆盖 60 题清单、失败请求不重试、离线重算、取消和恢复。真实 Plus 容器评分由 Docker 与原生 HTTP 集成测试覆盖。

Windows 沙箱的默认临时目录可能过长，导致文件读写报 `FileNotFoundError`。可为测试指定项目内的短目录，例如 `uv run pytest --basetemp .pytest_cache/run-tmp`。该目录仅存放测试临时文件，pytest 会清空它。
