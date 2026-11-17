# LLM Eval Suite

[English](README.md) · [CLI 完整参考](docs/reference_CN.md) · [网页部署](docs/deployment.md) · [CI](https://github.com/Miaoge-Ge/llm-eval-suite/actions/workflows/ci.yml)

通过命令行或网页，对 OpenAI 兼容接口进行模型能力评测和 API 服务质量测试。每轮保存原始回答、有效配置、计时与评分依据。

| 功能 | 状态 | 用途 |
| --- | --- | --- |
| Benchmark 测试 | 已实现 | 代码、数学、知识、指令遵循；标准题库，可设置题量 |
| API 测试 | 已实现 | 60 次固定请求，覆盖服务质量、流式延迟、输出检查与 tokens 消耗 |
| 模型验证 | 待开发 | 未来与可信参考模型比较；当前不生成模型真伪结论 |

## 快速开始

需要 Python 3.10+、[uv](https://docs.astral.sh/uv/)，代码题需要 Docker Linux 容器环境。

```bash
git clone https://github.com/Miaoge-Ge/llm-eval-suite.git
cd llm-eval-suite
uv sync --frozen --extra web
cp configs/model.example.yaml configs/model.yaml
# 设置 OPENAI_API_KEY，并修改配置中的模型名和接口地址。
docker pull python:3.12-slim
docker build -t llm-eval-python:3.12-numpy2.2.6 -f docker/Dockerfile.numpy docker
uv run llm-eval benchmark --task gsm --limit 10 --config configs/model.yaml --workers 10
uv run llm-eval api --config configs/model.yaml --workers 10
```

PowerShell 使用 `Copy-Item` 复制文件，用 `$env:OPENAI_API_KEY = '你的密钥'` 设置环境变量。真实密钥不要写入提交到 Git 的配置。兼容原有 `run`、`api-test` 命令。

完整题库包括 HumanEval/Plus、MBPP/Plus、GSM8K、AIME 2025/2026、GPQA、MMLU-Pro、IFEval、LiveCodeBench。LiveCodeBench 私有测试需要额外下载，见 [CLI 参考](docs/reference_CN.md)。

## Benchmark 题库与题量

以下为仓库当前内置文件的题量，不代表所有上游题库的完整规模。网页下拉列表显示服务器实际安装的题量，选择题库或修改题量上限时，会同步显示本次计划测试数。

| 题库 | CLI 任务名 | 内置题量 |
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

题量上限留空时运行当前题库全部题目；上限超过题库规模时，仅运行已有题目。这里统计的是题目数，选中的 Plus 题目仍完整执行题内扩展测试。LiveCodeBench 还需准备私有测试资产才能使用。Quick40 题包及 `quick-test` 命令已移除。

## 网页工作台

[原生架构与匿名隔离](docs/native-service.md) · [并发性能与迁移验证](docs/validation-native-2026-11-17.md)

生产网页使用 Linux C++20 服务：网页 HTTP、匿名会话、任务调度、模型 HTTP/SSE 请求均由 C++ 执行。既有 Python 官方评分器和代码题沙箱保留，保证评分规则不变。构建依赖和 HTTPS 配置见 [部署说明](docs/deployment.md)。

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
# 用于受保护的运维 API，网页已无管理员登录页。
export LLM_EVAL_ADMIN_PASSWORD='设置足够长的私有访问密码'
export LLM_EVAL_SESSION_SECRET="$(python -c 'import secrets; print(secrets.token_hex(32))')"
# 可选：DEEPSEEK_API_KEY 仅供已认证的运维 API 客户端使用。
uv run --extra web llm-eval-web
```

访问 `http://127.0.0.1:8080` 即可免注册、免登录进入，填写自己的模型 API 密钥即可评测。管理入口按钮和登录页面已删除；连接异常时提供“重新连接”。服务端签名的随机 Cookie 标识本浏览器，30 天有效，每次查询、下载、取消都校验任务归属。IP 只用于限流，同一 IP 下的不同浏览器不会共享记录。同一浏览器的多个标签页共享工作区；清 Cookie 或换浏览器后无法找回旧匿名任务。

默认最多 20 个运行任务、40 个模型请求、4 个评分进程、200 个运行及排队任务；每位访客最多 2 个运行及排队任务，每任务最多 10 workers。资源共享时会排队，调度等待与 API 延迟分别记录。4 核 / 4 GB 主机不能保证 100 个 CPU 密集评测同时独占资源。密钥仅在内存中使用，不传入评分进程。公网部署使用 HTTPS。

网页支持按模型联动选择思考强度，并把 `reasoning_effort` 原样发送给 OpenAI 兼容中转；`none` 请求关闭思考。预设来自指定模型清单，不能证明中转已支持或执行该参数。详见[API60 与思考强度](docs/api-testing.md)。

## API60 与 tokens 消耗

| 场景 | 请求数 | 观测与检查 |
| --- | ---: | --- |
| 短回答 | 10 | 首段延迟与请求完成情况 |
| 长回答 | 10 | 持续输出、停顿与吞吐 |
| 长上下文 | 10 | 约 4K 参考输入 tokens 下的响应 |
| JSON 输出 | 10 | JSON 模式支持、对象结构与字段值 |
| 多轮上下文 | 10 | 对修正后上下文的记忆 |
| 中英文混合 | 10 | 中文、英文、重音字符及 emoji 精确回显 |

各场景交错执行，失败请求原样保留，不自动重试或降级。输出检查独立于传输成功率。网页按场景显示成功数、首段正文 P50/P95、截断数和输出检查结果。

网页结果和导出报告展示**输入、输出、总 tokens 消耗**。网页随响应返回更新，包含失败或截断请求中已知的用量。输出按上游口径包含思考 tokens，不重复加算思考或缓存 tokens。用量缺失不会补成零，部分统计会显示缺失请求数；仅在输入、输出都已知时补算总和。这些是观测到的 tokens 数量，不是费用估算。

## 评测规则

- **Benchmark**：直接选择标准题库；Plus 题目的扩展测试完整执行。
- **API60**：短回答、长回答、长上下文、JSON 输出、多轮上下文、中英文混合各 10 次。每题只请求一次，不重试、不续跑；输出检查与流式成功率分开统计。
- **计时**：首段有效内容可为思考内容，首段正文单独记录；保留逐段时间戳。仅在 token 用量和正文时间窗口可靠时估算 TPS，否则显示缺失。传输成功与截断分别报告。
- **评分**：接口错误与答错分开；准确率同时报告评分覆盖率和错误数。只有思考内容、因 token 上限结束的回答保留截断标记，不再重复请求。
- **恢复**：Benchmark 支持续跑；Benchmark 和 API 测试均支持无 API 调用的离线重评分，结果另存，原始运行保留。

```bash
uv run llm-eval grade --run-dir results/<model>/<task>/<run>
uv run llm-eval report --run-dir results/<model>/<task>/<run>
uv run llm-eval resume --run-dir results/<model>/<task>/<run> --config configs/model.yaml
```

## 按功能划分目录

```text
llm_eval/
  benchmark/          # 能力任务、评分、报告
  api_testing/        # API 固定题包、流式指标、质量报告
  model_verification/ # 规划说明，尚未实现
  core/               # 共用客户端、调度、执行、存储
  web/                # 网页、原生服务启动器；保留旧服务作基线
  native_bridge.py    # 官方评分与归档 RPC，不接收模型密钥
  cli.py              # 命令入口
  tasks.py            # 公开任务注册表
server/               # C++20 HTTP、会话、调度、流式请求、评分池
datasets/benchmark/   # 原始 benchmark 数据
scripts/benchmark/    # 导入标准题库
scripts/api_testing/  # 构建 API 题包
scripts/native/       # 经 C++ 服务进行真实模型验证
tests/                # 按功能分类，另含集成与网页测试
configs/              # 无密钥示例、DeepSeek CI 配置
deploy/               # systemd、Caddy 模板
docs/                 # CLI、部署参考
```

## GitHub CI 与真实测试

[历史迁移验证](docs/validation-native-2026-11-17.md)记录了旧版本实测；当前 API60 规则与思考强度见[配置说明](docs/api-testing.md)。

普通 CI 包括 Linux/Windows 回归、网页交互逻辑、Ruff、mypy、真实 Docker、C++ Release 构建及原生 HTTP 权限/失败统计测试，不调用模型。手动运行 **DeepSeek live evaluation** 工作流时，从 Secret `DEEPSEEK_API_KEY` 读取密钥，通过 C++ 生产服务先跑 60 次 API 测试，再跑所选标准题库（默认 GSM8K 10 题），两者均为 **10 workers**。

DeepSeek 配置显式启用思考，思考与正文共享 **16,384 token** 输出预算。横向对比应使用相同题包、并发数、思考设置和输出预算；计时反映运行机器的网络条件。

[2026-11-17 的真实测试](https://github.com/Miaoge-Ge/llm-eval-suite/actions/runs/44046356695)完成 **60/60 次 API 请求**、**30/30 项输出检查**，无截断、无额外尝试。用量为**输入 44,782 + 输出 39,432 = 总计 84,214 tokens**。这仅是 `deepseek-flash` 的单轮观测，不代表所有思考强度预设已实测，也不代表长期可用率。

```bash
uv sync --frozen --extra dev --extra web
uv run ruff check .
uv run ruff format --check .
uv run mypy llm_eval
uv run pytest
# 准备 Docker 镜像后；PowerShell 先设置 $env:LLM_EVAL_DOCKER_TESTS='1'
LLM_EVAL_DOCKER_TESTS=1 uv run pytest tests/integration
```

Benchmark 分数、模型名称和自我介绍不能证明模型身份。未来模型验证功能将提供对比证据及不确定性，不以单次问答宣称确定真伪。
