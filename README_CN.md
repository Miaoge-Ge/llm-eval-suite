# LLM Eval Suite

[English](README.md) · [CLI 完整参考](docs/reference_CN.md) · [网页部署](docs/deployment.md)

通过命令行或网页，对 OpenAI 兼容接口进行模型能力评测和 API 服务质量测试。每轮保存原始回答、有效配置、计时与评分依据。

| 功能 | 状态 | 用途 |
| --- | --- | --- |
| Benchmark 测试 | 已实现 | 代码、数学、知识、指令遵循；完整题库与固定 Quick40 |
| API 测试 | 已实现 | 首试成功率、流式延迟、中断、截断、可观测吞吐 |
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
uv run llm-eval benchmark --task quick40 --config configs/model.yaml --workers 10
uv run llm-eval api --config configs/model.yaml --workers 10
```

PowerShell 使用 `Copy-Item` 复制文件，用 `$env:OPENAI_API_KEY = '你的密钥'` 设置环境变量。真实密钥不要写入提交到 Git 的配置。兼容原有 `run`、`api-test`、`quick-test` 命令。

完整题库包括 HumanEval/Plus、MBPP/Plus、GSM8K、AIME 2025/2026、GPQA、MMLU-Pro、IFEval、LiveCodeBench。LiveCodeBench 私有测试需要额外下载，见 [CLI 参考](docs/reference_CN.md)。

## 网页工作台

```bash
export LLM_EVAL_ADMIN_PASSWORD='设置足够长的私有访问密码'
export LLM_EVAL_SESSION_SECRET="$(python -c 'import secrets; print(secrets.token_hex(32))')"
# 可选：设置 DEEPSEEK_API_KEY，供默认 DeepSeek 地址使用。
uv run --extra web llm-eval-web
```

访问 `http://127.0.0.1:8080`，登录后可配置测试、查看逐题进度、停止任务、打开历史报告、下载原始记录。同一时间执行一个任务，每个任务最多 10 并发，运行及排队任务最多五个。模型验证页标注“开发中”。自定义密钥仅驻留内存；服务器默认密钥放在受保护的环境文件中。公网部署使用 HTTPS，详见 [部署说明](docs/deployment.md)。

## 评测规则

- **Quick40**：四个维度各 10 题，保留原题、来源 ID、内容哈希和选题理由。不能用 `--limit` 缩减固定题包；选中 Plus 题目的扩展测试完整执行。
- **API**：短回答、长回答、长上下文各 10 次流式请求。每题请求一次，不自动重试，不支持续跑。并发设置存入原始记录，离线重算保留原测量并发。
- **计时**：首段有效内容可为思考内容，首段正文单独记录；保留逐段时间戳。仅在 token 用量和正文时间窗口可靠时估算 TPS，否则显示缺失。传输成功与截断分别报告。
- **评分**：接口错误与答错分开；准确率同时报告评分覆盖率和错误数。只有思考内容、因 token 上限结束的回答保留截断标记，不再重复请求。
- **恢复**：Benchmark 支持续跑；两个固定题包均支持无 API 调用的离线重评分，结果另存，原始运行保留。

```bash
uv run llm-eval grade --run-dir results/<model>/<task>/<run>
uv run llm-eval report --run-dir results/<model>/<task>/<run>
uv run llm-eval resume --run-dir results/<model>/<task>/<run> --config configs/model.yaml
```

## 按功能划分目录

```text
llm_eval/
  benchmark/          # 能力任务、评分、Quick40、报告
  api_testing/        # API 固定题包、流式指标、质量报告
  model_verification/ # 规划说明，尚未实现
  core/               # 共用客户端、调度、执行、存储
  web/                # FastAPI 服务与网页界面
  cli.py              # 命令入口
  tasks.py            # 公开任务注册表
datasets/benchmark/   # 原始 benchmark 数据
scripts/benchmark/    # 导入题库、构建固定题包
scripts/api_testing/  # 构建 API 题包
tests/                # 按功能分类，另含集成与网页测试
configs/              # 无密钥示例、DeepSeek CI 配置
deploy/               # systemd、Caddy 模板
docs/                 # CLI、部署参考
```

## GitHub CI 与真实测试

普通 CI 包括 Linux/Windows 回归、Ruff、mypy、真实 Docker 测试，不调用模型。手动运行 **DeepSeek live evaluation** 工作流时，从 Secret `DEEPSEEK_API_KEY` 读取密钥，先跑 30 次 API 测试，再跑全部 40 道能力题，两者均为 **10 workers**。

DeepSeek 配置显式启用思考，思考与正文共享 **16,384 token** 输出预算。首轮实测因预算过小，在思考阶段耗尽额度，失败记录保留在 Actions 中。增加预算不改变原题及要求回答的长度；横向对比需要使用同一设置。计时反映运行机器的网络条件。

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
