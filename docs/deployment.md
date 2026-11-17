# 网页部署 / Web deployment

推荐 Ubuntu 24.04、Python 3.12、Docker、Caddy，域名 A 记录指向服务器并开放 80/443。网页是单管理员工作台，不是公开匿名测试服务。CLI 和网页调用同一套评分及计时逻辑。

## 安装

```bash
sudo apt-get update
sudo apt-get install -y python3-venv docker.io caddy
sudo useradd --system --create-home --home-dir /var/lib/llm-eval-suite --shell /usr/sbin/nologin llm-eval
sudo usermod -aG docker llm-eval
sudo git clone https://github.com/Miaoge-Ge/llm-eval-suite.git /opt/llm-eval-suite
cd /opt/llm-eval-suite
sudo python3 -m venv .venv
sudo .venv/bin/pip install -e '.[web]'
sudo docker pull python:3.12-slim
sudo docker build -t llm-eval-python:3.12-numpy2.2.6 -f docker/Dockerfile.numpy docker
```

需要复现锁定依赖时，用 `uv sync --frozen --extra web --python 3.12` 创建环境。保留源码检出与 `datasets/benchmark`，不要仅复制 wheel：题库默认路径位于项目目录。

代码题在无网络、非 root、只读根目录、带 CPU/内存/进程/输出限制的 Docker 容器中执行。工作台服务拥有 Docker 权限，相当于宿主机管理权限，应仅允许可信管理员登录。默认串行调度评测任务，每轮最多 10 个请求 worker。不要启动多个 Uvicorn worker；队列和认证限速为单进程状态。

## 密钥与服务

复制 `deploy/environment.example` 到 `/etc/llm-eval-suite.env`，分别生成管理员密码（至少 12 字符）和会话密钥（至少 32 字符），填写默认 DeepSeek key（可选）。设置 `root:root` 和 `0600`，不提交该文件。

```bash
sudo install -m 0600 deploy/environment.example /etc/llm-eval-suite.env
sudoedit /etc/llm-eval-suite.env
sudo install -m 0644 deploy/llm-eval-web.service /etc/systemd/system/llm-eval-web.service
sudo systemctl daemon-reload
sudo systemctl enable --now llm-eval-web
curl --fail http://127.0.0.1:8080/api/health
```

`DEEPSEEK_API_KEY` 只为默认 `LLM_EVAL_DEFAULT_BASE_URL` 使用。其他 API 地址须在网页单独填写密钥，密钥不写入任务记录。生产环境请使用 HTTPS，Cookie 为 HttpOnly、SameSite=Strict；HTTPS 请求设置 Secure。会话有效期 12 小时；修改 session secret 并重启可使已有会话失效。

## HTTPS

把 `deploy/Caddyfile` 中 `{$LLM_EVAL_DOMAIN}` 替换为实际域名后安装到 `/etc/caddy/Caddyfile`。如果 80/443 被旧服务占用，先确认迁移并保留其文件及配置，再停用旧服务。

```bash
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl enable caddy
sudo systemctl restart caddy
curl --fail https://your-domain.example/api/health
```

Caddy 自动申请和续期证书，反向代理到本机 8080。不要把 8080 直接公开。默认首页是登录页，任务、报告、原始记录均需登录。

## 运维与数据

- 服务日志：`journalctl -u llm-eval-web`；证书日志：`journalctl -u caddy`。
- 结果：`/var/lib/llm-eval-suite/jobs/<id>/`。保存 job 元数据、原始 generation、评分、逐段流式时间戳与报告；网页可下载 ZIP。
- 每个任务一份记录；重启时未完成任务标记为中断，保留已有结果。API 测试不得续跑，必须新建；CLI 的 benchmark 可以按兼容性检查续跑或离线评分。
- 停止任务会停止调度新请求，已在途请求可能持续到完成或超时。
- 定期备份结果及环境文件，注意磁盘容量。LiveCodeBench 私有资产约 818 MiB，未安装时网页标记题库待准备；下载方法见 [参考](reference_CN.md)。
- 更新源码/依赖后重启 `llm-eval-web`；有任务运行时先等它结束。保留旧版本源码与环境，可切回后重启。
- 服务器测量与 GitHub runner 测量的网络路径不同，不应混成同一轮 API 质量统计。

网页目前提供配置、启动、进度、取消、历史、报告和下载；离线重评分及 benchmark 断点续跑使用 CLI。模型验证仍是规划页。
