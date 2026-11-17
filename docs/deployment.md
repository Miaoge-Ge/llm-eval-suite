# 网页部署 / Web deployment

推荐 Ubuntu 24.04、C++20、Python 3.12、Docker、Caddy，开放 80/443。网页免注册，以签名 Cookie 隔离任务；已移除管理入口按钮和登录页面；部署脚本使用的管理员 API 仍受密码保护，用于默认密钥及容量查询。C++ 负责网页、调度与模型流式请求，Python 保留官方评分和代码题沙箱。

## 安装

```bash
sudo apt-get update
sudo apt-get install -y python3-venv docker.io caddy cmake ninja-build g++ pkg-config libcpp-httplib-dev nlohmann-json3-dev libcurl4-openssl-dev libssl-dev
sudo useradd --system --create-home --home-dir /var/lib/llm-eval-suite --shell /usr/sbin/nologin llm-eval
sudo usermod -aG docker llm-eval
sudo git clone https://github.com/Miaoge-Ge/llm-eval-suite.git /opt/llm-eval-suite
cd /opt/llm-eval-suite
uv sync --frozen --python 3.12
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
sudo docker pull python:3.12-slim
sudo docker build -t llm-eval-python:3.12-numpy2.2.6 -f docker/Dockerfile.numpy docker
```

先安装 [uv](https://docs.astral.sh/uv/)，并确保部署账号对检出目录有写权限。始终使用锁定依赖。保留源码检出与 `datasets/benchmark`，不要仅复制 wheel：题库默认路径位于项目目录。

代码题在无网络、非 root、只读根目录、带 CPU/内存/进程/输出限制的 Docker 容器中执行。评分进程不接收服务器密钥、管理员密码或会话签名密钥。容器带服务、评分进程和任务标签；评分进程异常或服务重启时仅清理对应容器。

原生服务只绑定 127.0.0.1，单实例持有状态目录锁。默认 20 个任务、40 个 HTTP 请求、4 个评分进程，每用户最多 2 个任务，全站最多 200 个运行/排队任务。可通过 `LLM_EVAL_MAX_RUNNING_JOBS`、`LLM_EVAL_REQUEST_CAPACITY`、`LLM_EVAL_EXECUTION_CAPACITY` 配置。20 个任务不等于 200 个同时发出的 API 请求：全局请求上限仍为 40，评分并发仍为 4。访客接口限 HTTPS 公网地址，禁止私网、重定向与 DNS 重绑定访问。

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

`DEEPSEEK_API_KEY` 只为默认 `LLM_EVAL_DEFAULT_BASE_URL` 使用。其他 API 地址须在网页单独填写密钥，密钥不写入任务记录。生产环境请使用 HTTPS，Cookie 为 HttpOnly、SameSite=Strict；HTTPS 请求设置 Secure。管理员会话 12 小时，匿名会话 30 天，刷新时续期。IP 只用于限流，不是身份。修改 session secret 会使所有会话失效，匿名历史将无法从浏览器找回。

## HTTPS

把 `deploy/Caddyfile` 中 `{$LLM_EVAL_DOMAIN}` 替换为实际域名后安装到 `/etc/caddy/Caddyfile`。如果 80/443 被旧服务占用，先确认迁移并保留其文件及配置，再停用旧服务。

```bash
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl enable caddy
sudo systemctl restart caddy
curl --fail https://your-domain.example/api/health
```

Caddy 自动申请和续期证书，反向代理到本机 8080。不要把 8080 直接公开。默认首页自动创建匿名工作区；任务、报告、下载和取消都检查服务端 owner。不同浏览器互不可见；同一浏览器的标签页共享身份。清除 Cookie、换设备或使用新隐身窗口会成为新访客。

## 运维与数据

- 服务日志：`journalctl -u llm-eval-web`；证书日志：`journalctl -u caddy`。
- 结果：`/var/lib/llm-eval-suite/jobs/<id>/`。保存 job 元数据、原始 generation、评分、逐段流式时间戳与报告；网页可下载 ZIP。
- 每个任务一份记录；重启时未完成任务标记为中断，保留已有结果。API 测试不得续跑，必须新建；CLI 的 benchmark 可以按兼容性检查续跑或离线评分。
- 停止任务会取消后续调度、通过 libcurl 回调中断在途请求，并通知代码沙箱停止。已完成的请求先落盘再评分；评分异常时恢复请求账目，不自动补发。
- 定期备份结果及环境文件，注意磁盘容量。LiveCodeBench 私有资产约 818 MiB，未安装时网页标记题库待准备；下载方法见 [参考](reference_CN.md)。
- 更新源码/依赖后重启 `llm-eval-web`；有任务运行时先等它结束。保留旧版本源码与环境，可切回后重启。
- 服务器测量与 GitHub runner 测量的网络路径不同，不应混成同一轮 API 质量统计。

网页目前提供配置、启动、进度、取消、历史、报告和下载；离线重评分及 benchmark 断点续跑使用 CLI。模型验证仍是规划页。

## 无域名时使用正式 IP 证书

域名受到云服务商拦截时，可为公网 IP 申请 Let's Encrypt `shortlived` 证书。它是受信任的正式证书，寿命约六天，必须配置自动续期。已在 Ubuntu 24.04、Caddy 2.6.2、Certbot 5.10.0 验证。

先用 Caddy 在 80 端口为 `/.well-known/acme-challenge/*` 提供 `/var/lib/llm-eval-acme` 下的文件，其余路径暂时返回 503。不要在 HTTP 页面提交访问密码。创建此目录后安装并申请证书（将示例 IP 替换为实际地址）：

```bash
sudo python3 -m venv /opt/llm-eval-certbot
sudo /opt/llm-eval-certbot/bin/pip install certbot==5.10.0
sudo install -d -m 0755 /var/lib/llm-eval-acme
sudo /opt/llm-eval-certbot/bin/certbot certonly --webroot -w /var/lib/llm-eval-acme \
  --ip-address 203.0.113.10 --cert-name llm-eval-ip --preferred-profile shortlived
```

根据 Certbot 提示完成账户设置。证书成功后：

1. 执行 `sudo env RENEWED_LINEAGE=/etc/letsencrypt/live/llm-eval-ip sh deploy/certbot-deploy-hook.sh`，把证书复制到 `/etc/caddy/certs/llm-eval-ip/`，文件权限为 `0640 root:caddy`。此时 reload 仍使用初始 HTTP 配置。
2. 替换 `deploy/Caddyfile.ip` 的 IP 占位符，安装到 `/etc/caddy/Caddyfile`，验证并 reload Caddy。`default_sni` 让不发送 SNI 的 IP 客户端选到正确证书。
3. 把 deploy hook 安装到 `/etc/letsencrypt/renewal-hooks/deploy/llm-eval-caddy`，权限 `0750 root:root`。续期成功时复制新证书并 reload Caddy；Ubuntu 的 Caddy unit 用 `--force` 重新加载证书。
4. 把 `deploy/llm-eval-cert-renew.service` 和 `.timer` 安装到 `/etc/systemd/system/`，执行 `systemctl daemon-reload`、`systemctl enable --now llm-eval-cert-renew.timer`。
5. 执行 `certbot renew --cert-name llm-eval-ip --dry-run` 验证续期链路；这里的 certbot 是 `/opt/llm-eval-certbot/bin/certbot`。定时器每天检查两次。

使用 `curl --fail https://实际IP/api/health` 验证受信任 HTTPS，不使用 `-k` 或跳过证书校验。HTTP 保留 ACME 验证路径，其余请求跳转到 HTTPS。
