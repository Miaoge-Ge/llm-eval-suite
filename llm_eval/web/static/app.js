"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  mode: "benchmark",
  config: null,
  jobs: [],
  selected: null,
  timer: null,
  busy: false,
};
const names = {
  QUEUED: "排队中",
  RUNNING: "运行中",
  CANCELLING: "正在停止",
  COMPLETED: "已完成",
  FAILED: "有错误",
  ABORTED: "已中止",
  INTERRUPTED: "已中断",
  PASSED: "通过",
  API_ERROR: "接口错误",
  API_ERROR_FATAL: "认证错误",
  CANCELLED: "未执行",
  TIMEOUT: "执行超时",
  RESOURCE_LIMIT: "资源限制",
  OUTPUT_LIMIT: "输出限制",
  EXECUTION_ERROR: "执行错误",
  STORAGE_ERROR: "存储错误",
  INTERNAL_ERROR: "内部错误",
  GENERATED: "已生成",
};
const running = new Set(["QUEUED", "RUNNING", "CANCELLING"]);
const graded = new Set([
  "PASSED",
  "FAILED",
  "TIMEOUT",
  "RESOURCE_LIMIT",
  "OUTPUT_LIMIT",
]);

function element(tag, text, className = "") {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  el.className = className;
  return el;
}
function showLogin() {
  $("workspace").hidden = true;
  $("login-view").hidden = false;
  $("api-key").value = "";
  clearTimeout(state.timer);
}
async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, {
    credentials: "same-origin",
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    if (response.status === 401 && path !== "/login") showLogin();
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : `请求失败 (${response.status})`,
    );
  }
  return response;
}
function badge(status) {
  return `badge ${running.has(status) ? "running" : status === "COMPLETED" ? "completed" : "error"}`;
}
function caseClass(status) {
  return status === "PASSED"
    ? "passed"
    : status === "FAILED"
      ? "failed"
      : !status || status === "CANCELLED"
        ? "pending"
        : "error";
}
function number(value, digits = 2) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits)
    : "—";
}
function metric(label, value, note) {
  const item = element("div", undefined, "metric");
  item.append(
    element("span", label),
    element("strong", value),
    element("small", note),
  );
  return item;
}
function configureTask() {
  const fixed = $("task").value === "quick40";
  $("limit-field").hidden = fixed;
  $("suite-help").textContent = fixed
    ? "四个维度各 10 题，完整保留代码测试。"
    : "使用题库原题和完整评分逻辑，可设置题量上限。";
  if (fixed) $("limit").value = "";
}
function setMode(mode) {
  state.mode = mode;
  const titles = {
    benchmark: [
      "模型能力",
      "Benchmark 评测",
      "选择题库并运行。完整评分与原始响应会一起保存。",
    ],
    api: [
      "服务质量",
      "API 测试",
      "从首次响应到完整回答，记录每一次请求的真实表现。",
    ],
    verification: [
      "模型身份",
      "模型验证",
      "以可信参考为依据，对比模型行为。此功能尚在规划中。",
    ],
  };
  const [caption, title, description] = titles[mode];
  $("section-caption").textContent = caption;
  $("page-title").textContent = title;
  $("page-description").textContent = description;
  document
    .querySelectorAll("[data-mode]")
    .forEach((button) =>
      button.classList.toggle("active", button.dataset.mode === mode),
    );
  $("verification-view").hidden = mode !== "verification";
  $("evaluation-view").hidden = mode === "verification";
  $("benchmark-fields").hidden = mode !== "benchmark";
  $("api-suite-note").hidden = mode !== "api";
  $("start-button").firstChild.textContent =
    mode === "api" ? "开始 API 测试 " : "开始 Benchmark 评测 ";
  $("form-error").textContent = "";
  state.selected = state.jobs.find((job) => job.mode === mode)?.id || null;
  renderHistory();
  if (state.selected) loadJob(state.selected);
  else {
    $("empty-results").hidden = false;
    $("job-results").hidden = true;
    $("job-badge").textContent = "等待开始";
    $("job-badge").className = "badge";
  }
}
function renderHistory() {
  const jobs = state.jobs.filter((job) => job.mode === state.mode);
  $("history").replaceChildren();
  $("empty-history").hidden = jobs.length !== 0;
  for (const job of jobs) {
    const row = element("button", undefined, "history-row");
    const title = element("span");
    title.append(
      element("strong", job.model),
      element(
        "small",
        `${job.workers} 并发 / ${job.max_output_tokens} token 预算`,
      ),
    );
    const status = element(
      "span",
      names[job.status] || job.status,
      badge(job.status),
    );
    row.append(
      title,
      element("span", job.task, "history-task"),
      element(
        "time",
        new Date(job.created_at * 1000).toLocaleString("zh-CN", {
          month: "2-digit",
          day: "2-digit",
          hour: "2-digit",
          minute: "2-digit",
        }),
      ),
      status,
    );
    row.addEventListener("click", () => {
      state.selected = job.id;
      loadJob(job.id);
      $("results-title").scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
    });
    $("history").append(row);
  }
}
function renderJob(job) {
  $("empty-results").hidden = true;
  $("job-results").hidden = false;
  $("job-badge").textContent = names[job.status] || job.status;
  $("job-badge").className = badge(job.status);
  $("job-model").textContent = job.model;
  $("job-meta").textContent =
    `${job.task} / ${job.workers} 并发 / ${job.max_output_tokens} token 预算`;
  $("cancel-button").hidden =
    !running.has(job.status) || job.status === "CANCELLING";
  const rows = job.cases || [];
  const summary = job.summary;
  const total =
    summary?.total_cases ??
    job.progress?.total_cases ??
    (job.task === "quick40" ? 40 : job.mode === "api" ? 30 : 0);
  const completed = rows.filter((row) => row.status !== "CANCELLED").length;
  $("progress-text").textContent =
    job.status === "QUEUED"
      ? "任务已排队，前一个任务结束后自动开始"
      : running.has(job.status)
        ? "正在收集回答与评分"
        : "本轮已结束，原始记录已保留";
  $("progress-count").textContent = total
    ? `${completed} / ${total}`
    : "准备题库";
  $("progress-fill").style.width =
    `${total ? Math.min(100, (completed / total) * 100) : 0}%`;
  $("case-grid").replaceChildren();
  const byIndex = new Map(rows.map((row) => [row.case_index, row]));
  for (let i = 0; i < Math.min(total, 100); i++) {
    const row = byIndex.get(i);
    const dot = element("span", String(i + 1), caseClass(row?.status));
    dot.title = row
      ? `${row.case_id}: ${row.status === "FAILED" ? "答错" : names[row.status] || row.status}`
      : `第 ${i + 1} 题：等待`;
    $("case-grid").append(dot);
  }
  $("job-error").textContent = job.error || "";
  $("metrics").replaceChildren();
  if (job.mode === "api") {
    const overall = summary?.metrics?.overall;
    const first = overall?.observed?.first_body_seconds;
    const tps = overall?.observed?.output_tps_estimate;
    const success = overall?.first_attempt_success_rate;
    $("metrics").append(
      metric(
        "首试成功率",
        success == null ? "—" : `${number(success * 100, 1)}%`,
        `${overall?.attempted ?? completed} 次尝试，禁止重试`,
      ),
      metric(
        "首段正文 P50",
        first?.p50 == null ? "—" : `${number(first.p50)} s`,
        `${first?.n ?? 0} 个有效观测`,
      ),
      metric(
        "输出 TPS P50",
        number(tps?.p50, 1),
        `估算值 / 截断 ${overall?.truncated ?? 0} 次`,
      ),
    );
  } else {
    const passed = rows.filter((row) => row.status === "PASSED").length;
    const scored = rows.filter((row) => graded.has(row.status)).length;
    $("metrics").append(
      metric(
        "通过 / 计划",
        `${passed} / ${total || "—"}`,
        "错误与未执行项单独保留",
      ),
      metric(
        "已评分准确率",
        scored ? `${number((passed / scored) * 100, 1)}%` : "—",
        `${scored} 题完成有效评分`,
      ),
      metric(
        "运行耗时",
        summary ? `${number(summary.wall_clock_seconds, 1)} s` : "—",
        summary
          ? `${summary.token_usage.total_tokens.toLocaleString()} tokens`
          : "完成后汇总",
      ),
    );
  }
  const finished = !running.has(job.status);
  $("show-report").hidden = !finished || !summary;
  $("download").hidden = !job.progress && !summary;
  $("download").href = `/api/jobs/${job.id}/artifacts`;
  $("case-rows").replaceChildren();
  $("cases-section").hidden = rows.length === 0;
  for (const row of rows) {
    const tr = element("tr");
    const status =
      row.status === "FAILED" ? "答错" : names[row.status] || row.status;
    let detail =
      row.error ||
      (row.status === "PASSED"
        ? "通过"
        : `参考 ${row.expected ?? "—"} / 回答 ${row.actual ?? "—"}`);
    if (row.metadata?.truncated || row.metadata?.experience?.truncated)
      detail = `输出被截断。${detail}`;
    tr.append(
      element("td", row.case_id),
      element("td", status),
      element("td", detail.slice(0, 220)),
    );
    $("case-rows").append(tr);
  }
}
async function loadJob(id) {
  try {
    const job = await (await api(`/jobs/${id}`)).json();
    if (state.selected === id) renderJob(job);
  } catch (error) {
    $("job-error").textContent = error.message;
  }
}
async function refresh() {
  clearTimeout(state.timer);
  if ($("workspace").hidden) return;
  try {
    state.jobs = await (await api("/jobs")).json();
    renderHistory();
    if (state.selected) await loadJob(state.selected);
    else if (state.jobs.some((job) => job.mode === state.mode)) {
      state.selected = state.jobs.find((job) => job.mode === state.mode).id;
      await loadJob(state.selected);
    }
  } catch (error) {
    if (!$("workspace").hidden)
      $("form-error").textContent = `连接暂时中断：${error.message}`;
  }
  if (!$("workspace").hidden)
    state.timer = setTimeout(
      refresh,
      state.jobs.some((job) => running.has(job.status)) ? 2000 : 8000,
    );
}
async function initialize() {
  state.config = await (await api("/config")).json();
  $("base-url").value = state.config.base_url;
  $("model").value = state.config.model;
  $("task").replaceChildren();
  for (const task of state.config.tasks) {
    const option = element(
      "option",
      task.label + (task.available ? "" : "（题库待准备）"),
    );
    option.value = task.id;
    option.disabled = !task.available;
    $("task").append(option);
  }
  if (state.config.server_key_available) {
    $("key-optional").textContent = "可选";
    $("api-key").placeholder = "留空使用服务器密钥";
    $("key-help").textContent =
      "服务器密钥仅用于默认 API 地址；自定义密钥不写入评测记录。";
  }
  $("login-view").hidden = true;
  $("workspace").hidden = false;
  configureTask();
  setMode(state.mode);
  await refresh();
}
$("login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("login-error").textContent = "";
  const button = event.currentTarget.querySelector("button");
  button.disabled = true;
  try {
    await api("/login", {
      method: "POST",
      body: JSON.stringify({ password: $("login-password").value }),
    });
    $("login-password").value = "";
    await initialize();
  } catch (error) {
    $("login-error").textContent = error.message;
  } finally {
    button.disabled = false;
  }
});
$("evaluation-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.busy) return;
  state.busy = true;
  $("start-button").disabled = true;
  $("form-error").textContent = "";
  const payload = {
    mode: state.mode,
    task: $("task").value,
    base_url: $("base-url").value.trim(),
    model: $("model").value.trim(),
    api_key: $("api-key").value,
    workers: Number($("workers").value),
    max_output_tokens: Number($("token-budget").value),
    thinking_enabled: $("thinking").checked,
    limit:
      state.mode === "benchmark" &&
      $("task").value !== "quick40" &&
      $("limit").value
        ? Number($("limit").value)
        : null,
  };
  try {
    const job = await (
      await api("/jobs", { method: "POST", body: JSON.stringify(payload) })
    ).json();
    state.selected = job.id;
    $("api-key").value = "";
    await refresh();
  } catch (error) {
    $("form-error").textContent = error.message;
  } finally {
    state.busy = false;
    $("start-button").disabled = false;
  }
});
$("cancel-button").addEventListener("click", async () => {
  if (!state.selected) return;
  try {
    await api(`/jobs/${state.selected}/cancel`, { method: "POST" });
    await refresh();
  } catch (error) {
    $("job-error").textContent = error.message;
  }
});
$("show-report").addEventListener("click", async () => {
  try {
    $("report-content").textContent = await (
      await api(`/jobs/${state.selected}/report`)
    ).text();
    $("report-dialog").showModal();
  } catch (error) {
    $("job-error").textContent = error.message;
  }
});
$("close-report").addEventListener("click", () => $("report-dialog").close());
$("logout").addEventListener("click", async () => {
  try {
    await api("/logout", { method: "POST" });
  } finally {
    showLogin();
  }
});
$("refresh-history").addEventListener("click", refresh);
$("task").addEventListener("change", configureTask);
document
  .querySelectorAll("[data-mode]")
  .forEach((button) =>
    button.addEventListener("click", () => setMode(button.dataset.mode)),
  );
for (let i = 0; i < 40; i++)
  document.querySelector(".sample-grid").append(element("i"));
initialize().catch(() => showLogin());
