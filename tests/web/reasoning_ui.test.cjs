const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const root = path.resolve(__dirname, "../..");
const presets = JSON.parse(fs.readFileSync(path.join(root, "llm_eval/core/reasoning_presets.json")));

function harness() {
  const nodes = new Map();
  function node() {
    return {
      value: "", children: [], listeners: {}, hidden: false,
      append(...items) { this.children.push(...items); },
      replaceChildren(...items) { this.children = items; this.value = ""; },
      addEventListener(name, callback) { this.listeners[name] = callback; },
    };
  }
  const get = (id) => {
    if (!nodes.has(id)) nodes.set(id, node());
    return nodes.get(id);
  };
  const context = vm.createContext({
    document: { getElementById: get, createElement: node, querySelectorAll: () => [], querySelector: get },
    setTimeout, clearTimeout, performance, console,
  });
  const source = fs.readFileSync(path.join(root, "llm_eval/web/static/app.js"), "utf8")
    .replace("initialize().catch(() => showLogin());", "");
  vm.runInContext(source, context);
  context.presets = presets;
  vm.runInContext("state.config = {reasoning_presets: presets};", context);
  return { context, get, run: (code) => vm.runInContext(code, context) };
}

test("model input exposes each configured level, including namespaced models", () => {
  const h = harness();
  for (const model of presets.models) {
    h.get("model").value = "gateway/" + model.id.toUpperCase();
    h.get("model").listeners.input();
    assert.deepEqual(h.get("reasoning-effort").children.map((o) => o.value), ["", ...model.efforts]);
  }
});

test("model changes preserve valid levels and explain invalid selection resets", () => {
  const h = harness();
  h.get("model").value = "deepseek-v4-pro";
  h.run("configureReasoning()");
  h.get("reasoning-effort").value = "high";
  h.get("model").value = "glm-5.3-flash";
  h.run("configureReasoning()");
  assert.equal(h.get("reasoning-effort").value, "high");
  h.get("reasoning-effort").value = "max";
  h.get("model").value = "qwen3.8-flash";
  h.run("configureReasoning()");
  assert.equal(h.get("reasoning-effort").value, "");
  assert.match(h.get("reasoning-help").textContent, /原档位不可用/);
});

test("form posts none and explicit strengths without losing the selection", async () => {
  const h = harness();
  let submitted;
  h.context.fetch = async (url, options) => {
    assert.equal(url, "/api/jobs");
    submitted = JSON.parse(options.body);
    return { ok: true, status: 202, json: async () => ({ id: "job" }) };
  };
  h.run("state.mode = 'api'; refresh = async () => {};");
  h.get("model").value = "custom";
  h.get("workers").value = "10";
  h.get("token-budget").value = "16384";
  for (const effort of ["", ...presets.efforts]) {
    h.get("reasoning-effort").value = effort;
    await h.get("evaluation-form").listeners.submit({ preventDefault() {} });
    assert.equal(submitted.reasoning_effort, effort || null);
    assert.equal(submitted.thinking_enabled, effort !== "none");
    assert.equal(submitted.limit, null);
  }
});

test("dataset choice and limit update planned counts without changing full Plus coverage", () => {
  const h = harness();
  h.run("state.config.tasks = [{id: 'plus', label: 'HumanEval+', total_cases: 164, available: true}, {id: 'gsm', label: 'GSM8K', total_cases: 1319, available: true}, {id: 'missing', label: 'Missing', total_cases: null, available: false}]");
  h.get("task").value = "plus";
  h.get("task").listeners.change();
  assert.match(h.get("suite-help").textContent, /共 164 题；本次计划 164 题/);
  assert.match(h.run("taskLabel(state.config.tasks[0])"), /164 题/);
  h.get("limit").value = "10";
  h.get("limit").listeners.input();
  assert.match(h.get("suite-help").textContent, /本次计划 10 题/);
  h.get("task").value = "gsm";
  h.get("task").listeners.change();
  assert.match(h.get("suite-help").textContent, /1,319 题；本次计划 10 题/);
  h.get("limit").value = "5000";
  h.get("limit").listeners.input();
  assert.match(h.get("suite-help").textContent, /本次计划 1,319 题/);
  h.get("limit").value = "-1";
  h.get("limit").listeners.input();
  assert.match(h.get("suite-help").textContent, /请输入/);
  h.get("task").value = "missing";
  h.get("task").listeners.change();
  assert.match(h.get("suite-help").textContent, /暂不可用/);
});

test("API token cards show observed totals and missing coverage for saved and running jobs", () => {
  const h = harness();
  h.context.rows = [
    {metadata: {experience: {attempted: true, prompt_tokens: 10, completion_tokens: 20}}},
    {metadata: {experience: {attempted: true, prompt_tokens: 0}}},
    {metadata: {experience: {attempted: false}}},
  ];
  h.run("renderTokenUsage(rows, null, 'RUNNING')");
  let cards = h.get("metrics").children;
  assert.deepEqual(cards.map((card) => card.children[1].textContent), ["10", "20", "30"]);
  assert.match(cards[2].children[2].textContent, /1 次缺失（部分统计）/);
  assert.match(cards[2].children[2].textContent, /随已返回结果更新/);
  h.get("metrics").replaceChildren();
  h.run("renderTokenUsage([{metadata:{experience:{attempted:true}}}], null, 'FAILED')");
  assert.deepEqual(h.get("metrics").children.map((card) => card.children[1].textContent), ["—", "—", "—"]);
  h.get("metrics").replaceChildren();
  h.run("renderTokenUsage([], {metrics:{overall:{token_usage:tokenUsage(rows)}}}, 'COMPLETED')");
  cards = h.get("metrics").children;
  assert.equal(cards[2].children[1].textContent, "30");
  assert.match(cards[1].children[2].textContent, /思考 tokens/);
});
