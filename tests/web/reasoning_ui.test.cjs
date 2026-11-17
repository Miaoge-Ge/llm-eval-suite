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
