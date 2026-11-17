#include "engine.hpp"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>

namespace eval {
namespace {
bool active(const std::string& status) { return status == "QUEUED" || status == "RUNNING" || status == "CANCELLING"; }
struct TaskInfo { const char* id; const char* label; const char* path; };
constexpr TaskInfo catalog[] = {
    {"humaneval", "HumanEval · 代码生成", "datasets/benchmark/HumanEval.jsonl"},
    {"humanevalplus", "HumanEval+ · 完整扩展测试", "datasets/benchmark/HumanEvalPlus.jsonl"},
    {"mbpp", "MBPP · Python 编程", "datasets/benchmark/MBPP.jsonl"},
    {"mbppplus", "MBPP+ · 完整扩展测试", "datasets/benchmark/MBPPPlus.jsonl"},
    {"gsm", "GSM8K · 数学推理", "datasets/benchmark/GSM8K.jsonl"},
    {"aime2025", "AIME 2025 · 数学竞赛", "datasets/benchmark/AIME2025.jsonl"},
    {"aime2026", "AIME 2026 · 数学竞赛", "datasets/benchmark/AIME2026.jsonl"},
    {"gpqa", "GPQA · 专业知识", "datasets/benchmark/GPQA.jsonl"},
    {"mmlu_pro", "MMLU-Pro · 多学科知识", "datasets/benchmark/MMLUPro.jsonl"},
    {"ifeval", "IFEval · 指令遵循", "datasets/benchmark/IFEval.jsonl"},
    {"livecodebench", "LiveCodeBench · 编程竞赛", "datasets/benchmark/LiveCodeBench.jsonl"}
};
int integer(const Json& source, const char* field, int fallback, int maximum) {
    if (!source.contains(field)) return fallback;
    if (!source[field].is_number_integer() || source[field].is_boolean()) throw std::invalid_argument("Invalid integer parameter");
    const auto value = source[field].get<long long>();
    if (value < 1 || value > maximum) throw std::invalid_argument("参数超出允许范围。");
    return static_cast<int>(value);
}
std::string text(const Json& source, const char* name, std::string fallback, std::size_t maximum) {
    if (!source.contains(name)) return fallback;
    if (!source[name].is_string()) throw std::invalid_argument("Invalid text parameter");
    auto result = source[name].get<std::string>();
    if (result.empty() || result.size() > maximum || result.find('\0') != std::string::npos ||
        result.find('\r') != std::string::npos || result.find('\n') != std::string::npos)
        throw std::invalid_argument("文本参数为空、过长或包含换行。");
    return result;
}
fs::path existing_run(const fs::path& directory) {
    if (!fs::is_directory(directory / "runs")) return {};
    for (const auto& task : fs::directory_iterator(directory / "runs")) {
        if (!task.is_directory()) continue;
        for (const auto& run : fs::directory_iterator(task.path()))
            if (run.is_directory() && fs::is_regular_file(run.path() / "manifest.json")) return run.path();
    }
    return {};
}
}
Engine::Engine(const Config& config): config_(config),
    reasoning_presets_(read_json(config.root / "llm_eval/core/reasoning_presets.json")), graders_(config) {
    fs::create_directories(config.state / "jobs");
    load();
    requests_ = std::make_unique<ThreadPool>(config.requests);
    grades_ = std::make_unique<ThreadPool>(config.graders);
    jobs_pool_ = std::make_unique<ThreadPool>(config.jobs);
}
Engine::~Engine() { shutdown(); }
void Engine::shutdown() {
    {
        std::lock_guard lock(mutex_);
        if (stopping_) return;
        stopping_ = true;
        for (const auto& [id, job] : jobs_) {
            (void)id;
            std::lock_guard job_lock(job->mutex);
            if (active(job->metadata.value("status", ""))) {
                job->cancelled = true;
                job->desired_status = "INTERRUPTED";
                job->stop_reason = "Service stopping";
                if (!job->run.empty()) { try { atomic_write(job->run / ".cancelled", "1"); } catch (...) {} }
            }
        }
    }
    jobs_pool_.reset();
    requests_.reset();
    grades_.reset();
}
void Engine::load() {
    for (const auto& entry : fs::directory_iterator(config_.state / "jobs")) {
        const auto id = entry.path().filename().string();
        if (!entry.is_directory() || !hex_id(id) || !fs::is_regular_file(entry.path() / "job.json")) continue;
        try {
            auto job = std::make_shared<Job>();
            job->metadata = read_json(entry.path() / "job.json");
            job->metadata["id"] = id;
            job->owner = job->metadata.value("owner_id", "admin");
            job->directory = entry.path(); job->run = existing_run(entry.path());
            if (active(job->metadata.value("status", ""))) {
                job->metadata["status"] = "INTERRUPTED";
                job->metadata["error"] = "服务重启，任务已中断；原始记录仍保留。";
                if (!job->run.empty()) {
                    try {
                        const auto final = graders_.call({{"action", "finish"}, {"root", job->run.string()},
                            {"status", "INTERRUPTED"}, {"error", job->metadata["error"]},
                            {"wall_clock_seconds", std::max(0.0, epoch_seconds() - job->metadata.value("started_at", epoch_seconds()))}});
                        job->metadata["summary"] = final.at("summary");
                    } catch (...) { /* Keep interrupted records visible even if their archive needs repair. */ }
                }
                persist(*job);
            }
            jobs_[id] = std::move(job);
        } catch (const std::exception&) { /* Leave corrupt records intact for operator recovery. */ }
    }
}
void Engine::persist(Job& job) {
    job.metadata["owner_id"] = job.owner;
    job.metadata["updated_at"] = epoch_seconds();
    job.cached_body.clear();
    write_json(job.directory / "job.json", job.metadata);
}
Json Engine::public_metadata(const Job& job, bool summary) const {
    Json result = job.metadata;
    result.erase("owner_id"); result.erase("run_root");
    if (!summary) result.erase("summary");
    return result;
}
Json Engine::tasks() const {
    Json result = Json::array();
    for (const auto& task : catalog) {
        bool ready = fs::is_regular_file(config_.root / task.path);
        if (std::string(task.id) == "livecodebench")
            ready = ready && fs::is_directory(config_.root / "datasets/benchmark/LiveCodeBench.private");
        result.push_back({{"id", task.id}, {"label", task.label}, {"available", ready}});
    }
    return result;
}
Json Engine::submit(Json input, std::string owner, std::string key) {
    if (!input.is_object()) throw std::invalid_argument("请求参数必须为对象。");
    Json spec;
    spec["mode"] = text(input, "mode", "benchmark", 16);
    if (spec["mode"] != "benchmark" && spec["mode"] != "api") throw std::invalid_argument("请选择有效评测功能。");
    const auto task = spec["mode"] == "api" ? "api_experience" : text(input, "task", "gsm", 64);
    const TaskInfo* selected = nullptr;
    for (const auto& item : catalog) if (task == item.id) selected = &item;
    if (task != "api_experience" && selected == nullptr) throw std::invalid_argument("请选择有效题库。");
    spec["task"] = task;
    spec["base_url"] = text(input, "base_url", config_.base_url, 2048);
    spec["model"] = text(input, "model", config_.model, 200);
    spec["workers"] = integer(input, "workers", 10, 10);
    spec["max_output_tokens"] = integer(input, "max_output_tokens", 16384, 65536);
    if (input.contains("thinking_enabled") && !input["thinking_enabled"].is_boolean())
        throw std::invalid_argument("思考模式必须为布尔值。");
    spec["thinking_enabled"] = input.value("thinking_enabled", true);
    spec["reasoning_effort"] = nullptr;
    if (input.contains("reasoning_effort") && !input["reasoning_effort"].is_null()) {
        const auto effort = text(input, "reasoning_effort", "", 16);
        auto supported = reasoning_presets_["efforts"];
        auto model = spec["model"].get<std::string>();
        const auto slash = model.find_last_of('/');
        if (slash != std::string::npos) model = model.substr(slash + 1);
        const auto lower = [](std::string value) {
            std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) { return std::tolower(ch); });
            return value;
        };
        for (const auto& preset : reasoning_presets_["models"])
            if (lower(preset["id"].get<std::string>()) == lower(model)) supported = preset["efforts"];
        if (std::find(supported.begin(), supported.end(), effort) == supported.end())
            throw std::invalid_argument("此模型的思考强度无效，请重新选择。");
        spec["reasoning_effort"] = effort;
        spec["thinking_enabled"] = effort != "none";
    }
    spec["limit"] = input.contains("limit") && !input["limit"].is_null() ? Json(integer(input, "limit", 1, 5000)) : Json(nullptr);
    if (task == "api_experience" && !spec["limit"].is_null())
        throw std::invalid_argument("固定题包必须完整运行，不能限制题量。");
    const auto target = endpoint(spec["base_url"], owner != "admin");
    if (owner != "admin") resolve_public(target.host, target.port);
    if (key.empty() || key.size() > 4096 || key.find_first_of("\r\n") != std::string::npos)
        throw std::invalid_argument("请输入有效的模型 API 密钥。");
    std::size_t count = 60;
    std::size_t assets = 0;
    if (selected) {
        const auto dataset = read_file(config_.root / selected->path);
        count = static_cast<std::size_t>(std::count(dataset.begin(), dataset.end(), '\n'));
        assets = dataset.size() * 3;
    }
    if (!spec["limit"].is_null()) count = std::min(count, spec["limit"].get<std::size_t>());
    if (task == "livecodebench") {
        const auto folder = config_.root / "datasets/benchmark/LiveCodeBench.private";
        if (!fs::is_directory(folder)) throw std::invalid_argument("LiveCodeBench 私有题库尚未准备。");
        for (const auto& file : fs::directory_iterator(folder)) if (file.is_regular_file()) assets += file.file_size();
    }
    const std::size_t reservation = count * spec["max_output_tokens"].get<std::size_t>() * 32 + assets;
    std::lock_guard lock(mutex_);
    if (stopping_) throw std::invalid_argument("服务正在停止，请稍后再试。");
    int total = 0, own = 0;
    std::size_t reserved = 0;
    for (const auto& [id, job] : jobs_) {
        (void)id;
        std::lock_guard job_lock(job->mutex);
        if (active(job->metadata.value("status", ""))) {
            ++total; if (job->owner == owner) ++own; reserved += job->reservation;
        }
    }
    if (total >= config_.max_pending) throw std::invalid_argument("服务繁忙，请稍后再提交。");
    if (own >= config_.per_owner) throw std::invalid_argument("你的运行及排队任务已达到上限（2 个）。");
    if (fs::space(config_.state).available < reservation + reserved + 2ULL * 1024 * 1024 * 1024)
        throw std::invalid_argument("可用存储不足，请降低题量或输出预算后重试。");
    auto job = std::make_shared<Job>();
    const auto id = random_hex();
    job->owner = std::move(owner); job->spec = std::move(spec); job->key = std::move(key);
    job->directory = config_.state / "jobs" / id;
    job->run = job->directory / "runs" / task / "native";
    job->reservation = reservation;
    job->metadata = {{"id", id}, {"status", "QUEUED"}, {"created_at", epoch_seconds()},
                     {"mode", job->spec["mode"]}, {"task", task}, {"model", job->spec["model"]},
                     {"workers", job->spec["workers"]}, {"max_output_tokens", job->spec["max_output_tokens"]},
                     {"thinking_enabled", job->spec["thinking_enabled"]}, {"reasoning_effort", job->spec["reasoning_effort"]},
                     {"runtime", "cpp"}, {"error", nullptr}};
    persist(*job); jobs_[id] = job;
    auto response = public_metadata(*job, false);
    jobs_pool_->submit([this, job] { run(job); });
    return response;
}
std::shared_ptr<Job> Engine::owned(const std::string& id, const std::string& owner) {
    std::lock_guard lock(mutex_);
    const auto found = jobs_.find(id);
    if (found == jobs_.end() || found->second->owner != owner) throw std::out_of_range("任务不存在。");
    return found->second;
}
Json Engine::list(const std::string& owner) {
    std::vector<Json> rows;
    std::lock_guard lock(mutex_);
    for (const auto& [id, job] : jobs_) {
        (void)id;
        if (job->owner != owner) continue;
        std::lock_guard job_lock(job->mutex);
        rows.push_back(public_metadata(*job, false));
    }
    std::sort(rows.begin(), rows.end(), [](const Json& a, const Json& b) { return a["created_at"] > b["created_at"]; });
    if (rows.size() > 50) rows.resize(50);
    return rows;
}
std::pair<std::string, std::string> Engine::snapshot(const std::shared_ptr<Job>& job) {
    std::lock_guard lock(job->mutex);
    if (!job->cases_loaded && !job->run.empty()) {
        const auto folder = job->run / "cases";
        if (fs::is_directory(folder)) for (const auto& entry : fs::directory_iterator(folder)) {
            if (entry.path().extension() == ".json") job->cases.push_back(read_json(entry.path()));
        }
        std::sort(job->cases.begin(), job->cases.end(), [](const Json& a, const Json& b) { return a["case_index"] < b["case_index"]; });
        job->cases_loaded = true;
    }
    if (job->cached_body.empty()) {
        auto data = public_metadata(*job, true); data["cases"] = job->cases;
        if (job->metadata.contains("total_cases"))
            data["progress"] = {{"total_cases", job->metadata["total_cases"]}, {"completed_cases", job->cases.size()}};
        job->cached_body = data.dump(); job->etag = '"' + sha256(job->cached_body) + '"';
    }
    return {job->cached_body, job->etag};
}
void Engine::cancel(const std::shared_ptr<Job>& job) {
    std::lock_guard lock(job->mutex);
    if (!active(job->metadata.value("status", ""))) return;
    job->cancelled = true; job->desired_status = "INTERRUPTED"; job->stop_reason = "Evaluation cancelled by owner";
    atomic_write(job->run / ".cancelled", "1");
    if (job->metadata["status"] == "QUEUED") {
        job->metadata["status"] = "INTERRUPTED"; job->metadata["finished_at"] = epoch_seconds(); job->key.clear();
    } else job->metadata["status"] = "CANCELLING";
    job->metadata["error"] = job->stop_reason; persist(*job);
}
fs::path Engine::report(const std::shared_ptr<Job>& job) {
    const auto path = job->run / "report.md";
    if (job->run.empty() || !fs::is_regular_file(path)) throw std::out_of_range("报告尚未生成。");
    return path;
}
fs::path Engine::artifacts(const std::shared_ptr<Job>& job) {
    { std::lock_guard lock(job->mutex);
      if (active(job->metadata.value("status", ""))) throw std::invalid_argument("任务结束后可以下载完整记录。"); }
    if (job->run.empty() || !fs::exists(job->run / "manifest.json")) throw std::out_of_range("任务尚未生成记录。");
    std::lock_guard archive_lock(job->archive_mutex);
    return graders_.call({{"action", "export"}, {"root", job->run.string()}}).at("path").get<std::string>();
}
Json Engine::capacity() {
    int queued = 0;
    std::lock_guard lock(mutex_);
    for (const auto& [id, job] : jobs_) {
        (void)id;
        std::lock_guard job_lock(job->mutex);
        if (job->metadata.value("status", "") == "QUEUED") ++queued;
    }
    return {{"runtime", "cpp"}, {"max_running_jobs", config_.jobs}, {"running_jobs", job_activity_.current.load()},
            {"peak_running_jobs", job_activity_.peak.load()}, {"queued_jobs", queued},
            {"requests", {{"limit", config_.requests}, {"active", request_activity_.current.load()}, {"peak", request_activity_.peak.load()}}},
            {"graders", graders_.activity()}};
}

void Engine::run(const std::shared_ptr<Job>& job) {
    { std::lock_guard lock(job->mutex);
      if (job->cancelled) return;
      job->metadata["status"] = "RUNNING"; job->metadata["started_at"] = epoch_seconds(); persist(*job); }
    ActivityGuard job_guard(job_activity_);
    const auto started = Clock::now();
    bool prepared = false;
    int lock_fd = -1;
    struct Pending { std::size_t index; Json request; std::future<Json> future; bool grading; Clock::time_point started; };
    std::vector<Pending> pending;
    Json requests;
    try {
        fs::create_directories(job->run);
        lock_fd = open((job->run / ".run.lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (lock_fd < 0 || flock(lock_fd, LOCK_EX | LOCK_NB) != 0) throw std::runtime_error("Run is already in use");
        auto preparation = graders_.call({{"action", "prepare"}, {"root", job->run.string()}, {"spec", job->spec}});
        requests = preparation.at("requests"); prepared = true;
        { std::lock_guard lock(job->mutex);
          job->metadata["total_cases"] = requests.size(); job->cases_loaded = true; persist(*job); }
        std::size_t next = 0;
        while (!pending.empty() || (!job->cancelled && next < requests.size())) {
            while (!job->cancelled && next < requests.size() && pending.size() < job->spec["workers"].get<std::size_t>()) {
                const auto request = requests[next]; const auto queued = Clock::now();
                const auto index = next;
                auto future = requests_->submit([this, job, request, queued, index] {
                    ActivityGuard activity(request_activity_);
                    const auto target = endpoint(job->spec["base_url"], job->owner != "admin");
                    return generate(target, job->key, request, job->owner != "admin", 120, job->cancelled,
                                    job->spec["workers"], config_.requests, elapsed(queued),
                                    [job, request, index, queued](const Json& generation) {
                                        write_json(job->run / "native_pending" / (std::to_string(index) + ".json"),
                                            {{"index", index}, {"request", request["body"]}, {"generation", generation},
                                             {"duration_seconds", elapsed(queued)}});
                                    });
                });
                pending.push_back({next++, request, std::move(future), false, queued});
            }
            for (auto item = pending.begin(); item != pending.end();) {
                if (item->future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) { ++item; continue; }
                auto result = item->future.get();
                if (!item->grading) {
                    const auto index = item->index; const auto request = item->request;
                    const auto duration = elapsed(item->started);
                    // Persist before entering a fallible external scoring process.
                    write_json(job->run / "native_pending" / (std::to_string(index) + ".json"),
                               {{"index", index}, {"request", request["body"]}, {"generation", result}, {"duration_seconds", duration}});
                    item->future = grades_->submit([this, job, index, request, result, duration] {
                        return graders_.call({{"action", "grade"}, {"root", job->run.string()}, {"index", index},
                                              {"request", request["body"]}, {"generation", result}, {"duration_seconds", duration}});
                    });
                    item->grading = true; ++item;
                } else {
                    const auto status = result.value("status", "INTERNAL_ERROR");
                    { std::lock_guard lock(job->mutex);
                      job->cases.push_back(result);
                      std::sort(job->cases.begin(), job->cases.end(), [](const Json& a, const Json& b) { return a["case_index"] < b["case_index"]; });
                      if (status == "API_ERROR_FATAL" || status == "EXECUTION_ERROR" || status == "STORAGE_ERROR") {
                          job->cancelled = true; job->desired_status = "ABORTED";
                          job->stop_reason = result["error"].is_string() ? result["error"].get<std::string>() : status;
                          atomic_write(job->run / ".cancelled", "1");
                      }
                      persist(*job); }
                    item = pending.erase(item);
                }
            }
            if (!pending.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    } catch (const std::exception& error) {
        { std::lock_guard lock(job->mutex);
          job->cancelled = true; job->desired_status = "FAILED"; job->stop_reason = redact(error.what(), job->key); }
        try { atomic_write(job->run / ".cancelled", "1"); } catch (...) {}
        // Drain in-flight work before finalizing or clearing the credential.
        for (auto& item : pending) if (item.future.valid()) {
            try {
                const auto result = item.future.get();
                if (!item.grading) write_json(job->run / "native_pending" / (std::to_string(item.index) + ".json"),
                    {{"index", item.index}, {"request", item.request["body"]}, {"generation", result}, {"duration_seconds", elapsed(item.started)}});
            } catch (...) { /* The job remains failed; existing durable records are retained. */ }
        }
    }
    try {
        Json final;
        std::string status, error;
        { std::lock_guard lock(job->mutex); status = job->desired_status; error = job->stop_reason; }
        if (prepared) final = graders_.call({{"action", "finish"}, {"root", job->run.string()}, {"status", status},
                                            {"error", error.empty() ? Json(nullptr) : Json(error)}, {"wall_clock_seconds", elapsed(started)}});
        std::lock_guard lock(job->mutex);
        if (prepared) {
            job->metadata["summary"] = final.at("summary");
            job->metadata["status"] = final["summary"]["run_status"];
            job->metadata["error"] = final["summary"]["error"];
            job->cases = final.at("cases");
        } else { job->metadata["status"] = status; job->metadata["error"] = error; }
        job->metadata["finished_at"] = epoch_seconds(); job->key.clear(); persist(*job);
    } catch (const std::exception& error) {
        std::lock_guard lock(job->mutex);
        job->metadata["status"] = "FAILED"; job->metadata["error"] = redact(error.what(), job->key);
        job->key.clear(); job->metadata["finished_at"] = epoch_seconds();
        try { persist(*job); } catch (...) {}
    }
    if (lock_fd >= 0) close(lock_fd);
}
}
