#pragma once
#include "common.hpp"
#include "grader.hpp"
#include "model_client.hpp"
#include <map>

namespace eval {
struct Job {
    mutable std::mutex mutex;
    Json metadata;
    Json spec;
    Json cases = Json::array();
    std::string owner;
    std::string key;
    fs::path directory;
    fs::path run;
    std::atomic<bool> cancelled{false};
    std::size_t reservation = 0;
    std::string desired_status = "COMPLETED";
    std::string stop_reason;
    std::string cached_body;
    std::string etag;
    bool cases_loaded = false;
    std::mutex archive_mutex;
};

class Engine {
public:
    explicit Engine(const Config& config);
    ~Engine();
    Json submit(Json spec, std::string owner, std::string key);
    Json list(const std::string& owner);
    std::shared_ptr<Job> owned(const std::string& id, const std::string& owner);
    std::pair<std::string, std::string> snapshot(const std::shared_ptr<Job>& job);
    void cancel(const std::shared_ptr<Job>& job);
    fs::path report(const std::shared_ptr<Job>& job);
    fs::path artifacts(const std::shared_ptr<Job>& job);
    Json capacity();
    Json tasks() const;
    const Json& reasoning_presets() const { return reasoning_presets_; }
    void shutdown();
private:
    void run(const std::shared_ptr<Job>& job);
    void persist(Job& job);
    Json public_metadata(const Job& job, bool summary) const;
    void load();
    const Config& config_;
    Json reasoning_presets_;
    GraderPool graders_;
    std::unique_ptr<ThreadPool> requests_;
    std::unique_ptr<ThreadPool> grades_;
    std::unique_ptr<ThreadPool> jobs_pool_;
    std::mutex mutex_;
    std::map<std::string, std::shared_ptr<Job>> jobs_;
    Activity request_activity_;
    Activity job_activity_;
    bool stopping_ = false;
};
}
