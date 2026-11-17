#pragma once

#include <nlohmann/json.hpp>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <functional>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace eval {
using Json = nlohmann::json;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

double epoch_seconds();
double elapsed(Clock::time_point start);
std::string env(const char* name, std::string fallback = "");
int env_int(const char* name, int fallback, int maximum = 200);
std::string random_hex(std::size_t bytes = 16);
std::string sha256(const std::string& value);
std::string hmac_sha256(const std::string& secret, const std::string& value);
bool constant_equal(const std::string& left, const std::string& right);
bool hex_id(const std::string& value);
std::string read_file(const fs::path& path, std::size_t maximum = 128 * 1024 * 1024);
Json read_json(const fs::path& path);
void atomic_write(const fs::path& path, const std::string& value);
void write_json(const fs::path& path, const Json& value);
std::string redact(std::string value, const std::string& secret);

struct Config {
    fs::path root;
    fs::path state;
    fs::path python;
    std::string password;
    std::string session_secret;
    std::string default_key;
    std::string base_url = "https://api.deepseek.com/v1";
    std::string model = "deepseek-flash";
    int port = 8080;
    int jobs = 20;
    int requests = 40;
    int graders = 4;
    int max_pending = 200;
    int per_owner = 2;
    static Config from_env();
};

class ThreadPool {
public:
    explicit ThreadPool(int size);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    template<class Function> auto submit(Function function) -> std::future<std::invoke_result_t<Function>> {
        using Result = std::invoke_result_t<Function>;
        auto task = std::make_shared<std::packaged_task<Result()>>(std::move(function));
        auto future = task->get_future();
        {
            std::lock_guard lock(mutex_);
            if (stopping_) throw std::runtime_error("Worker pool is stopping");
            queue_.emplace_back([task] { (*task)(); });
        }
        condition_.notify_one();
        return future;
    }
private:
    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<std::function<void()>> queue_;
    std::vector<std::thread> threads_;
    bool stopping_ = false;
};

struct Activity {
    std::atomic<int> current{0};
    std::atomic<int> peak{0};
    void enter();
    void leave();
};

class ActivityGuard {
public:
    explicit ActivityGuard(Activity& activity): activity_(activity) { activity_.enter(); }
    ~ActivityGuard() { activity_.leave(); }
private:
    Activity& activity_;
};
}
