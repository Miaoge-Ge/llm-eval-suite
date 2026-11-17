#include "common.hpp"
#include <openssl/crypto.h>
#include <openssl/hmac.h>
#include <openssl/rand.h>
#include <openssl/sha.h>
#include <cstdlib>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>

namespace eval {
namespace {
std::string hex(const unsigned char* bytes, std::size_t size) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result(size * 2, '0');
    for (std::size_t i = 0; i < size; ++i) {
        result[2 * i] = digits[bytes[i] >> 4];
        result[2 * i + 1] = digits[bytes[i] & 15];
    }
    return result;
}
}
double epoch_seconds() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}
double elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
std::string env(const char* name, std::string fallback) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::move(fallback);
}
int env_int(const char* name, int fallback, int maximum) {
    const std::string text = env(name);
    if (text.empty()) return fallback;
    std::size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != text.size() || value < 1 || value > maximum) throw std::runtime_error(std::string("Invalid ") + name);
    return value;
}
std::string random_hex(std::size_t bytes) {
    std::vector<unsigned char> data(bytes);
    if (RAND_bytes(data.data(), static_cast<int>(bytes)) != 1) throw std::runtime_error("Secure random generation failed");
    return hex(data.data(), bytes);
}
std::string sha256(const std::string& value) {
    unsigned char output[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(value.data()), value.size(), output);
    return hex(output, sizeof(output));
}
std::string hmac_sha256(const std::string& secret, const std::string& value) {
    unsigned char output[EVP_MAX_MD_SIZE];
    unsigned int length = 0;
    if (!HMAC(EVP_sha256(), secret.data(), static_cast<int>(secret.size()),
              reinterpret_cast<const unsigned char*>(value.data()), value.size(), output, &length))
        throw std::runtime_error("Session signing failed");
    return hex(output, length);
}
bool constant_equal(const std::string& left, const std::string& right) {
    return left.size() == right.size() && CRYPTO_memcmp(left.data(), right.data(), left.size()) == 0;
}
bool hex_id(const std::string& value) {
    return value.size() == 32 && value.find_first_not_of("0123456789abcdef") == std::string::npos;
}
std::string read_file(const fs::path& path, std::size_t maximum) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read " + path.string());
    const auto size = fs::file_size(path);
    if (size > maximum) throw std::runtime_error("Saved file exceeds supported size");
    std::string result(size, '\0');
    input.read(result.data(), static_cast<std::streamsize>(size));
    if (!input && size) throw std::runtime_error("Cannot read complete saved file");
    return result;
}
Json read_json(const fs::path& path) { return Json::parse(read_file(path)); }
void atomic_write(const fs::path& path, const std::string& value) {
    fs::create_directories(path.parent_path());
    const fs::path temporary = path.parent_path() / ("." + random_hex() + ".tmp");
    int fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) throw std::runtime_error("Cannot create result file");
    try {
        std::size_t offset = 0;
        while (offset < value.size()) {
            const auto written = ::write(fd, value.data() + offset, value.size() - offset);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) throw std::runtime_error("Cannot persist result file");
            offset += static_cast<std::size_t>(written);
        }
        if (::fsync(fd) != 0) throw std::runtime_error("Cannot flush result file");
        ::close(fd);
        fd = -1;
        fs::rename(temporary, path);
    } catch (...) {
        if (fd >= 0) ::close(fd);
        std::error_code ignored;
        fs::remove(temporary, ignored);
        throw;
    }
}
void write_json(const fs::path& path, const Json& value) { atomic_write(path, value.dump() + "\n"); }
std::string redact(std::string value, const std::string& secret) {
    if (secret.empty()) return value;
    std::size_t position = 0;
    while ((position = value.find(secret, position)) != std::string::npos) {
        value.replace(position, secret.size(), "[REDACTED]"); position += 10;
    }
    return value;
}
Config Config::from_env() {
    Config result;
    result.root = fs::absolute(env("LLM_EVAL_ROOT", fs::current_path().string()));
    result.state = fs::absolute(env("LLM_EVAL_STATE_DIR", (result.root / "results/web").string()));
    result.python = env("LLM_EVAL_PYTHON", (result.root / ".venv/bin/python").string());
    result.password = env("LLM_EVAL_ADMIN_PASSWORD");
    result.session_secret = env("LLM_EVAL_SESSION_SECRET");
    result.default_key = env("DEEPSEEK_API_KEY");
    result.base_url = env("LLM_EVAL_DEFAULT_BASE_URL", result.base_url);
    result.model = env("LLM_EVAL_DEFAULT_MODEL", result.model);
    result.port = env_int("LLM_EVAL_PORT", 8080, 65535);
    result.jobs = env_int("LLM_EVAL_MAX_RUNNING_JOBS", 20, 100);
    result.requests = env_int("LLM_EVAL_REQUEST_CAPACITY", 40);
    result.graders = env_int("LLM_EVAL_EXECUTION_CAPACITY", 4, 16);
    if (result.password.size() < 12 || result.session_secret.size() < 32)
        throw std::runtime_error("Set an administrator password (12+) and session secret (32+)");
    if (!fs::exists(result.python) || !fs::exists(result.root / "llm_eval/native_bridge.py"))
        throw std::runtime_error("Python grader runtime or source tree is missing");
    return result;
}
ThreadPool::ThreadPool(int size) {
    for (int i = 0; i < size; ++i) threads_.emplace_back([this] {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock lock(mutex_);
                condition_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
                if (stopping_ && queue_.empty()) return;
                task = std::move(queue_.front()); queue_.pop_front();
            }
            task();
        }
    });
}
ThreadPool::~ThreadPool() {
    { std::lock_guard lock(mutex_); stopping_ = true; }
    condition_.notify_all();
    for (auto& thread : threads_) if (thread.joinable()) thread.join();
}
void Activity::enter() {
    const int active = ++current;
    int previous = peak.load();
    while (active > previous && !peak.compare_exchange_weak(previous, active)) {}
}
void Activity::leave() { --current; }
}
