#include "grader.hpp"
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cstring>
#include <sstream>

namespace eval {
namespace {
std::string docker_command(std::vector<std::string> args) {
    int output[2];
    if (pipe2(output, O_CLOEXEC) != 0) return {};
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, output[1], STDOUT_FILENO);
    posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addclose(&actions, output[0]);
    posix_spawn_file_actions_addclose(&actions, output[1]);
    std::vector<char*> argv;
    args.insert(args.begin(), {"timeout", "15", "docker"});
    for (auto& argument : args) argv.push_back(argument.data());
    argv.push_back(nullptr);
    char path[] = "PATH=/usr/bin:/bin", lang[] = "LANG=C.UTF-8";
    char* environment[] = {path, lang, nullptr};
    pid_t child;
    const int error = posix_spawnp(&child, "timeout", &actions, nullptr, argv.data(), environment);
    posix_spawn_file_actions_destroy(&actions);
    close(output[1]);
    std::string result;
    if (!error) {
        char buffer[4096]; ssize_t count;
        while ((count = read(output[0], buffer, sizeof(buffer))) > 0)
            if (result.size() < 1024 * 1024) result.append(buffer, count);
        waitpid(child, nullptr, 0);
    }
    close(output[0]);
    return result;
}
}
void cleanup_containers(const std::string& label) {
    std::istringstream ids(docker_command({"ps", "-aq", "--filter", "label=" + label}));
    std::string id;
    std::vector<std::string> arguments{"rm", "-f"};
    while (ids >> id) {
        if (id.size() >= 12 && id.size() <= 64 && id.find_first_not_of("0123456789abcdef") == std::string::npos)
            arguments.push_back(id);
    }
    if (arguments.size() > 2) docker_command(std::move(arguments));
}
GraderProcess::GraderProcess(const Config& config): config_(config) { start(); }
GraderProcess::~GraderProcess() { stop(); }
void GraderProcess::start() {
    container_label_ = random_hex();
    int to_child[2], from_child[2];
    if (pipe2(to_child, O_CLOEXEC) != 0) throw std::runtime_error("Cannot open grader input pipe");
    if (pipe2(from_child, O_CLOEXEC) != 0) {
        close(to_child[0]); close(to_child[1]); throw std::runtime_error("Cannot open grader output pipe");
    }
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, to_child[0], STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&actions, from_child[1], STDOUT_FILENO);
    for (int fd : {to_child[0], to_child[1], from_child[0], from_child[1]})
        posix_spawn_file_actions_addclose(&actions, fd);
    std::vector<std::string> values = {
        "PATH=" + env("PATH", "/usr/bin:/bin"), "PYTHONPATH=" + config_.root.string(),
        "PYTHONUNBUFFERED=1", "PYTHONIOENCODING=utf-8", "LANG=C.UTF-8",
        "LLM_EVAL_STATE_DIR=" + config_.state.string(),
        "LLM_EVAL_SERVICE_ID=" + sha256(config_.state.string()),
        "LLM_EVAL_GRADER_ID=" + container_label_
    };
    std::vector<char*> environment;
    for (auto& value : values) environment.push_back(value.data());
    environment.push_back(nullptr);
    std::string python = config_.python.string();
    char module[] = "llm_eval.native_bridge", module_option[] = "-m";
    char* arguments[] = {python.data(), module_option, module, nullptr};
    const int error = posix_spawn(&pid_, python.c_str(), &actions, nullptr, arguments, environment.data());
    posix_spawn_file_actions_destroy(&actions);
    close(to_child[0]); close(from_child[1]);
    if (error != 0) {
        close(to_child[1]); close(from_child[0]); pid_ = -1;
        throw std::runtime_error("Cannot start official grader: " + std::string(std::strerror(error)));
    }
    input_ = to_child[1]; output_ = from_child[0]; pending_.clear();
}
void GraderProcess::stop() {
    if (input_ >= 0) close(input_);
    if (output_ >= 0) close(output_);
    input_ = output_ = -1;
    if (pid_ > 0) {
        kill(pid_, SIGTERM);
        for (int i = 0; i < 20; ++i) {
            if (waitpid(pid_, nullptr, WNOHANG) == pid_) { pid_ = -1; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (pid_ > 0) { kill(pid_, SIGKILL); waitpid(pid_, nullptr, 0); pid_ = -1; }
    }
    if (!container_label_.empty()) cleanup_containers("llm-eval.grader=" + container_label_);
    container_label_.clear();
}
Json GraderProcess::call(const Json& request) {
    if (pid_ < 0) start();
    try {
        const std::string line = request.dump() + "\n";
        std::size_t offset = 0;
        while (offset < line.size()) {
            const auto count = write(input_, line.data() + offset, line.size() - offset);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) throw std::runtime_error("Official grader input disconnected");
            offset += static_cast<std::size_t>(count);
        }
        const auto started = Clock::now();
        while (true) {
            const auto newline = pending_.find('\n');
            if (newline != std::string::npos) {
                const auto response = Json::parse(pending_.substr(0, newline));
                pending_.erase(0, newline + 1);
                if (!response.value("ok", false)) throw std::runtime_error(response.value("error", "Grading failed"));
                return response.at("result");
            }
            if (pending_.size() > 128 * 1024 * 1024 || elapsed(started) > 1800)
                throw std::runtime_error("Official grader exceeded response limit");
            pollfd descriptor{output_, POLLIN, 0};
            const int ready = poll(&descriptor, 1, 1000);
            if (ready < 0 && errno == EINTR) continue;
            if (ready < 0) throw std::runtime_error("Cannot monitor official grader");
            if (ready == 0) continue;
            char buffer[65536];
            const auto count = read(output_, buffer, sizeof(buffer));
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) throw std::runtime_error("Official grader exited before returning a result");
            pending_.append(buffer, static_cast<std::size_t>(count));
        }
    } catch (...) { stop(); throw; }
}
GraderPool::GraderPool(const Config& config) {
    for (int i = 0; i < config.graders; ++i) {
        workers_.push_back(std::make_unique<GraderProcess>(config)); available_.push_back(i);
    }
}
Json GraderPool::call(const Json& request) {
    std::size_t index;
    {
        std::unique_lock lock(mutex_);
        condition_.wait(lock, [this] { return !available_.empty(); });
        index = available_.front(); available_.pop_front();
    }
    try {
        Json result;
        {
            ActivityGuard guard(activity_);
            result = workers_[index]->call(request);
        }
        { std::lock_guard lock(mutex_); available_.push_back(index); }
        condition_.notify_one();
        return result;
    } catch (...) {
        { std::lock_guard lock(mutex_); available_.push_back(index); }
        condition_.notify_one(); throw;
    }
}
Json GraderPool::activity() const {
    return {{"limit", workers_.size()}, {"active", activity_.current.load()}, {"peak", activity_.peak.load()}};
}
}
