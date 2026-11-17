#pragma once
#include "common.hpp"
#include <sys/types.h>

namespace eval {
void cleanup_containers(const std::string& label);
class GraderProcess {
public:
    explicit GraderProcess(const Config& config);
    ~GraderProcess();
    Json call(const Json& request);
private:
    void start();
    void stop();
    const Config& config_;
    pid_t pid_ = -1;
    int input_ = -1;
    int output_ = -1;
    std::string pending_;
    std::string container_label_;
};

class GraderPool {
public:
    explicit GraderPool(const Config& config);
    Json call(const Json& request);
    Json activity() const;
private:
    std::vector<std::unique_ptr<GraderProcess>> workers_;
    std::deque<std::size_t> available_;
    std::mutex mutex_;
    std::condition_variable condition_;
    Activity activity_;
};
}
