#pragma once
#include "common.hpp"

namespace eval {
struct Endpoint {
    std::string url;
    std::string host;
    std::string scheme;
    int port;
};
Endpoint endpoint(const std::string& base_url, bool public_only);
bool public_ip(const std::string& address);
std::vector<std::string> resolve_public(const std::string& host, int port);

class StreamDecoder {
public:
    explicit StreamDecoder(bool capture): capture_(capture), start_(Clock::now()) {}
    void feed(std::string_view data);
    void finish();
    Json record() const;
    std::string content;
    std::string reasoning;
    Json usage = Json::object();
    Json metadata = Json::object();
private:
    void line(std::string value);
    void event();
    bool capture_;
    Clock::time_point start_;
    std::string pending_;
    std::string data_;
    Json events_ = Json::array();
};

Json generate(const Endpoint& target, const std::string& key, const Json& request,
              bool public_only, int timeout_seconds, std::atomic<bool>& cancelled,
              int concurrency, int shared_limit, double scheduler_wait,
              const std::function<void(const Json&)>& journal = {});
}
