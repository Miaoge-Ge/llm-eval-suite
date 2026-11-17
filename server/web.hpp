#pragma once
#include "engine.hpp"
#include <httplib.h>

namespace eval {
std::string session_token(const std::string& secret, const std::string& owner, int lifetime);
std::string session_identity(const std::string& secret, const std::string& token);
void configure_web(httplib::Server& server, Engine& engine, const Config& config);
}
