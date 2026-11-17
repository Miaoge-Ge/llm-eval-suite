#include "web.hpp"
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace eval {
namespace {
struct HttpError : std::runtime_error {
    int status;
    HttpError(int code, const std::string& message): std::runtime_error(message), status(code) {}
};
std::vector<std::string> split(const std::string& text, char delimiter) {
    std::vector<std::string> result;
    std::stringstream input(text); std::string part;
    while (std::getline(input, part, delimiter)) result.push_back(part);
    return result;
}
std::string cookie(const httplib::Request& request, const std::string& name) {
    for (auto part : split(request.get_header_value("Cookie"), ';')) {
        const auto first = part.find_first_not_of(" \t");
        if (first != std::string::npos) part.erase(0, first);
        if (part.starts_with(name + '=')) return part.substr(name.size() + 1);
    }
    return {};
}
bool secure(const httplib::Request& request) { return request.get_header_value("X-Forwarded-Proto") == "https"; }
void set_cookie(httplib::Response& response, const httplib::Request& request, const std::string& name,
                const std::string& value, int lifetime) {
    response.headers.emplace("Set-Cookie", name + '=' + value + "; Path=/; Max-Age=" + std::to_string(lifetime) +
                             "; HttpOnly; SameSite=Strict" + (secure(request) ? "; Secure" : ""));
}
std::string owner(const httplib::Request& request, const Config& config) {
    if (session_identity(config.session_secret, cookie(request, "llm_eval_session")) == "admin") return "admin";
    const auto guest = session_identity(config.session_secret, cookie(request, "llm_eval_guest"));
    if (hex_id(guest)) return guest;
    throw HttpError(401, "会话已失效，请刷新网页重新进入。");
}
void json_response(httplib::Response& response, const Json& body, int status = 200) {
    response.status = status; response.set_content(body.dump(), "application/json; charset=utf-8");
}
Json input(const httplib::Request& request) {
    auto value = Json::parse(request.body, nullptr, false);
    if (value.is_discarded() || !value.is_object()) throw HttpError(422, "提交参数无效。");
    return value;
}
std::string api_key(const Json& body) {
    if (!body.contains("api_key")) return {};
    if (!body["api_key"].is_string()) throw HttpError(422, "API 密钥格式无效。");
    return body["api_key"].get<std::string>();
}
std::string trim_slash(std::string value) { while (!value.empty() && value.back() == '/') value.pop_back(); return value; }
using Handler = std::function<void(const httplib::Request&, httplib::Response&)>;
Handler checked(Handler action) {
    return [action = std::move(action)](const auto& request, auto& response) {
        try { action(request, response); }
        catch (const HttpError& error) { json_response(response, {{"detail", error.what()}}, error.status); }
        catch (const Json::exception&) { json_response(response, {{"detail", "提交参数类型无效。"}}, 422); }
        catch (const std::out_of_range&) { json_response(response, {{"detail", "任务或记录不存在。"}}, 404); }
        catch (const std::invalid_argument& error) { json_response(response, {{"detail", error.what()}}, 422); }
        catch (const std::exception&) { json_response(response, {{"detail", "操作未完成，请稍后重试。"}}, 503); }
    };
}
std::string address(const httplib::Request& request) {
    const auto forwarded = request.get_header_value("X-Forwarded-For");
    if (request.remote_addr == "127.0.0.1" && !forwarded.empty()) return forwarded.substr(0, forwarded.find(','));
    return request.remote_addr;
}
struct Limits {
    std::mutex mutex;
    std::unordered_map<std::string, std::deque<double>> failures;
    std::unordered_map<std::string, std::deque<double>> submissions;
    bool allow(std::unordered_map<std::string, std::deque<double>>& entries, const std::string& key,
               int maximum, double window, bool consume) {
        const double now = epoch_seconds();
        auto& history = entries[key];
        while (!history.empty() && now - history.front() > window) history.pop_front();
        if (history.size() >= static_cast<std::size_t>(maximum)) return false;
        if (consume) history.push_back(now);
        return true;
    }
    void prune() {
        for (auto* entries : {&failures, &submissions}) {
            if (entries->size() > 4096) {
                for (auto item = entries->begin(); item != entries->end();) {
                    if (item->second.empty() || epoch_seconds() - item->second.back() > 600) item = entries->erase(item);
                    else ++item;
                }
                while (entries->size() > 4096) entries->erase(entries->begin());
            }
        }
    }
};
}
std::string session_token(const std::string& secret, const std::string& identity, int lifetime) {
    const auto payload = std::to_string(static_cast<long long>(epoch_seconds()) + lifetime) + '.' + identity + '.' + random_hex();
    return payload + '.' + hmac_sha256(secret, payload);
}
std::string session_identity(const std::string& secret, const std::string& token) {
    if (token.size() > 256) return {};
    const auto parts = split(token, '.');
    if (parts.size() != 4 || parts[0].empty() || parts[0].find_first_not_of("0123456789") != std::string::npos ||
        (parts[1] != "admin" && !hex_id(parts[1])) || !hex_id(parts[2])) return {};
    try {
        if (std::stoll(parts[0]) <= epoch_seconds()) return {};
        const auto payload = parts[0] + '.' + parts[1] + '.' + parts[2];
        if (!constant_equal(parts[3], hmac_sha256(secret, payload))) return {};
        return parts[1];
    } catch (...) { return {}; }
}
void configure_web(httplib::Server& server, Engine& engine, const Config& config) {
    auto limits = std::make_shared<Limits>();
    server.new_task_queue = [] { return new httplib::ThreadPool(32); };
    server.set_payload_max_length(64 * 1024);
    server.set_read_timeout(15);
    server.set_write_timeout(30);
    server.set_keep_alive_max_count(100);
    server.set_pre_routing_handler([](const auto& request, auto& response) {
        if (request.method == "POST" || request.method == "PUT" || request.method == "DELETE" || request.method == "PATCH") {
            const auto origin = request.get_header_value("Origin");
            const auto expected = std::string(secure(request) ? "https://" : "http://") + request.get_header_value("Host");
            if ((!origin.empty() && origin != expected) || request.get_header_value("Sec-Fetch-Site") == "cross-site") {
                json_response(response, {{"detail", "不允许跨站操作。"}}, 403);
                return httplib::Server::HandlerResponse::Handled;
            }
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    server.set_post_routing_handler([](const auto&, auto& response) {
        response.set_header("Cache-Control", "no-store");
        response.set_header("X-Content-Type-Options", "nosniff");
        response.set_header("Referrer-Policy", "same-origin");
        response.set_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'");
    });
    server.Get("/api/health", checked([](const auto&, auto& response) { json_response(response, {{"status", "ok"}, {"runtime", "cpp"}}); }));
    server.Post("/api/session", checked([&config](const auto& request, auto& response) {
        auto identity = session_identity(config.session_secret, cookie(request, "llm_eval_guest"));
        if (!hex_id(identity)) identity = random_hex();
        set_cookie(response, request, "llm_eval_guest", session_token(config.session_secret, identity, 30 * 86400), 30 * 86400);
        json_response(response, {{"ok", true}});
    }));
    server.Post("/api/login", checked([&config, limits](const auto& request, auto& response) {
        const auto body = input(request);
        if (!body.contains("password") || !body["password"].is_string()) throw HttpError(422, "密码格式无效。");
        const auto ip = address(request);
        std::lock_guard lock(limits->mutex); limits->prune();
        if (!limits->allow(limits->failures, ip, 8, 600, false)) throw HttpError(429, "登录尝试过多，请十分钟后重试。");
        if (!constant_equal(sha256(body["password"].template get<std::string>()), sha256(config.password))) {
            limits->allow(limits->failures, ip, 8, 600, true); throw HttpError(401, "登录密码不正确。");
        }
        limits->failures.erase(ip);
        set_cookie(response, request, "llm_eval_session", session_token(config.session_secret, "admin", 43200), 43200);
        json_response(response, {{"ok", true}});
    }));
    server.Post("/api/logout", checked([&config](const auto& request, auto& response) {
        owner(request, config); set_cookie(response, request, "llm_eval_session", "", 0); json_response(response, {{"ok", true}});
    }));
    server.Get("/api/config", checked([&engine, &config](const auto& request, auto& response) {
        const auto identity = owner(request, config);
        json_response(response, {{"tasks", engine.tasks()}, {"base_url", config.base_url}, {"model", config.model},
                                  {"role", identity == "admin" ? "admin" : "guest"}, {"runtime", "cpp"},
                                  {"server_key_available", identity == "admin" && !config.default_key.empty()},
                                  {"max_workers", 10}, {"model_verification", "planned"}});
    }));
    server.Get("/api/admin/capacity", checked([&engine, &config](const auto& request, auto& response) {
        if (owner(request, config) != "admin") throw HttpError(404, "页面不存在。");
        json_response(response, engine.capacity());
    }));
    server.Get("/api/jobs", checked([&engine, &config](const auto& request, auto& response) {
        json_response(response, engine.list(owner(request, config)));
    }));
    server.Post("/api/jobs", checked([&engine, &config, limits](const auto& request, auto& response) {
        const auto identity = owner(request, config);
        auto body = input(request); auto key = api_key(body); body.erase("api_key");
        if (identity == "admin" && key.empty() && trim_slash(body.value("base_url", config.base_url)) == trim_slash(config.base_url))
            key = config.default_key;
        { std::lock_guard lock(limits->mutex); limits->prune();
          if (!limits->allow(limits->submissions, address(request), 120, 60, true)) throw HttpError(429, "提交过于频繁，请稍后重试。"); }
        json_response(response, engine.submit(std::move(body), identity, std::move(key)), 202);
    }));
    server.Get(R"(/api/jobs/([0-9a-f]{32}))", checked([&engine, &config](const auto& request, auto& response) {
        const auto job = engine.owned(request.matches[1], owner(request, config));
        const auto [body, etag] = engine.snapshot(job);
        response.set_header("ETag", etag);
        if (request.get_header_value("If-None-Match") == etag) { response.status = 304; return; }
        response.set_content(body, "application/json; charset=utf-8");
    }));
    server.Post(R"(/api/jobs/([0-9a-f]{32})/cancel)", checked([&engine, &config](const auto& request, auto& response) {
        engine.cancel(engine.owned(request.matches[1], owner(request, config))); json_response(response, {{"ok", true}});
    }));
    server.Get(R"(/api/jobs/([0-9a-f]{32})/report)", checked([&engine, &config](const auto& request, auto& response) {
        const auto path = engine.report(engine.owned(request.matches[1], owner(request, config)));
        response.set_content(read_file(path), "text/plain; charset=utf-8");
    }));
    server.Get(R"(/api/jobs/([0-9a-f]{32})/artifacts)", checked([&engine, &config](const auto& request, auto& response) {
        const auto path = engine.artifacts(engine.owned(request.matches[1], owner(request, config)));
        const auto stream = std::make_shared<std::ifstream>(path, std::ios::binary);
        if (!*stream) throw std::runtime_error("Cannot open archive");
        response.set_header("Content-Disposition", "attachment; filename=\"" + request.matches[1].str() + ".zip\"");
        response.set_content_provider(fs::file_size(path), "application/zip", [stream](std::size_t offset, std::size_t length, httplib::DataSink& sink) {
            stream->clear(); stream->seekg(static_cast<std::streamoff>(offset));
            char buffer[65536];
            while (length) {
                const auto chunk = std::min(length, sizeof(buffer));
                stream->read(buffer, static_cast<std::streamsize>(chunk));
                const auto count = stream->gcount();
                if (count <= 0 || !sink.write(buffer, static_cast<std::size_t>(count))) return false;
                length -= static_cast<std::size_t>(count);
            }
            return true;
        });
    }));
    const auto static_path = config.root / "llm_eval/web/static";
    for (const auto& [route, file, mime] : std::vector<std::tuple<std::string, std::string, std::string>>{
            {"/", "index.html", "text/html; charset=utf-8"}, {"/static/app.js", "app.js", "text/javascript; charset=utf-8"},
            {"/static/styles.css", "styles.css", "text/css; charset=utf-8"}}) {
        const auto body = std::make_shared<std::string>(read_file(static_path / file));
        server.Get(route, [body, mime](const auto&, auto& response) { response.set_content(*body, mime); });
    }
}
}
