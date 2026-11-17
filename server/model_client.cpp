#include "model_client.hpp"
#include <curl/curl.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <memory>
#include <set>

namespace eval {
namespace {
std::string url_part(CURLU* handle, CURLUPart part, unsigned flags = 0) {
    char* text = nullptr;
    const auto code = curl_url_get(handle, part, &text, flags);
    if (code != CURLUE_OK) return {};
    std::string value(text); curl_free(text); return value;
}
bool nonempty(const std::string& text) {
    return std::any_of(text.begin(), text.end(), [](unsigned char ch) { return !std::isspace(ch); });
}
bool cancelled_wait(std::atomic<bool>& cancelled, int milliseconds) {
    for (int i = 0; i < milliseconds && !cancelled; i += 50) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    return cancelled;
}
long long count(const Json& object, const char* key) {
    if (!object.contains(key) || object[key].is_null()) return 0;
    if (!object[key].is_number_integer() || object[key].get<long long>() < 0)
        throw std::runtime_error("Invalid token usage in model stream");
    return object[key].get<long long>();
}
struct Transfer {
    StreamDecoder decoder;
    std::atomic<bool>& cancelled;
    long status = 0;
    std::size_t bytes = 0;
    std::string error_body;
    std::string protocol_error;
    explicit Transfer(bool timing, std::atomic<bool>& flag): decoder(timing), cancelled(flag) {}
};
struct CurlHandle {
    CURL* value = curl_easy_init();
    ~CurlHandle() { if (value) curl_easy_cleanup(value); }
};
size_t receive(char* data, size_t size, size_t count_bytes, void* opaque) {
    const auto length = size * count_bytes;
    auto& state = *static_cast<Transfer*>(opaque);
    try {
        state.bytes += length;
        if (state.bytes > 16 * 1024 * 1024) throw std::runtime_error("Model response exceeds 16 MiB limit");
        if (state.status >= 200 && state.status < 300) state.decoder.feed(std::string_view(data, length));
        else if (state.error_body.size() < 65536) state.error_body.append(data, std::min(length, 65536 - state.error_body.size()));
        return state.cancelled ? 0 : length;
    } catch (const std::exception& error) {
        state.protocol_error = error.what(); return 0;
    }
}
size_t header(char* data, size_t size, size_t count_bytes, void* opaque) {
    const auto length = size * count_bytes;
    auto& state = *static_cast<Transfer*>(opaque);
    const std::string line(data, length);
    if (line.starts_with("HTTP/")) {
        const auto space = line.find(' ');
        if (space != std::string::npos) state.status = std::strtol(line.c_str() + space + 1, nullptr, 10);
    }
    return length;
}
int progress(void* opaque, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
    return static_cast<Transfer*>(opaque)->cancelled ? 1 : 0;
}
}
Endpoint endpoint(const std::string& base_url, bool public_only) {
    std::unique_ptr<CURLU, decltype(&curl_url_cleanup)> handle(curl_url(), curl_url_cleanup);
    if (!handle || curl_url_set(handle.get(), CURLUPART_URL, base_url.c_str(), 0) != CURLUE_OK)
        throw std::invalid_argument("API 地址无效。");
    const auto scheme = url_part(handle.get(), CURLUPART_SCHEME);
    const auto host = url_part(handle.get(), CURLUPART_HOST);
    if ((scheme != "https" && (public_only || scheme != "http")) || host.empty() ||
        !url_part(handle.get(), CURLUPART_USER).empty() || !url_part(handle.get(), CURLUPART_PASSWORD).empty() ||
        !url_part(handle.get(), CURLUPART_QUERY).empty() || !url_part(handle.get(), CURLUPART_FRAGMENT).empty())
        throw std::invalid_argument("API 地址需要有效的 HTTPS 公网地址，不能含内嵌凭据或查询参数。");
    const int port = std::stoi(url_part(handle.get(), CURLUPART_PORT, CURLU_DEFAULT_PORT));
    std::string url = base_url;
    while (!url.empty() && url.back() == '/') url.pop_back();
    std::string hostname = host;
    if (hostname.starts_with('[') && hostname.ends_with(']')) hostname = hostname.substr(1, hostname.size() - 2);
    return {url + "/chat/completions", hostname, scheme, port};
}
bool public_ip(const std::string& address) {
    in_addr ipv4{};
    if (inet_pton(AF_INET, address.c_str(), &ipv4) == 1) {
        const auto ip = ntohl(ipv4.s_addr);
        const std::pair<unsigned, unsigned> blocked[] = {
            {0x00000000, 8}, {0x0a000000, 8}, {0x64400000, 10}, {0x7f000000, 8},
            {0xa9fe0000, 16}, {0xac100000, 12}, {0xc0000000, 24}, {0xc0000200, 24},
            {0xc0586300, 24}, {0xc0a80000, 16}, {0xc6120000, 15}, {0xc6336400, 24},
            {0xcb007100, 24}, {0xe0000000, 3}
        };
        for (auto [network, prefix] : blocked) if ((ip & (0xffffffffu << (32 - prefix))) == network) return false;
        return true;
    }
    in6_addr ipv6{};
    if (inet_pton(AF_INET6, address.c_str(), &ipv6) != 1) return false;
    const auto* bytes = ipv6.s6_addr;
    // Only native global unicast, excluding documentation and transition ranges.
    if ((bytes[0] & 0xe0) != 0x20) return false;
    if (bytes[0] == 0x20 && bytes[1] == 0x02) return false;
    if (bytes[0] == 0x20 && bytes[1] == 0x01 &&
        ((bytes[2] == 0x0d && bytes[3] == 0xb8) || bytes[2] < 2)) return false;
    return true;
}
std::vector<std::string> resolve_public(const std::string& host, int port) {
    addrinfo hints{}; hints.ai_socktype = SOCK_STREAM; hints.ai_family = AF_UNSPEC;
    addrinfo* raw = nullptr;
    const auto status = getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &raw);
    if (status != 0) throw std::runtime_error("Cannot resolve API hostname");
    std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> records(raw, freeaddrinfo);
    std::set<std::string> addresses;
    for (auto* row = raw; row; row = row->ai_next) {
        char text[INET6_ADDRSTRLEN]{};
        const void* address = row->ai_family == AF_INET ?
            static_cast<void*>(&reinterpret_cast<sockaddr_in*>(row->ai_addr)->sin_addr) :
            static_cast<void*>(&reinterpret_cast<sockaddr_in6*>(row->ai_addr)->sin6_addr);
        if (!inet_ntop(row->ai_family, address, text, sizeof(text)) || !public_ip(text))
            throw std::runtime_error("Anonymous API requests cannot access private or reserved addresses");
        addresses.emplace(text);
    }
    if (addresses.empty()) throw std::runtime_error("API hostname has no public address");
    return {addresses.begin(), addresses.end()};
}
void StreamDecoder::feed(std::string_view data) {
    pending_.append(data);
    std::size_t newline;
    while ((newline = pending_.find('\n')) != std::string::npos) {
        auto value = pending_.substr(0, newline); pending_.erase(0, newline + 1);
        if (!value.empty() && value.back() == '\r') value.pop_back();
        line(std::move(value));
    }
}
void StreamDecoder::line(std::string value) {
    if (value.empty()) { event(); return; }
    if (value.starts_with("data:")) {
        value.erase(0, 5);
        if (value.starts_with(' ')) value.erase(0, 1);
        if (!data_.empty()) data_ += '\n';
        data_ += value;
    }
}
void StreamDecoder::event() {
    std::string data = std::move(data_); data_.clear();
    if (data.empty() || data == "[DONE]") return;
    const Json chunk = Json::parse(data);
    if (chunk.contains("error")) throw std::runtime_error("Provider returned an error inside the model stream");
    if (chunk.contains("model") && chunk["model"].is_string()) metadata["response_model"] = chunk["model"];
    if (chunk.contains("usage") && chunk["usage"].is_object()) {
        const auto& source = chunk["usage"];
        usage = Json::object();
        for (const auto* field : {"prompt_tokens", "completion_tokens", "total_tokens"})
            if (source.contains(field) && !source[field].is_null()) usage[field] = count(source, field);
        if (source.contains("completion_tokens_details") && source["completion_tokens_details"].is_object()) {
            const auto& details = source["completion_tokens_details"];
            if (details.contains("reasoning_tokens") && details["reasoning_tokens"].is_number_integer())
                metadata["reasoning_tokens"] = details["reasoning_tokens"];
        }
        if (source.contains("prompt_cache_hit_tokens")) metadata["cached_prompt_tokens"] = source["prompt_cache_hit_tokens"];
        if (source.contains("prompt_tokens_details") && source["prompt_tokens_details"].is_object() &&
            source["prompt_tokens_details"].contains("cached_tokens"))
            metadata["cached_prompt_tokens"] = source["prompt_tokens_details"]["cached_tokens"];
    }
    if (!chunk.contains("choices") || !chunk["choices"].is_array()) return;
    for (const auto& choice : chunk["choices"]) {
        if (choice.value("index", -1) != 0) continue;
        if (choice.contains("finish_reason") && choice["finish_reason"].is_string())
            metadata["finish_reason"] = choice["finish_reason"];
        if (!choice.contains("delta") || !choice["delta"].is_object()) continue;
        const auto& delta = choice["delta"];
        const auto body = delta.contains("content") && delta["content"].is_string() ? delta["content"].get<std::string>() : "";
        const auto thought = delta.contains("reasoning_content") && delta["reasoning_content"].is_string() ?
            delta["reasoning_content"].get<std::string>() : "";
        content += body; reasoning += thought;
        if (capture_ && (!body.empty() || !thought.empty()))
            events_.push_back({{"at_seconds", elapsed(start_)}, {"content", body}, {"reasoning", thought}});
    }
}
void StreamDecoder::finish() {
    if (!pending_.empty()) { line(std::move(pending_)); pending_.clear(); }
    event();
}
Json StreamDecoder::record() const {
    Json result = metadata;
    result["content"] = content; result["reasoning_content"] = reasoning; result["usage"] = usage;
    if (capture_) result["timing"] = {{"duration_seconds", elapsed(start_)}, {"events", events_}};
    return result;
}
Json generate(const Endpoint& target, const std::string& key, const Json& request,
              bool public_only, int timeout_seconds, std::atomic<bool>& cancelled,
              int concurrency, int shared_limit, double scheduler_wait,
              const std::function<void(const Json&)>& journal) {
    const bool timing = request.value("capture_timing", false);
    const int max_attempts = std::clamp(request.value("max_attempts", 3), 1, 3);
    Json records = Json::array();
    Json total = {{"prompt_tokens", 0}, {"completion_tokens", 0}, {"total_tokens", 0}};
    bool complete = true, fatal = false, success = false;
    std::string last_error = "Generation cancelled", partial;
    long last_status = 0;
    for (int attempt = 1; attempt <= max_attempts && !cancelled; ++attempt) {
        if (journal) {
            auto unfinished = records;
            Json marker = {{"attempt", attempt}, {"error", "Request interrupted before its outcome was saved"},
                           {"error_type", "GenerationCancelled"}, {"usage", Json::object()},
                           {"delivery_confirmed", false}, {"http_status_code", nullptr}};
            if (timing) {
                marker["timing"] = {{"events", Json::array()}};
                marker["request_concurrency"] = concurrency;
                marker["shared_request_limit"] = shared_limit;
                marker["scheduler_wait_seconds"] = scheduler_wait;
            }
            unfinished.push_back(std::move(marker));
            journal({{"content", partial}, {"usage", total}, {"error", "Request outcome unknown after interruption"},
                     {"fatal", false}, {"http_status_code", nullptr}, {"attempts", attempt},
                     {"usage_complete", false}, {"cancelled", true}, {"attempt_records", unfinished}});
        }
        last_status = 0;
        Transfer transfer(timing, cancelled);
        const auto started = Clock::now();
        bool retryable = false;
        CURLcode result = CURLE_OK;
        std::string error;
        std::string error_type;
        // Keep TLS connections between requests, but never reuse an administrator's
        // potentially private connection for a public-only guest request.
        thread_local CurlHandle handles[2];
        CURL* curl = handles[public_only ? 1 : 0].value;
        if (curl) curl_easy_reset(curl);
        curl_slist* raw_headers = nullptr;
        curl_slist* raw_resolve = nullptr;
        const std::string body = request.at("body").dump();
        try {
            if (!curl) throw std::runtime_error("Cannot initialize HTTP client");
            if (public_only) {
                const auto addresses = resolve_public(target.host, target.port);
                std::string mapping = target.host + ":" + std::to_string(target.port) + ":";
                for (std::size_t i = 0; i < addresses.size(); ++i) {
                    if (i) mapping += ',';
                    mapping += addresses[i].find(':') != std::string::npos ? "[" + addresses[i] + "]" : addresses[i];
                }
                raw_resolve = curl_slist_append(raw_resolve, mapping.c_str());
                curl_easy_setopt(curl, CURLOPT_RESOLVE, raw_resolve);
            }
            raw_headers = curl_slist_append(raw_headers, "Content-Type: application/json");
            raw_headers = curl_slist_append(raw_headers, "Accept: text/event-stream");
            raw_headers = curl_slist_append(raw_headers, ("Authorization: Bearer " + key).c_str());
            curl_easy_setopt(curl, CURLOPT_URL, target.url.c_str());
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, raw_headers);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(body.size()));
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, receive);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &transfer);
            curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, header);
            curl_easy_setopt(curl, CURLOPT_HEADERDATA, &transfer);
            curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress);
            curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &transfer);
            curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
            curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
            curl_easy_setopt(curl, CURLOPT_TIMEOUT, static_cast<long>(timeout_seconds));
            curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 15L);
            curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 0L);
            curl_easy_setopt(curl, CURLOPT_PROXY, "");
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
            result = curl_easy_perform(curl);
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &last_status);
            transfer.decoder.finish();
            if (cancelled) { error = "Generation cancelled"; error_type = "GenerationCancelled"; }
            else if (!transfer.protocol_error.empty()) { error = transfer.protocol_error; error_type = "InvalidStream"; }
            else if (result != CURLE_OK) {
                error = curl_easy_strerror(result);
                error_type = result == CURLE_OPERATION_TIMEDOUT ? "APITimeoutError" : "APIConnectionError";
                retryable = true;
            } else if (last_status < 200 || last_status >= 300) {
                error = "HTTP " + std::to_string(last_status) + ": " + transfer.error_body;
                error_type = "HTTPStatusError";
                fatal = last_status == 401 || last_status == 403;
                retryable = last_status == 408 || last_status == 409 || last_status == 429 || last_status >= 500;
            } else if (!transfer.decoder.metadata.contains("finish_reason")) {
                error = "Model stream ended without a finish_reason; answer may be truncated";
                error_type = "IncompleteStream"; retryable = true;
            } else if (!nonempty(transfer.decoder.content)) {
                error = "Model returned an empty completion"; error_type = "EmptyCompletion";
                retryable = transfer.decoder.metadata.value("finish_reason", "") != "length";
            } else success = true;
        } catch (const std::exception& exception) {
            error = exception.what(); error_type = "InvalidStreamOrEndpoint";
        }
        curl_slist_free_all(raw_headers); curl_slist_free_all(raw_resolve);
        Json record = transfer.decoder.record();
        record["attempt"] = attempt;
        record["duration_seconds"] = elapsed(started);
        record["http_status_code"] = last_status ? Json(last_status) : Json(nullptr);
        record["error"] = error.empty() ? Json(nullptr) : Json(redact(error, key));
        if (!error_type.empty()) record["error_type"] = error_type;
        if (timing) {
            record["request_concurrency"] = concurrency;
            record["shared_request_limit"] = shared_limit;
            record["scheduler_wait_seconds"] = attempt == 1 ? scheduler_wait : 0;
        }
        for (const auto* field : {"prompt_tokens", "completion_tokens", "total_tokens"})
            if (!transfer.decoder.usage.contains(field)) complete = false;
        for (const char* name : {"prompt_tokens", "completion_tokens", "total_tokens"})
            total[name] = total[name].get<long long>() + count(transfer.decoder.usage, name);
        records.push_back(std::move(record));
        partial = transfer.decoder.content;
        last_error = redact(error, key);
        if (success || fatal || !retryable || cancelled) break;
        if (attempt < max_attempts && cancelled_wait(cancelled, attempt * 1000)) break;
    }
    const bool was_cancelled = cancelled.load() && !success;
    return {{"content", partial}, {"usage", total}, {"error", success ? Json(nullptr) : Json(last_error)},
            {"fatal", fatal}, {"http_status_code", last_status ? Json(last_status) : Json(nullptr)},
            {"attempts", records.size()}, {"usage_complete", complete}, {"cancelled", was_cancelled},
            {"attempt_records", records}};
}
}
