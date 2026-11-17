#include "common.hpp"
#include "model_client.hpp"
#include <curl/curl.h>
#include <cassert>
#include <iostream>

int main() {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    assert(eval::sha256("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    assert(eval::hex_id(eval::random_hex()));
    assert(eval::constant_equal("same", "same") && !eval::constant_equal("same", "other"));
    for (const auto* ip : {"127.0.0.1", "10.0.0.1", "169.254.169.254", "172.16.1.1", "192.168.1.1", "100.64.0.1", "::1", "fc00::1", "::ffff:127.0.0.1", "2001:db8::1", "224.0.0.1"})
        assert(!eval::public_ip(ip));
    assert(eval::public_ip("1.1.1.1") && eval::public_ip("2606:4700:4700::1111"));
    assert(eval::endpoint("https://api.example.com/v1/", true).url == "https://api.example.com/v1/chat/completions");
    bool denied = false;
    try { eval::endpoint("http://127.0.0.1/v1", true); } catch (...) { denied = true; }
    assert(denied);
    eval::StreamDecoder stream(true);
    const std::string input = "data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"think\"}}]}\r\n\r\n"
        "data: {\"choices\":[{\"index\":1,\"delta\":{\"content\":\"ignored\"}},{\"index\":0,\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\n"
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":4,\"total_tokens\":7,\"completion_tokens_details\":{\"reasoning_tokens\":2}}}\n\n"
        "data: [DONE]\n\n";
    for (char ch : input) stream.feed(std::string_view(&ch, 1));
    stream.finish();
    assert(stream.content == "answer" && stream.reasoning == "think");
    const auto record = stream.record();
    assert(record["finish_reason"] == "stop" && record["usage"]["total_tokens"] == 7);
    assert(record["reasoning_tokens"] == 2 && record["timing"]["events"].size() == 2);
    assert(record["timing"]["events"][0]["at_seconds"] <= record["timing"]["events"][1]["at_seconds"]);
    eval::StreamDecoder missing(true);
    missing.feed("data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5}}\n\n");
    assert(!missing.usage.contains("completion_tokens"));
    eval::ThreadPool pool(4);
    std::vector<std::future<int>> futures;
    for (int i = 0; i < 100; ++i) futures.push_back(pool.submit([i] { return i * i; }));
    for (int i = 0; i < 100; ++i) assert(futures[i].get() == i * i);
    std::cout << "Native stream, address validation, hashing and worker tests passed\n";
    curl_global_cleanup();
}
