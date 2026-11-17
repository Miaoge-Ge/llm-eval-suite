#include "web.hpp"
#include <curl/curl.h>
#include <csignal>
#include <iostream>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace { volatile std::sig_atomic_t stopped = 0; void stop(int) { stopped = 1; } }
int main() {
    umask(0077);
    std::signal(SIGPIPE, SIG_IGN);
    std::signal(SIGTERM, stop); std::signal(SIGINT, stop);
    curl_global_init(CURL_GLOBAL_DEFAULT);
    int lock_fd = -1;
    try {
        const auto config = eval::Config::from_env();
        eval::fs::create_directories(config.state);
        lock_fd = open((config.state / ".native.lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (lock_fd < 0 || flock(lock_fd, LOCK_EX | LOCK_NB) != 0) throw std::runtime_error("Another service owns this state directory");
        eval::cleanup_containers("llm-eval.service=" + eval::sha256(config.state.string()));
        eval::Engine engine(config);
        httplib::Server server;
        eval::configure_web(server, engine, config);
        if (!server.bind_to_port("127.0.0.1", config.port)) throw std::runtime_error("Cannot bind local service port");
        std::cout << "Native service listening on 127.0.0.1:" << config.port << std::endl;
        std::atomic<bool> listener_stopped{false};
        std::thread listener([&] { server.listen_after_bind(); listener_stopped = true; });
        while (!stopped && !listener_stopped) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        server.stop(); listener.join(); engine.shutdown();
        close(lock_fd); lock_fd = -1;
    } catch (const std::exception& error) {
        std::cerr << "Native service failed: " << error.what() << '\n';
        if (lock_fd >= 0) close(lock_fd);
        curl_global_cleanup(); return 1;
    }
    curl_global_cleanup(); return 0;
}
