// test_session_tf.cpp
// Session tests with real TensorFlow C library
//
// Tests Session, SessionOptions, Graph, Buffer, DeviceList with real TF runtime.

#include "tf_wrap/session.hpp"
#include "tf_wrap/graph.hpp"
#include "tf_wrap/tensor.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace tf_wrap;

// Simple test framework
static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    void test_##name(); \
    struct TestRunner_##name { \
        TestRunner_##name() { \
            std::cout << "Testing " #name "... " << std::flush; \
            tests_run++; \
            try { \
                test_##name(); \
                std::cout << "PASSED\n"; \
                tests_passed++; \
            } catch (const std::exception& e) { \
                std::cout << "FAILED: " << e.what() << "\n"; \
            } catch (...) { \
                std::cout << "FAILED: unknown exception\n"; \
            } \
        } \
    } test_runner_##name; \
    void test_##name()

#define REQUIRE(cond) \
    do { if (!(cond)) throw std::runtime_error("REQUIRE failed: " #cond); } while (0)

#define REQUIRE_THROWS(expr) \
    do { \
        bool threw = false; \
        try { (void)(expr); } catch (...) { threw = true; } \
        if (!threw) throw std::runtime_error("REQUIRE_THROWS failed: " #expr); \
    } while (0)

// ============================================================================
// SessionOptions Tests
// ============================================================================

TEST(session_options_default) {
    SessionOptions opts;
    REQUIRE(opts.handle() != nullptr);
}

TEST(session_options_move) {
    SessionOptions opts1;
    auto* handle1 = opts1.handle();
    
    SessionOptions opts2(std::move(opts1));
    REQUIRE(opts2.handle() == handle1);
    REQUIRE(opts1.handle() == nullptr);
}

TEST(session_options_set_target) {
    SessionOptions opts;
    auto& ref = opts.SetTarget("");
    REQUIRE(&ref == &opts);
}

// ============================================================================
// Buffer Tests
// ============================================================================

TEST(buffer_default) {
    Buffer buf;
    REQUIRE(buf.handle() != nullptr);
    REQUIRE(buf.empty());
}

TEST(buffer_from_data) {
    const char* data = "hello";
    Buffer buf(data, 5);
    
    REQUIRE(!buf.empty());
    REQUIRE(buf.length() == 5);
}

TEST(buffer_to_bytes) {
    const char* data = "test";
    Buffer buf(data, 4);
    
    auto bytes = buf.to_bytes();
    REQUIRE(bytes.size() == 4);
    REQUIRE(bytes[0] == 't');
}

TEST(buffer_move) {
    Buffer buf1("abc", 3);
    auto* handle1 = buf1.handle();
    
    Buffer buf2(std::move(buf1));
    REQUIRE(buf2.handle() == handle1);
    REQUIRE(buf1.handle() == nullptr);
}

// ============================================================================
// Graph Tests
// ============================================================================

TEST(graph_default) {
    Graph graph;
    REQUIRE(graph.handle() != nullptr);
    REQUIRE(graph.valid());
}

TEST(graph_move) {
    Graph g1;
    auto* handle1 = g1.handle();
    
    Graph g2(std::move(g1));
    REQUIRE(g2.handle() == handle1);
    REQUIRE(!g1.valid());
}

TEST(graph_get_operation_not_found) {
    Graph graph;
    auto op = graph.GetOperation("nonexistent");
    REQUIRE(!op.has_value());
}

TEST(graph_get_all_operations_empty) {
    Graph graph;
    auto ops = graph.GetAllOperations();
    REQUIRE(ops.empty());
}

// ============================================================================
// Session Construction Tests
// ============================================================================

TEST(session_from_graph) {
    Graph graph;
    Session session(graph);
    
    REQUIRE(session.valid());
    REQUIRE(session.handle() != nullptr);
    REQUIRE(session.graph_handle() != nullptr);
}

TEST(session_from_graph_with_options) {
    Graph graph;
    SessionOptions opts;
    Session session(graph, opts);
    
    REQUIRE(session.valid());
}

TEST(session_move) {
    Graph graph;
    Session s1(graph);
    auto* handle1 = s1.handle();
    
    Session s2(std::move(s1));
    REQUIRE(s2.handle() == handle1);
    REQUIRE(!s1.valid());
}

// ============================================================================
// Session resolve Tests
// ============================================================================

TEST(session_resolve_not_found) {
    Graph graph;
    Session session(graph);
    
    REQUIRE_THROWS(session.resolve("nonexistent"));
}

// ============================================================================
// DeviceList Tests
// ============================================================================

TEST(session_list_devices) {
    Graph graph;
    Session session(graph);
    
    auto devices = session.ListDevices();
    // Real TF should have at least CPU
    REQUIRE(devices.count() >= 1);
    
    // First device should be accessible
    auto dev = devices.at(0);
    REQUIRE(!dev.name.empty());
    REQUIRE(!dev.type.empty());
}

TEST(session_has_gpu) {
    Graph graph;
    Session session(graph);
    
    // Just verify it doesn't crash - result depends on hardware
    bool has_gpu = session.HasGPU();
    (void)has_gpu;
}

TEST(device_list_all) {
    Graph graph;
    Session session(graph);
    
    auto devices = session.ListDevices();
    auto all = devices.all();
    
    REQUIRE(all.size() == static_cast<std::size_t>(devices.count()));
    
    // Should have CPU
    bool has_cpu = false;
    for (const auto& d : all) {
        if (d.is_cpu()) has_cpu = true;
    }
    REQUIRE(has_cpu);
}

// ============================================================================
// Session Error Cases
// ============================================================================

TEST(session_run_on_moved_from) {
    Graph graph;
    Session s1(graph);
    Session s2(std::move(s1));
    
    std::vector<Feed> feeds;
    std::vector<Fetch> fetches;
    REQUIRE_THROWS(s1.Run(feeds, fetches));
}

TEST(session_resolve_on_moved_from) {
    Graph graph;
    Session s1(graph);
    Session s2(std::move(s1));
    
    REQUIRE_THROWS(s1.resolve("x"));
}

TEST(session_list_devices_on_moved_from) {
    Graph graph;
    Session s1(graph);
    Session s2(std::move(s1));
    
    REQUIRE_THROWS(s1.ListDevices());
}

// ============================================================================
// LoadSavedModel Error Case
// ============================================================================

TEST(load_savedmodel_nonexistent_throws) {
    REQUIRE_THROWS(Session::LoadSavedModel("/nonexistent/path/to/model"));
}

// ============================================================================
// Feed/Fetch/Target Construction
// ============================================================================

TEST(feed_from_tensor) {
    auto tensor = Tensor::FromScalar<float>(1.0f);
    TF_Output output{nullptr, 0};
    
    Feed feed(output, tensor);
    REQUIRE(feed.tensor == tensor.handle());
    REQUIRE(feed.keepalive != nullptr);
}

TEST(fetch_construction) {
    TF_Output output{nullptr, 5};
    Fetch fetch(output);
    REQUIRE(fetch.output.index == 5);
}

TEST(target_construction) {
    Target target(nullptr);
    REQUIRE(target.oper == nullptr);
}

// ============================================================================
// RunContext Tests
// ============================================================================

TEST(run_context_basic) {
    RunContext ctx(16, 8);
    auto tensor = Tensor::FromScalar<float>(1.0f);
    TF_Output output{nullptr, 0};
    
    ctx.add_feed(output, tensor);
    ctx.add_fetch(output);
    ctx.add_target(nullptr);
    ctx.reset();
    
    // After reset, can add again
    ctx.add_feed(output, tensor);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== TensorFlowWrap Session Tests (Real TF) ===\n\n";
    
    // Tests run automatically via static initialization
    
    std::cout << "\n=== Results: " << tests_passed << "/" << tests_run << " passed ===\n";
    
    return (tests_passed == tests_run) ? 0 : 1;
}
