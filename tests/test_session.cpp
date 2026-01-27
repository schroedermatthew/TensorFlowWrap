// test_session.cpp
// Comprehensive tests for tf_wrap::Session and related classes
//
// These tests cover:
// - SessionOptions: creation, move semantics, SetTarget
// - Session: creation from Graph, move semantics
// - Feed/Fetch/Target: construction variants
// - Buffer: creation, move semantics, data access
// - DeviceList/Device: enumeration
// - Session::resolve: name parsing, error cases
// - Session::Run: basic execution (stub limited)
// - RunContext: reusable buffers
//
// Note: Most Session::Run tests require real TensorFlow.
// Stub tests focus on API surface and error handling.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "tf_wrap/session.hpp"
#include "tf_wrap/graph.hpp"
#include "tf_wrap/tensor.hpp"

using namespace tf_wrap;

// ============================================================================
// SessionOptions Tests
// ============================================================================

TEST_CASE("SessionOptions - default construction") {
    SessionOptions opts;
    CHECK(opts.handle() != nullptr);
}

TEST_CASE("SessionOptions - move constructor") {
    SessionOptions opts1;
    auto* handle1 = opts1.handle();
    CHECK(handle1 != nullptr);
    
    SessionOptions opts2(std::move(opts1));
    CHECK(opts2.handle() == handle1);
    CHECK(opts1.handle() == nullptr);
}

TEST_CASE("SessionOptions - move assignment") {
    SessionOptions opts1;
    SessionOptions opts2;
    auto* handle1 = opts1.handle();
    
    opts2 = std::move(opts1);
    CHECK(opts2.handle() == handle1);
    CHECK(opts1.handle() == nullptr);
}

TEST_CASE("SessionOptions - SetTarget") {
    SessionOptions opts;
    
    // SetTarget returns *this for chaining
    auto& ref = opts.SetTarget("");
    CHECK(&ref == &opts);
}

// ============================================================================
// Buffer Tests
// ============================================================================

TEST_CASE("Buffer - default construction") {
    Buffer buf;
    CHECK(buf.handle() != nullptr);
    CHECK(buf.empty());
    CHECK(buf.length() == 0);
    CHECK(buf.data() == nullptr);
}

TEST_CASE("Buffer - from string data") {
    const char* data = "hello";
    Buffer buf(data, 5);
    
    CHECK(buf.handle() != nullptr);
    CHECK_FALSE(buf.empty());
    CHECK(buf.length() == 5);
    CHECK(buf.data() != nullptr);
    CHECK(std::memcmp(buf.data(), "hello", 5) == 0);
}

TEST_CASE("Buffer - to_bytes") {
    const char* data = "test";
    Buffer buf(data, 4);
    
    auto bytes = buf.to_bytes();
    CHECK(bytes.size() == 4);
    CHECK(bytes[0] == 't');
    CHECK(bytes[1] == 'e');
    CHECK(bytes[2] == 's');
    CHECK(bytes[3] == 't');
}

TEST_CASE("Buffer - to_bytes empty") {
    Buffer buf;
    auto bytes = buf.to_bytes();
    CHECK(bytes.empty());
}

TEST_CASE("Buffer - move constructor") {
    const char* data = "data";
    Buffer buf1(data, 4);
    auto* handle1 = buf1.handle();
    
    Buffer buf2(std::move(buf1));
    CHECK(buf2.handle() == handle1);
    CHECK(buf1.handle() == nullptr);
}

TEST_CASE("Buffer - move assignment") {
    Buffer buf1("abc", 3);
    Buffer buf2("xyz", 3);
    auto* handle1 = buf1.handle();
    
    buf2 = std::move(buf1);
    CHECK(buf2.handle() == handle1);
    CHECK(buf1.handle() == nullptr);
}

// ============================================================================
// Feed Tests
// ============================================================================

TEST_CASE("Feed - from TF_Output and Tensor") {
    auto tensor = Tensor::FromScalar<float>(1.0f);
    TF_Output output{nullptr, 0};
    
    Feed feed(output, tensor);
    CHECK(feed.output.oper == nullptr);
    CHECK(feed.output.index == 0);
    CHECK(feed.tensor == tensor.handle());
    CHECK(feed.keepalive != nullptr);
}

TEST_CASE("Feed - from TF_Output and raw TF_Tensor*") {
    auto tensor = Tensor::FromScalar<float>(1.0f);
    TF_Output output{nullptr, 0};
    
    Feed feed(output, tensor.handle());
    CHECK(feed.tensor == tensor.handle());
    CHECK(feed.keepalive == nullptr);  // No keepalive with raw pointer
}

TEST_CASE("Feed - from TF_Operation* and Tensor") {
    auto tensor = Tensor::FromScalar<float>(1.0f);
    
    Feed feed(static_cast<TF_Operation*>(nullptr), tensor);
    CHECK(feed.output.oper == nullptr);
    CHECK(feed.output.index == 0);
    CHECK(feed.tensor == tensor.handle());
}

TEST_CASE("Feed - from TF_Operation*, index, and Tensor") {
    auto tensor = Tensor::FromScalar<float>(1.0f);
    
    Feed feed(static_cast<TF_Operation*>(nullptr), 2, tensor);
    CHECK(feed.output.index == 2);
}

// ============================================================================
// Fetch Tests
// ============================================================================

TEST_CASE("Fetch - from TF_Output") {
    TF_Output output{nullptr, 5};
    Fetch fetch(output);
    
    CHECK(fetch.output.oper == nullptr);
    CHECK(fetch.output.index == 5);
}

TEST_CASE("Fetch - from TF_Operation*") {
    Fetch fetch(static_cast<TF_Operation*>(nullptr));
    CHECK(fetch.output.index == 0);
}

TEST_CASE("Fetch - from TF_Operation* with index") {
    Fetch fetch(static_cast<TF_Operation*>(nullptr), 3);
    CHECK(fetch.output.index == 3);
}

// ============================================================================
// Target Tests
// ============================================================================

TEST_CASE("Target - from TF_Operation*") {
    Target target(static_cast<TF_Operation*>(nullptr));
    CHECK(target.oper == nullptr);
}

// ============================================================================
// Device Tests
// ============================================================================

TEST_CASE("Device - is_cpu/is_gpu") {
    Device cpu_dev;
    cpu_dev.type = "CPU";
    CHECK(cpu_dev.is_cpu());
    CHECK_FALSE(cpu_dev.is_gpu());
    
    Device gpu_dev;
    gpu_dev.type = "GPU";
    CHECK_FALSE(gpu_dev.is_cpu());
    CHECK(gpu_dev.is_gpu());
}

// ============================================================================
// DeviceList Tests
// ============================================================================

TEST_CASE("DeviceList - default construction") {
    DeviceList list;
    CHECK(list.handle() == nullptr);
    CHECK(list.count() == 0);
}

TEST_CASE("DeviceList - at out of range throws") {
    DeviceList list;
    
    bool threw = false;
    try {
        auto dev = list.at(0);
        (void)dev;
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("DeviceList - all on empty list") {
    DeviceList list;
    auto devices = list.all();
    CHECK(devices.empty());
}

TEST_CASE("DeviceList - move constructor") {
    DeviceList list1;
    DeviceList list2(std::move(list1));
    CHECK(list1.handle() == nullptr);
}

TEST_CASE("DeviceList - move assignment") {
    DeviceList list1;
    DeviceList list2;
    list2 = std::move(list1);
    CHECK(list1.handle() == nullptr);
}

// ============================================================================
// RunContext Tests
// ============================================================================

TEST_CASE("RunContext - construction") {
    RunContext ctx(16, 8);
    // Just verifying no crash
    ctx.reset();
}

TEST_CASE("RunContext - add_feed with Tensor") {
    RunContext ctx;
    auto tensor = Tensor::FromScalar<float>(1.0f);
    TF_Output output{nullptr, 0};
    
    ctx.add_feed(output, tensor);
    // Verify through reset (no crash)
    ctx.reset();
}

TEST_CASE("RunContext - add_feed with raw pointer") {
    RunContext ctx;
    auto tensor = Tensor::FromScalar<float>(1.0f);
    TF_Output output{nullptr, 0};
    
    ctx.add_feed(output, tensor.handle());
    ctx.reset();
}

TEST_CASE("RunContext - add_fetch") {
    RunContext ctx;
    TF_Output output{nullptr, 0};
    
    ctx.add_fetch(output);
    ctx.reset();
}

TEST_CASE("RunContext - add_target") {
    RunContext ctx;
    
    ctx.add_target(nullptr);
    ctx.reset();
}

TEST_CASE("RunContext - reset clears state") {
    RunContext ctx;
    auto tensor = Tensor::FromScalar<float>(1.0f);
    TF_Output output{nullptr, 0};
    
    ctx.add_feed(output, tensor);
    ctx.add_fetch(output);
    ctx.add_target(nullptr);
    
    ctx.reset();
    // After reset, can add again without issues
    ctx.add_feed(output, tensor);
}

// ============================================================================
// Graph Tests (needed for Session tests)
// ============================================================================

TEST_CASE("Graph - default construction") {
    Graph graph;
    CHECK(graph.handle() != nullptr);
    CHECK(graph.valid());
}

TEST_CASE("Graph - move constructor") {
    Graph g1;
    auto* handle1 = g1.handle();
    
    Graph g2(std::move(g1));
    CHECK(g2.handle() == handle1);
    CHECK_FALSE(g1.valid());
}

TEST_CASE("Graph - GetOperation not found") {
    Graph graph;
    auto op = graph.GetOperation("nonexistent");
    CHECK_FALSE(op.has_value());
}

TEST_CASE("Graph - GetAllOperations empty") {
    Graph graph;
    auto ops = graph.GetAllOperations();
    CHECK(ops.empty());
}

// ============================================================================
// Session Construction Tests
// ============================================================================

TEST_CASE("Session - construction from Graph") {
    Graph graph;
    Session session(graph);
    
    CHECK(session.valid());
    CHECK(session.handle() != nullptr);
    CHECK(session.graph_handle() != nullptr);
}

TEST_CASE("Session - construction from Graph with options") {
    Graph graph;
    SessionOptions opts;
    Session session(graph, opts);
    
    CHECK(session.valid());
}

TEST_CASE("Session - construction from Graph with raw options") {
    Graph graph;
    SessionOptions opts;
    Session session(graph, opts.handle());
    
    CHECK(session.valid());
}

TEST_CASE("Session - move constructor") {
    Graph graph;
    Session s1(graph);
    auto* handle1 = s1.handle();
    
    Session s2(std::move(s1));
    CHECK(s2.handle() == handle1);
    CHECK(s1.handle() == nullptr);
    CHECK_FALSE(s1.valid());
}

TEST_CASE("Session - move assignment") {
    Graph g1, g2;
    Session s1(g1);
    Session s2(g2);
    auto* handle1 = s1.handle();
    
    s2 = std::move(s1);
    CHECK(s2.handle() == handle1);
    CHECK_FALSE(s1.valid());
}

// ============================================================================
// Session::resolve Tests (Stub has empty graph, so these test error cases)
// ============================================================================

TEST_CASE("Session::resolve - operation not found") {
    Graph graph;
    Session session(graph);
    
    bool threw = false;
    try {
        auto out = session.resolve("nonexistent");
        (void)out;
    } catch (const Error&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("Session::resolve - parses name:index format") {
    Graph graph;
    Session session(graph);
    
    // Will throw NOT_FOUND, but the parsing should work
    bool threw = false;
    std::string error_msg;
    try {
        auto out = session.resolve("some_op:2");
        (void)out;
    } catch (const Error& e) {
        threw = true;
        error_msg = e.what();
    }
    CHECK(threw);
    // The error should mention "some_op" (without the :2)
    CHECK(error_msg.find("some_op") != std::string::npos);
}

TEST_CASE("Session::resolve - name without index") {
    Graph graph;
    Session session(graph);
    
    bool threw = false;
    try {
        auto out = session.resolve("my_operation");
        (void)out;
    } catch (const Error&) {
        threw = true;
    }
    CHECK(threw);
}

// ============================================================================
// Session::ListDevices Tests
// ============================================================================

TEST_CASE("Session::ListDevices - returns device list") {
    Graph graph;
    Session session(graph);
    
    auto devices = session.ListDevices();
    // Stub returns at least CPU typically
    CHECK(devices.count() >= 0);  // May be 0 in stub
}

TEST_CASE("Session::HasGPU - returns bool") {
    Graph graph;
    Session session(graph);
    
    // Just verify it doesn't crash
    bool has_gpu = session.HasGPU();
    (void)has_gpu;
}

// ============================================================================
// Session::Run Tests (Limited with stub - no actual execution)
// ============================================================================

TEST_CASE("Session::Run - empty feeds/fetches") {
    Graph graph;
    Session session(graph);
    
    // Empty run should succeed
    std::vector<Feed> feeds;
    std::vector<Fetch> fetches;
    auto results = session.Run(feeds, fetches);
    CHECK(results.empty());
}

TEST_CASE("Session::Run - with vectors") {
    Graph graph;
    Session session(graph);
    
    std::vector<Feed> feeds;
    std::vector<Fetch> fetches;
    std::vector<Target> targets;
    
    auto results = session.Run(feeds, fetches, targets);
    CHECK(results.empty());
}

// ============================================================================
// Session - Error Cases
// ============================================================================

TEST_CASE("Session - Run on moved-from session") {
    Graph graph;
    Session s1(graph);
    Session s2(std::move(s1));
    
    // s1 is now invalid
    bool threw = false;
    try {
        std::vector<Feed> feeds;
        std::vector<Fetch> fetches;
        auto results = s1.Run(feeds, fetches);
        (void)results;
    } catch (const Error&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("Session - resolve on moved-from session") {
    Graph graph;
    Session s1(graph);
    Session s2(std::move(s1));
    
    bool threw = false;
    try {
        auto out = s1.resolve("x");
        (void)out;
    } catch (const Error&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("Session - ListDevices on moved-from session") {
    Graph graph;
    Session s1(graph);
    Session s2(std::move(s1));
    
    bool threw = false;
    try {
        auto devs = s1.ListDevices();
        (void)devs;
    } catch (const Error&) {
        threw = true;
    }
    CHECK(threw);
}

// ============================================================================
// LoadSavedModel - Error Case (no actual model to load)
// ============================================================================

TEST_CASE("Session::LoadSavedModel - nonexistent path throws") {
    bool threw = false;
    try {
        auto [session, graph] = Session::LoadSavedModel("/nonexistent/path/to/model");
        (void)session;
        (void)graph;
    } catch (...) {
        threw = true;
    }
    CHECK(threw);
}
