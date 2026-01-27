// test_facade.cpp
// Comprehensive tests for tf_wrap::Runner and tf_wrap::Model
//
// Framework: doctest
// Runs with: Stub TensorFlow (all platforms)
//
// These tests cover:
// - Runner: construction, feed, fetch, target, chaining, clear, error cases
// - Model: default construction, move semantics, valid(), error cases
//
// Note: Tests requiring actual SavedModel loading and inference are in
// test_facade_tf.cpp which runs with real TensorFlow.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "tf_wrap/facade.hpp"
#include "tf_wrap/session.hpp"
#include "tf_wrap/graph.hpp"
#include "tf_wrap/tensor.hpp"

using namespace tf_wrap;

// ============================================================================
// Runner Construction Tests
// ============================================================================

TEST_CASE("Runner - construction from Session") {
    Graph graph;
    Session session(graph);
    
    Runner runner(session);
    // Runner constructed successfully - no crash
}

// ============================================================================
// Runner Feed Tests
// ============================================================================

TEST_CASE("Runner - feed with Tensor") {
    Graph graph;
    Session session(graph);
    Runner runner(session);
    
    auto tensor = Tensor::FromScalar<float>(1.0f);
    TF_Output output{nullptr, 0};
    
    auto& ref = runner.feed(output, tensor);
    CHECK(&ref == &runner);
}

TEST_CASE("Runner - feed with raw TF_Tensor*") {
    Graph graph;
    Session session(graph);
    Runner runner(session);
    
    auto tensor = Tensor::FromScalar<float>(1.0f);
    TF_Output output{nullptr, 0};
    
    auto& ref = runner.feed(output, tensor.handle());
    CHECK(&ref == &runner);
}

TEST_CASE("Runner - feed null Tensor throws") {
    Graph graph;
    Session session(graph);
    Runner runner(session);
    
    Tensor tensor;  // Default constructed - null handle
    TF_Output output{nullptr, 0};
    
    bool threw = false;
    try {
        runner.feed(output, tensor);
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_INVALID_ARGUMENT);
    }
    CHECK(threw);
}

TEST_CASE("Runner - feed null TF_Tensor* throws") {
    Graph graph;
    Session session(graph);
    Runner runner(session);
    
    TF_Output output{nullptr, 0};
    
    bool threw = false;
    try {
        runner.feed(output, static_cast<TF_Tensor*>(nullptr));
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_INVALID_ARGUMENT);
    }
    CHECK(threw);
}

TEST_CASE("Runner - feed multiple tensors") {
    Graph graph;
    Session session(graph);
    Runner runner(session);
    
    auto t1 = Tensor::FromScalar<float>(1.0f);
    auto t2 = Tensor::FromScalar<float>(2.0f);
    auto t3 = Tensor::FromScalar<float>(3.0f);
    TF_Output out1{nullptr, 0};
    TF_Output out2{nullptr, 1};
    TF_Output out3{nullptr, 2};
    
    runner.feed(out1, t1).feed(out2, t2).feed(out3, t3);
    // No crash - feeds accumulated
}

// ============================================================================
// Runner Fetch Tests
// ============================================================================

TEST_CASE("Runner - fetch single output") {
    Graph graph;
    Session session(graph);
    Runner runner(session);
    
    TF_Output output{nullptr, 0};
    
    auto& ref = runner.fetch(output);
    CHECK(&ref == &runner);
}

TEST_CASE("Runner - fetch multiple outputs") {
    Graph graph;
    Session session(graph);
    Runner runner(session);
    
    TF_Output out1{nullptr, 0};
    TF_Output out2{nullptr, 1};
    TF_Output out3{nullptr, 2};
    
    runner.fetch(out1).fetch(out2).fetch(out3);
    // No crash - fetches accumulated
}

// ============================================================================
// Runner Target Tests
// ============================================================================

TEST_CASE("Runner - target operation") {
    Graph graph;
    Session session(graph);
    Runner runner(session);
    
    auto& ref = runner.target(nullptr);
    CHECK(&ref == &runner);
}

TEST_CASE("Runner - multiple targets") {
    Graph graph;
    Session session(graph);
    Runner runner(session);
    
    runner.target(nullptr).target(nullptr).target(nullptr);
    // No crash - targets accumulated
}

// ============================================================================
// Runner with_options / with_metadata Tests
// ============================================================================

TEST_CASE("Runner - with_options") {
    Graph graph;
    Session session(graph);
    Runner runner(session);
    
    Buffer options;
    auto& ref = runner.with_options(options);
    CHECK(&ref == &runner);
}

TEST_CASE("Runner - with_metadata") {
    Graph graph;
    Session session(graph);
    Runner runner(session);
    
    Buffer metadata;
    auto& ref = runner.with_metadata(metadata);
    CHECK(&ref == &runner);
}

// ============================================================================
// Runner Chaining (Fluent API) Tests
// ============================================================================

TEST_CASE("Runner - fluent chaining lvalue") {
    Graph graph;
    Session session(graph);
    Runner runner(session);
    
    auto tensor = Tensor::FromScalar<float>(1.0f);
    TF_Output input{nullptr, 0};
    TF_Output output{nullptr, 1};
    Buffer options;
    Buffer metadata;
    
    runner.with_options(options)
          .with_metadata(metadata)
          .feed(input, tensor)
          .fetch(output)
          .target(nullptr);
    // No crash - all operations chain
}

TEST_CASE("Runner - fluent chaining rvalue") {
    Graph graph;
    Session session(graph);
    
    auto tensor = Tensor::FromScalar<float>(1.0f);
    TF_Output input{nullptr, 0};
    TF_Output output{nullptr, 1};
    
    // Construct and chain in one expression
    auto runner = Runner(session)
        .feed(input, tensor)
        .fetch(output);
    // No crash - rvalue chaining works
    (void)runner;
}

// ============================================================================
// Runner Clear Tests
// ============================================================================

TEST_CASE("Runner - clear resets state") {
    Graph graph;
    Session session(graph);
    Runner runner(session);
    
    auto tensor = Tensor::FromScalar<float>(1.0f);
    TF_Output input{nullptr, 0};
    TF_Output output{nullptr, 1};
    Buffer options;
    Buffer metadata;
    
    runner.with_options(options)
          .with_metadata(metadata)
          .feed(input, tensor)
          .fetch(output)
          .target(nullptr);
    
    runner.clear();
    
    // After clear, can add new feeds/fetches
    auto tensor2 = Tensor::FromScalar<int>(42);
    runner.feed(input, tensor2).fetch(output);
}

// ============================================================================
// Runner run_one Error Tests
// ============================================================================

TEST_CASE("Runner - run_one with zero fetches throws") {
    Graph graph;
    Session session(graph);
    Runner runner(session);
    
    // No fetches added
    bool threw = false;
    try {
        auto result = runner.run_one();
        (void)result;
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_INVALID_ARGUMENT);
        // Error message should mention expected 1 fetch
        std::string msg = e.what();
        CHECK(msg.find("1") != std::string::npos);
    }
    CHECK(threw);
}

TEST_CASE("Runner - run_one with multiple fetches throws") {
    Graph graph;
    Session session(graph);
    Runner runner(session);
    
    TF_Output out1{nullptr, 0};
    TF_Output out2{nullptr, 1};
    runner.fetch(out1).fetch(out2);
    
    bool threw = false;
    try {
        auto result = runner.run_one();
        (void)result;
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_INVALID_ARGUMENT);
    }
    CHECK(threw);
}

// ============================================================================
// Runner run (stub limited - empty graph)
// ============================================================================

#ifdef TF_WRAPPER_TF_STUB_ENABLED
TEST_CASE("Runner - run with empty feeds/fetches (stub only)") {
    Graph graph;
    Session session(graph);
    Runner runner(session);
    
    auto results = runner.run();
    CHECK(results.empty());
}
#endif

// ============================================================================
// Model Default Construction Tests
// ============================================================================

TEST_CASE("Model - default construction") {
    Model model;
    CHECK_FALSE(model.valid());
    CHECK_FALSE(static_cast<bool>(model));
}

// ============================================================================
// Model Move Semantics Tests
// ============================================================================

TEST_CASE("Model - move constructor from default") {
    Model m1;
    Model m2(std::move(m1));
    
    CHECK_FALSE(m1.valid());
    CHECK_FALSE(m2.valid());
}

TEST_CASE("Model - move assignment from default") {
    Model m1;
    Model m2;
    
    m2 = std::move(m1);
    CHECK_FALSE(m1.valid());
    CHECK_FALSE(m2.valid());
}

// ============================================================================
// Model Error Cases (unloaded model)
// ============================================================================

TEST_CASE("Model - runner on unloaded model throws") {
    Model model;
    
    bool threw = false;
    try {
        auto runner = model.runner();
        (void)runner;
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_FAILED_PRECONDITION);
    }
    CHECK(threw);
}

TEST_CASE("Model - session on unloaded model throws") {
    Model model;
    
    bool threw = false;
    try {
        auto& session = model.session();
        (void)session;
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_FAILED_PRECONDITION);
    }
    CHECK(threw);
}

TEST_CASE("Model - graph on unloaded model throws") {
    Model model;
    
    bool threw = false;
    try {
        auto& graph = model.graph();
        (void)graph;
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_FAILED_PRECONDITION);
    }
    CHECK(threw);
}

TEST_CASE("Model - resolve on unloaded model throws") {
    Model model;
    
    bool threw = false;
    try {
        auto output = model.resolve("some_op");
        (void)output;
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_FAILED_PRECONDITION);
    }
    CHECK(threw);
}

TEST_CASE("Model - resolve pair on unloaded model throws") {
    Model model;
    
    bool threw = false;
    try {
        auto endpoints = model.resolve("input", "output");
        (void)endpoints;
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_FAILED_PRECONDITION);
    }
    CHECK(threw);
}

// ============================================================================
// Model validate_input Tests (can test without loaded model)
// ============================================================================

TEST_CASE("Model - validate_input on unloaded model returns error") {
    Model model;
    auto tensor = Tensor::FromScalar<float>(1.0f);
    TF_Output input{nullptr, 0};
    
    auto error = model.validate_input(input, tensor);
    CHECK_FALSE(error.empty());
    CHECK(error.find("not loaded") != std::string::npos);
}

TEST_CASE("Model - validate_input with null tensor returns error") {
    Model model;
    Tensor tensor;  // null handle
    TF_Output input{nullptr, 0};
    
    auto error = model.validate_input(input, tensor);
    CHECK_FALSE(error.empty());
    // Either "not loaded" or "null" - depends on check order
}

// ============================================================================
// Model::Load Error Cases
// ============================================================================

TEST_CASE("Model - Load nonexistent path throws") {
    bool threw = false;
    try {
        auto model = Model::Load("/nonexistent/path/to/model");
        (void)model;
    } catch (...) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("Model - Load with custom tags on nonexistent path throws") {
    bool threw = false;
    try {
        auto model = Model::Load("/nonexistent/path", {"serve", "gpu"});
        (void)model;
    } catch (...) {
        threw = true;
    }
    CHECK(threw);
}

// ============================================================================
// Model warmup Error Cases (unloaded model)
// ============================================================================

TEST_CASE("Model - warmup on unloaded model throws") {
    Model model;
    auto tensor = Tensor::FromScalar<float>(1.0f);
    TF_Output input{nullptr, 0};
    TF_Output output{nullptr, 1};
    
    bool threw = false;
    try {
        model.warmup(input, tensor, output);
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_FAILED_PRECONDITION);
    }
    CHECK(threw);
}

TEST_CASE("Model - warmup span version on unloaded model throws") {
    Model model;
    std::vector<Feed> feeds;
    std::vector<Fetch> fetches;
    
    bool threw = false;
    try {
        model.warmup(feeds, fetches);
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_FAILED_PRECONDITION);
    }
    CHECK(threw);
}

// ============================================================================
// Model require_valid_input Error Cases
// ============================================================================

TEST_CASE("Model - require_valid_input on unloaded model throws") {
    Model model;
    auto tensor = Tensor::FromScalar<float>(1.0f);
    TF_Output input{nullptr, 0};
    
    bool threw = false;
    try {
        model.require_valid_input(input, tensor);
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_INVALID_ARGUMENT);
    }
    CHECK(threw);
}

// ============================================================================
// Model BatchRun Error Cases (unloaded model)
// ============================================================================

TEST_CASE("Model - BatchRun on unloaded model throws") {
    Model model;
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor::FromScalar<float>(1.0f));
    TF_Output input{nullptr, 0};
    TF_Output output{nullptr, 1};
    
    bool threw = false;
    try {
        auto results = model.BatchRun(input, inputs, output);
        (void)results;
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_FAILED_PRECONDITION);
    }
    CHECK(threw);
}

TEST_CASE("Model - BatchRunStacked on unloaded model throws") {
    Model model;
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor::FromScalar<float>(1.0f));
    TF_Output input{nullptr, 0};
    TF_Output output{nullptr, 1};
    
    bool threw = false;
    try {
        auto results = model.BatchRunStacked(input, std::span<const Tensor>(inputs), output);
        (void)results;
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_FAILED_PRECONDITION);
    }
    CHECK(threw);
}

// ============================================================================
// Model operator bool Tests
// ============================================================================

TEST_CASE("Model - operator bool returns valid()") {
    Model model;
    CHECK(static_cast<bool>(model) == model.valid());
    CHECK_FALSE(static_cast<bool>(model));
}

// ============================================================================
// Runner reuse after run (stub only)
// ============================================================================

#ifdef TF_WRAPPER_TF_STUB_ENABLED
TEST_CASE("Runner - reuse after run (stub only)") {
    Graph graph;
    Session session(graph);
    Runner runner(session);
    
    // First run (empty)
    auto results1 = runner.run();
    CHECK(results1.empty());
    
    // Can still add feeds/fetches and run again
    auto tensor = Tensor::FromScalar<float>(1.0f);
    TF_Output input{nullptr, 0};
    runner.feed(input, tensor);
    
    auto results2 = runner.run();
    CHECK(results2.empty());  // Stub doesn't actually execute
}

TEST_CASE("Runner - clear and reuse (stub only)") {
    Graph graph;
    Session session(graph);
    Runner runner(session);
    
    auto tensor = Tensor::FromScalar<float>(1.0f);
    TF_Output input{nullptr, 0};
    runner.feed(input, tensor);
    
    auto results1 = runner.run();
    
    runner.clear();
    
    // After clear, empty run again
    auto results2 = runner.run();
    CHECK(results2.empty());
}
#endif
