// test_facade_tf.cpp
// Facade tests with real TensorFlow C library
//
// Framework: Custom (see below)
// Runs with: Real TensorFlow (Linux CI only)
//
// These tests cover:
// - Model::Load: actual SavedModel loading
// - Model::resolve: endpoint resolution in loaded model
// - Model::runner: inference execution
// - Model::warmup: production warmup feature
// - Model::validate_input: dtype validation
// - Runner::run / run_one: actual inference

#include "tf_wrap/facade.hpp"
#include "tf_wrap/session.hpp"
#include "tf_wrap/graph.hpp"
#include "tf_wrap/tensor.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace tf_wrap;

// ============================================================================
// Test Framework
// ============================================================================

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

#define REQUIRE_CLOSE(a, b, eps) \
    do { \
        if (std::abs((a) - (b)) > (eps)) { \
            throw std::runtime_error("REQUIRE_CLOSE failed: " #a " vs " #b); \
        } \
    } while (0)

#define REQUIRE_THROWS(expr) \
    do { \
        bool threw = false; \
        try { (void)(expr); } catch (...) { threw = true; } \
        if (!threw) throw std::runtime_error("REQUIRE_THROWS failed: " #expr); \
    } while (0)

// ============================================================================
// Test Model Path
// ============================================================================

// The test SavedModel is created by CI with signature:
//   serving_default: x (float32[None]) -> output (float32[None])
//   Model computes: y = x * 2.0 + 1.0

static const char* TEST_MODEL_PATH = "test_savedmodel";

// ============================================================================
// Model::Load Tests
// ============================================================================

TEST(model_load_savedmodel) {
    auto model = Model::Load(TEST_MODEL_PATH);
    REQUIRE(model.valid());
    REQUIRE(static_cast<bool>(model));
}

TEST(model_load_with_tags) {
    auto model = Model::Load(TEST_MODEL_PATH, {"serve"});
    REQUIRE(model.valid());
}

TEST(model_load_nonexistent_throws) {
    REQUIRE_THROWS(Model::Load("/nonexistent/path/to/model"));
}

// ============================================================================
// Model State Access Tests
// ============================================================================

TEST(model_session_access) {
    auto model = Model::Load(TEST_MODEL_PATH);
    const Session& session = model.session();
    REQUIRE(session.valid());
    REQUIRE(session.handle() != nullptr);
}

TEST(model_graph_access) {
    auto model = Model::Load(TEST_MODEL_PATH);
    const Graph& graph = model.graph();
    REQUIRE(graph.valid());
    REQUIRE(graph.handle() != nullptr);
}

// ============================================================================
// Model Move Semantics Tests
// ============================================================================

TEST(model_move_constructor) {
    auto m1 = Model::Load(TEST_MODEL_PATH);
    REQUIRE(m1.valid());
    
    Model m2(std::move(m1));
    REQUIRE(m2.valid());
    REQUIRE(!m1.valid());
}

TEST(model_move_assignment) {
    auto m1 = Model::Load(TEST_MODEL_PATH);
    REQUIRE(m1.valid());
    
    Model m2;
    m2 = std::move(m1);
    REQUIRE(m2.valid());
    REQUIRE(!m1.valid());
}

TEST(model_moved_from_state) {
    auto m1 = Model::Load(TEST_MODEL_PATH);
    Model m2(std::move(m1));
    
    // m1 is now invalid - operations should throw
    REQUIRE_THROWS(m1.runner());
    REQUIRE_THROWS(m1.session());
    REQUIRE_THROWS(m1.graph());
}

// ============================================================================
// Model::resolve Tests
// ============================================================================

TEST(model_resolve_input) {
    auto model = Model::Load(TEST_MODEL_PATH);
    
    // The test model has input named "serving_default_x"
    auto input = model.resolve("serving_default_x");
    REQUIRE(input.oper != nullptr);
    REQUIRE(input.index == 0);
}

TEST(model_resolve_output) {
    auto model = Model::Load(TEST_MODEL_PATH);
    
    // The test model has output named "StatefulPartitionedCall"
    auto output = model.resolve("StatefulPartitionedCall");
    REQUIRE(output.oper != nullptr);
}

TEST(model_resolve_pair) {
    auto model = Model::Load(TEST_MODEL_PATH);
    
    auto [input, output] = model.resolve("serving_default_x", "StatefulPartitionedCall");
    REQUIRE(input.oper != nullptr);
    REQUIRE(output.oper != nullptr);
}

TEST(model_resolve_not_found_throws) {
    auto model = Model::Load(TEST_MODEL_PATH);
    REQUIRE_THROWS(model.resolve("nonexistent_operation"));
}

// ============================================================================
// Runner Inference Tests
// ============================================================================

TEST(runner_basic_inference) {
    auto model = Model::Load(TEST_MODEL_PATH);
    auto input_op = model.resolve("serving_default_x");
    auto output_op = model.resolve("StatefulPartitionedCall");
    
    // Input: [1.0, 2.0, 3.0]
    auto input = Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    
    auto results = model.runner()
        .feed(input_op, input)
        .fetch(output_op)
        .run();
    
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].dtype() == TF_FLOAT);
    
    // Expected: x * 2.0 + 1.0 = [3.0, 5.0, 7.0]
    auto view = results[0].read<float>();
    REQUIRE(view.size() == 3);
    REQUIRE_CLOSE(view[0], 3.0f, 1e-5f);
    REQUIRE_CLOSE(view[1], 5.0f, 1e-5f);
    REQUIRE_CLOSE(view[2], 7.0f, 1e-5f);
}

TEST(runner_run_one) {
    auto model = Model::Load(TEST_MODEL_PATH);
    auto input_op = model.resolve("serving_default_x");
    auto output_op = model.resolve("StatefulPartitionedCall");
    
    auto input = Tensor::FromScalar<float>(5.0f);
    
    auto result = model.runner()
        .feed(input_op, input)
        .fetch(output_op)
        .run_one();
    
    REQUIRE(result.dtype() == TF_FLOAT);
    
    // Expected: 5.0 * 2.0 + 1.0 = 11.0
    REQUIRE_CLOSE(result.ToScalar<float>(), 11.0f, 1e-5f);
}

TEST(runner_run_one_wrong_fetch_count_throws) {
    auto model = Model::Load(TEST_MODEL_PATH);
    auto input_op = model.resolve("serving_default_x");
    auto output_op = model.resolve("StatefulPartitionedCall");
    
    auto input = Tensor::FromScalar<float>(1.0f);
    
    // No fetches - should throw
    REQUIRE_THROWS(model.runner().feed(input_op, input).run_one());
    
    // Two fetches - should throw
    REQUIRE_THROWS(model.runner()
        .feed(input_op, input)
        .fetch(output_op)
        .fetch(output_op)
        .run_one());
}

TEST(runner_multiple_runs) {
    auto model = Model::Load(TEST_MODEL_PATH);
    auto input_op = model.resolve("serving_default_x");
    auto output_op = model.resolve("StatefulPartitionedCall");
    
    // Run inference multiple times with same runner
    auto runner = model.runner();
    
    for (int i = 0; i < 3; ++i) {
        runner.clear();
        auto input = Tensor::FromScalar<float>(static_cast<float>(i));
        runner.feed(input_op, input).fetch(output_op);
        
        auto result = runner.run_one();
        float expected = static_cast<float>(i) * 2.0f + 1.0f;
        REQUIRE_CLOSE(result.ToScalar<float>(), expected, 1e-5f);
    }
}

// ============================================================================
// Model::warmup Tests
// ============================================================================

TEST(model_warmup_single) {
    auto model = Model::Load(TEST_MODEL_PATH);
    auto input_op = model.resolve("serving_default_x");
    auto output_op = model.resolve("StatefulPartitionedCall");
    
    auto dummy = Tensor::FromVector<float>({1}, {0.0f});
    
    // warmup should not throw
    model.warmup(input_op, dummy, output_op);
}

TEST(model_warmup_span) {
    auto model = Model::Load(TEST_MODEL_PATH);
    auto input_op = model.resolve("serving_default_x");
    auto output_op = model.resolve("StatefulPartitionedCall");
    
    auto dummy = Tensor::FromVector<float>({1}, {0.0f});
    
    std::vector<Feed> feeds = {{input_op, dummy}};
    std::vector<Fetch> fetches = {{output_op}};
    
    // warmup should not throw
    model.warmup(feeds, fetches);
}

// ============================================================================
// Model::validate_input Tests
// ============================================================================

TEST(model_validate_input_correct_dtype) {
    auto model = Model::Load(TEST_MODEL_PATH);
    auto input_op = model.resolve("serving_default_x");
    
    // Input should be float32
    auto tensor = Tensor::FromScalar<float>(1.0f);
    auto error = model.validate_input(input_op, tensor);
    
    REQUIRE(error.empty());
}

TEST(model_validate_input_wrong_dtype) {
    auto model = Model::Load(TEST_MODEL_PATH);
    auto input_op = model.resolve("serving_default_x");
    
    // Input should be float32, but we provide int32
    auto tensor = Tensor::FromScalar<int>(1);
    auto error = model.validate_input(input_op, tensor);
    
    REQUIRE(!error.empty());
    // Error should mention dtype mismatch
}

TEST(model_require_valid_input_correct) {
    auto model = Model::Load(TEST_MODEL_PATH);
    auto input_op = model.resolve("serving_default_x");
    
    auto tensor = Tensor::FromScalar<float>(1.0f);
    
    // Should not throw
    model.require_valid_input(input_op, tensor);
}

TEST(model_require_valid_input_wrong_throws) {
    auto model = Model::Load(TEST_MODEL_PATH);
    auto input_op = model.resolve("serving_default_x");
    
    auto tensor = Tensor::FromScalar<int>(1);
    
    REQUIRE_THROWS(model.require_valid_input(input_op, tensor));
}

// ============================================================================
// Runner with Options/Metadata Tests
// ============================================================================

TEST(runner_with_empty_options) {
    auto model = Model::Load(TEST_MODEL_PATH);
    auto input_op = model.resolve("serving_default_x");
    auto output_op = model.resolve("StatefulPartitionedCall");
    
    auto input = Tensor::FromScalar<float>(1.0f);
    Buffer options;  // Empty buffer is valid
    
    auto result = model.runner()
        .with_options(options)
        .feed(input_op, input)
        .fetch(output_op)
        .run_one();
    
    REQUIRE_CLOSE(result.ToScalar<float>(), 3.0f, 1e-5f);
}

TEST(runner_with_metadata_buffer) {
    auto model = Model::Load(TEST_MODEL_PATH);
    auto input_op = model.resolve("serving_default_x");
    auto output_op = model.resolve("StatefulPartitionedCall");
    
    auto input = Tensor::FromScalar<float>(1.0f);
    Buffer metadata;
    
    auto result = model.runner()
        .with_metadata(metadata)
        .feed(input_op, input)
        .fetch(output_op)
        .run_one();
    
    REQUIRE_CLOSE(result.ToScalar<float>(), 3.0f, 1e-5f);
    // metadata buffer may or may not be populated depending on TF version
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== TensorFlowWrap Facade Tests (Real TF) ===\n\n";
    
    // Tests run automatically via static initialization
    
    std::cout << "\n=== Results: " << tests_passed << "/" << tests_run << " passed ===\n";
    
    return (tests_passed == tests_run) ? 0 : 1;
}
