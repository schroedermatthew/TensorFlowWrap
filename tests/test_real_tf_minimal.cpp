// test_real_tf_minimal.cpp
// Minimal test for TensorFlowWrap with real TensorFlow C library
//
// Tests:
// 1. Load SavedModel
// 2. Run inference using signature inputs/outputs
// 3. Verify results

#include "tf_wrap/core.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

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
    do { \
        if (!(cond)) { \
            throw std::runtime_error("REQUIRE failed: " #cond); \
        } \
    } while (0)

#define REQUIRE_CLOSE(a, b, eps) \
    do { \
        if (std::abs((a) - (b)) > (eps)) { \
            throw std::runtime_error( \
                "REQUIRE_CLOSE failed: " #a " ≈ " #b \
                " (got " + std::to_string(a) + " vs " + std::to_string(b) + ")"); \
        } \
    } while (0)

// Path to test SavedModel (created by CI)
static const char* SAVEDMODEL_PATH = "test_savedmodel";

// ============================================================================
// Test: Load SavedModel
// ============================================================================
TEST(load_savedmodel) {
    auto model = tf_wrap::Model::Load(SAVEDMODEL_PATH);
    REQUIRE(model.valid());
}

// ============================================================================
// Test: Tensor creation and access
// ============================================================================
TEST(tensor_basics) {
    // FromScalar
    auto t1 = tf_wrap::Tensor::FromScalar<float>(3.14f);
    REQUIRE(t1.num_elements() == 1);
    REQUIRE_CLOSE(t1.ToScalar<float>(), 3.14f, 0.001f);
    
    // FromVector
    auto t2 = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    REQUIRE(t2.num_elements() == 3);
    auto view = t2.read<float>();
    REQUIRE_CLOSE(view[0], 1.0f, 0.001f);
    REQUIRE_CLOSE(view[1], 2.0f, 0.001f);
    REQUIRE_CLOSE(view[2], 3.0f, 0.001f);
    
    // Zeros
    auto t3 = tf_wrap::Tensor::Zeros<float>({2, 2});
    REQUIRE(t3.num_elements() == 4);
    auto view3 = t3.read<float>();
    for (int i = 0; i < 4; ++i) {
        REQUIRE_CLOSE(view3[i], 0.0f, 0.001f);
    }
    
    // Clone
    auto t4 = t2.Clone();
    REQUIRE(t4.num_elements() == 3);
    auto view4 = t4.read<float>();
    REQUIRE_CLOSE(view4[1], 2.0f, 0.001f);
}

// ============================================================================
// Test: Run inference using SavedModel signature
// ============================================================================
TEST(run_savedmodel_inference) {
    auto model = tf_wrap::Model::Load(SAVEDMODEL_PATH);
    auto& graph = model.graph();
    
    // For SavedModel signatures, the input placeholder is named after the signature
    // Input: "serving_default_x:0"
    // Output: Try "PartitionedCall" first (TF 2.16+), then "StatefulPartitionedCall"
    
    // Find input - should be "serving_default_x"
    auto input_op = graph.GetOperation("serving_default_x");
    REQUIRE(input_op.has_value());
    TF_Output input{*input_op, 0};
    
    // Find output - try PartitionedCall first (avoids saver dependencies in TF 2.16+)
    TF_Output output{nullptr, 0};
    if (auto op = graph.GetOperation("PartitionedCall")) {
        output = TF_Output{*op, 0};
    } else if (auto op = graph.GetOperation("StatefulPartitionedCall")) {
        output = TF_Output{*op, 0};
    }
    REQUIRE(output.oper != nullptr);
    
    // Create input tensor: [1.0, 2.0, 3.0]
    auto input_tensor = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    
    // Run inference
    auto results = model.session().Run(
        {tf_wrap::Feed(input, input_tensor)},
        {tf_wrap::Fetch(output)},
        {}
    );
    
    // Model computes: output = input * 2 + 1
    // Expected: [3.0, 5.0, 7.0]
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].num_elements() == 3);
    
    auto view = results[0].read<float>();
    REQUIRE_CLOSE(view[0], 3.0f, 0.001f);
    REQUIRE_CLOSE(view[1], 5.0f, 0.001f);
    REQUIRE_CLOSE(view[2], 7.0f, 0.001f);
}

// ============================================================================
// Test: Run inference multiple times
// ============================================================================
TEST(run_inference_repeated) {
    auto model = tf_wrap::Model::Load(SAVEDMODEL_PATH);
    auto& graph = model.graph();
    
    auto input_op = graph.GetOperation("serving_default_x");
    REQUIRE(input_op.has_value());
    TF_Output input{*input_op, 0};
    
    TF_Output output{nullptr, 0};
    if (auto op = graph.GetOperation("PartitionedCall")) {
        output = TF_Output{*op, 0};
    } else if (auto op = graph.GetOperation("StatefulPartitionedCall")) {
        output = TF_Output{*op, 0};
    }
    REQUIRE(output.oper != nullptr);
    
    // Run 50 times
    for (int i = 0; i < 50; ++i) {
        float val = static_cast<float>(i);
        auto input_tensor = tf_wrap::Tensor::FromVector<float>({1}, {val});
        
        auto results = model.session().Run(
            {tf_wrap::Feed(input, input_tensor)},
            {tf_wrap::Fetch(output)},
            {}
        );
        
        // output = input * 2 + 1
        float expected = val * 2.0f + 1.0f;
        auto view = results[0].read<float>();
        REQUIRE_CLOSE(view[0], expected, 0.001f);
    }
}

// ============================================================================
// Test: Runner fluent API
// ============================================================================
TEST(runner_api) {
    auto model = tf_wrap::Model::Load(SAVEDMODEL_PATH);
    auto& graph = model.graph();
    
    auto input_op = graph.GetOperation("serving_default_x");
    REQUIRE(input_op.has_value());
    TF_Output input{*input_op, 0};
    
    TF_Output output{nullptr, 0};
    if (auto op = graph.GetOperation("PartitionedCall")) {
        output = TF_Output{*op, 0};
    } else if (auto op = graph.GetOperation("StatefulPartitionedCall")) {
        output = TF_Output{*op, 0};
    }
    REQUIRE(output.oper != nullptr);
    
    auto input_tensor = tf_wrap::Tensor::FromVector<float>({3}, {10.0f, 20.0f, 30.0f});
    
    // Use Runner fluent API
    auto result = model.runner()
        .feed(input, input_tensor)
        .fetch(output)
        .run_one();
    
    REQUIRE(result.num_elements() == 3);
    
    auto view = result.read<float>();
    REQUIRE_CLOSE(view[0], 21.0f, 0.001f);  // 10*2+1
    REQUIRE_CLOSE(view[1], 41.0f, 0.001f);  // 20*2+1
    REQUIRE_CLOSE(view[2], 61.0f, 0.001f);  // 30*2+1
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "=== TensorFlowWrap Real TF Minimal Test ===\n\n";
    std::cout << "Using SavedModel: " << SAVEDMODEL_PATH << "\n\n";
    
    // Tests run automatically via static initialization
    
    std::cout << "\n=== Results: " << tests_passed << "/" << tests_run << " passed ===\n";
    
    return (tests_passed == tests_run) ? 0 : 1;
}
