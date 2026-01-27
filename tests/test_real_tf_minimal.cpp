// test_real_tf_minimal.cpp
// Minimal test for TensorFlowWrap with real TensorFlow C library
//
// Tests:
// 1. Load SavedModel
// 2. Resolve endpoints
// 3. Run inference with handle-based API
// 4. Verify results

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
// Test: Resolve endpoints
// ============================================================================
TEST(resolve_endpoints) {
    auto model = tf_wrap::Model::Load(SAVEDMODEL_PATH);
    
    // The model has signature "serving_default" with input "x" and output
    // SavedModel signature names vary, so we look for common patterns
    auto& graph = model.graph();
    
    // Print available operations for debugging
    std::cout << "\n  Available ops: ";
    auto ops = graph.GetAllOperations();
    int count = 0;
    for (auto* op : ops) {
        if (count++ < 10) {
            std::cout << TF_OperationName(op) << " ";
        }
    }
    if (ops.size() > 10) std::cout << "... (" << ops.size() << " total)";
    std::cout << "\n  ";
    
    // Try to find input/output - names vary by TF version
    // Common patterns: "serving_default_x", "x", "StatefulPartitionedCall"
    TF_Output input{nullptr, 0};
    TF_Output output{nullptr, 0};
    
    // Try common input names
    const char* input_names[] = {
        "serving_default_x", "serving_default_input", "x", "input"
    };
    for (const char* name : input_names) {
        if (auto op = graph.GetOperation(name)) {
            input = TF_Output{*op, 0};
            std::cout << "Found input: " << name << "\n  ";
            break;
        }
    }
    
    // Try common output names
    const char* output_names[] = {
        "StatefulPartitionedCall", "PartitionedCall", "Identity", "output"
    };
    for (const char* name : output_names) {
        if (auto op = graph.GetOperation(name)) {
            output = TF_Output{*op, 0};
            std::cout << "Found output: " << name << "\n  ";
            break;
        }
    }
    
    REQUIRE(input.oper != nullptr);
    REQUIRE(output.oper != nullptr);
}

// ============================================================================
// Test: Run inference with handle-based API
// ============================================================================
TEST(run_inference) {
    auto model = tf_wrap::Model::Load(SAVEDMODEL_PATH);
    auto& graph = model.graph();
    
    // Find endpoints (same logic as above)
    TF_Output input{nullptr, 0};
    TF_Output output{nullptr, 0};
    
    const char* input_names[] = {"serving_default_x", "serving_default_input", "x", "input"};
    for (const char* name : input_names) {
        if (auto op = graph.GetOperation(name)) {
            input = TF_Output{*op, 0};
            break;
        }
    }
    
    const char* output_names[] = {"StatefulPartitionedCall", "PartitionedCall", "Identity", "output"};
    for (const char* name : output_names) {
        if (auto op = graph.GetOperation(name)) {
            output = TF_Output{*op, 0};
            break;
        }
    }
    
    REQUIRE(input.oper != nullptr);
    REQUIRE(output.oper != nullptr);
    
    // Create input tensor: [1.0, 2.0, 3.0]
    auto input_tensor = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    
    // Run inference using handle-based API
    auto result = model.runner()
        .feed(input, input_tensor)
        .fetch(output)
        .run_one();
    
    // Model computes: output = input * 2 + 1
    // Expected: [3.0, 5.0, 7.0]
    REQUIRE(result.num_elements() == 3);
    
    auto view = result.read<float>();
    REQUIRE_CLOSE(view[0], 3.0f, 0.001f);
    REQUIRE_CLOSE(view[1], 5.0f, 0.001f);
    REQUIRE_CLOSE(view[2], 7.0f, 0.001f);
}

// ============================================================================
// Test: Run inference multiple times (verify no leaks/crashes)
// ============================================================================
TEST(run_inference_repeated) {
    auto model = tf_wrap::Model::Load(SAVEDMODEL_PATH);
    auto& graph = model.graph();
    
    // Find endpoints
    TF_Output input{nullptr, 0};
    TF_Output output{nullptr, 0};
    
    const char* input_names[] = {"serving_default_x", "serving_default_input", "x", "input"};
    for (const char* name : input_names) {
        if (auto op = graph.GetOperation(name)) {
            input = TF_Output{*op, 0};
            break;
        }
    }
    
    const char* output_names[] = {"StatefulPartitionedCall", "PartitionedCall", "Identity", "output"};
    for (const char* name : output_names) {
        if (auto op = graph.GetOperation(name)) {
            output = TF_Output{*op, 0};
            break;
        }
    }
    
    REQUIRE(input.oper != nullptr);
    REQUIRE(output.oper != nullptr);
    
    // Run 100 times
    for (int i = 0; i < 100; ++i) {
        float val = static_cast<float>(i);
        auto input_tensor = tf_wrap::Tensor::FromVector<float>({1}, {val});
        
        auto result = model.runner()
            .feed(input, input_tensor)
            .fetch(output)
            .run_one();
        
        // output = input * 2 + 1
        float expected = val * 2.0f + 1.0f;
        auto view = result.read<float>();
        REQUIRE_CLOSE(view[0], expected, 0.001f);
    }
}

// ============================================================================
// Test: Session::Run with Feed/Fetch structs
// ============================================================================
TEST(session_run_structs) {
    auto model = tf_wrap::Model::Load(SAVEDMODEL_PATH);
    auto& graph = model.graph();
    
    // Find endpoints
    TF_Output input{nullptr, 0};
    TF_Output output{nullptr, 0};
    
    const char* input_names[] = {"serving_default_x", "serving_default_input", "x", "input"};
    for (const char* name : input_names) {
        if (auto op = graph.GetOperation(name)) {
            input = TF_Output{*op, 0};
            break;
        }
    }
    
    const char* output_names[] = {"StatefulPartitionedCall", "PartitionedCall", "Identity", "output"};
    for (const char* name : output_names) {
        if (auto op = graph.GetOperation(name)) {
            output = TF_Output{*op, 0};
            break;
        }
    }
    
    REQUIRE(input.oper != nullptr);
    REQUIRE(output.oper != nullptr);
    
    // Create input
    auto input_tensor = tf_wrap::Tensor::FromVector<float>({3}, {10.0f, 20.0f, 30.0f});
    
    // Run using Feed/Fetch structs directly
    auto results = model.session().Run(
        {tf_wrap::Feed(input, input_tensor)},
        {tf_wrap::Fetch(output)},
        {}
    );
    
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].num_elements() == 3);
    
    auto view = results[0].read<float>();
    REQUIRE_CLOSE(view[0], 21.0f, 0.001f);  // 10*2+1
    REQUIRE_CLOSE(view[1], 41.0f, 0.001f);  // 20*2+1
    REQUIRE_CLOSE(view[2], 61.0f, 0.001f);  // 30*2+1
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    std::cout << "=== TensorFlowWrap Real TF Minimal Test ===\n\n";
    
    // Check if savedmodel exists
    std::cout << "Using SavedModel: " << SAVEDMODEL_PATH << "\n\n";
    
    // Tests run automatically via static initialization
    
    std::cout << "\n=== Results: " << tests_passed << "/" << tests_run << " passed ===\n";
    
    return (tests_passed == tests_run) ? 0 : 1;
}
