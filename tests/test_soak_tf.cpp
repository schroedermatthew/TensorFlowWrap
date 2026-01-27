// test_soak_tf.cpp
// Soak tests with real TensorFlow C library
//
// Framework: Custom (see below)
// Runs with: Real TensorFlow (Linux CI only, under ASan)
//
// These tests run many iterations to catch:
// - Memory leaks (via ASan leak detection)
// - Resource exhaustion
// - Gradual memory growth
//
// Note: These tests are slower than unit tests by design.
// They should be run under AddressSanitizer with detect_leaks=1.

#include "tf_wrap/facade.hpp"
#include "tf_wrap/session.hpp"
#include "tf_wrap/graph.hpp"
#include "tf_wrap/tensor.hpp"

#include <chrono>
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

// ============================================================================
// Test Model Path
// ============================================================================

static const char* TEST_MODEL_PATH = "test_savedmodel";

// ============================================================================
// Helper: Find operations (handles TF version differences)
// ============================================================================

static TF_Output find_output_op(const Graph& graph) {
    if (auto op = graph.GetOperation("PartitionedCall")) {
        return TF_Output{*op, 0};
    }
    if (auto op = graph.GetOperation("StatefulPartitionedCall")) {
        return TF_Output{*op, 0};
    }
    throw std::runtime_error("Could not find output operation");
}

static TF_Output find_input_op(const Graph& graph) {
    if (auto op = graph.GetOperation("serving_default_x")) {
        return TF_Output{*op, 0};
    }
    throw std::runtime_error("Could not find input operation");
}

// ============================================================================
// Helper: Progress reporter for long-running tests
// ============================================================================

class ProgressReporter {
public:
    ProgressReporter(int total, int report_interval = 1000)
        : total_(total), interval_(report_interval), count_(0) {}
    
    void tick() {
        ++count_;
        if (count_ % interval_ == 0) {
            std::cout << "[" << count_ << "/" << total_ << "] " << std::flush;
        }
    }
    
    void done() {
        std::cout << "[" << count_ << "/" << total_ << " complete] " << std::flush;
    }
    
private:
    int total_;
    int interval_;
    int count_;
};

// ============================================================================
// Soak Test: Inference (10,000 iterations)
// ============================================================================
// 
// Purpose: Catch memory leaks in the inference hot path.
// Each iteration creates input tensors, runs inference, and reads results.
// Any leaked TF_Tensor* or internal buffers will be caught by ASan.

TEST(soak_inference_10k) {
    constexpr int ITERATIONS = 10'000;
    
    auto model = Model::Load(TEST_MODEL_PATH);
    REQUIRE(model.valid());
    
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    ProgressReporter progress(ITERATIONS, 2000);
    
    for (int i = 0; i < ITERATIONS; ++i) {
        // Create fresh input tensor each iteration
        float input_val = static_cast<float>(i % 1000);
        auto input = Tensor::FromScalar<float>(input_val);
        
        // Run inference
        auto result = model.runner()
            .feed(input_op, input)
            .fetch(output_op)
            .run_one();
        
        // Verify result (y = x * 2 + 1)
        float expected = input_val * 2.0f + 1.0f;
        float actual = result.ToScalar<float>();
        REQUIRE_CLOSE(actual, expected, 0.001f);
        
        progress.tick();
    }
    
    progress.done();
}

// ============================================================================
// Soak Test: Inference with Vector Tensors (5,000 iterations)
// ============================================================================
//
// Purpose: Catch leaks with larger tensors that have heap allocations.

TEST(soak_inference_vectors_5k) {
    constexpr int ITERATIONS = 5'000;
    constexpr int VECTOR_SIZE = 100;
    
    auto model = Model::Load(TEST_MODEL_PATH);
    REQUIRE(model.valid());
    
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    ProgressReporter progress(ITERATIONS, 1000);
    
    for (int i = 0; i < ITERATIONS; ++i) {
        // Create vector input
        std::vector<float> input_data(VECTOR_SIZE);
        for (int j = 0; j < VECTOR_SIZE; ++j) {
            input_data[j] = static_cast<float>((i + j) % 1000);
        }
        
        auto input = Tensor::FromVector<float>({VECTOR_SIZE}, input_data);
        
        // Run inference
        auto result = model.runner()
            .feed(input_op, input)
            .fetch(output_op)
            .run_one();
        
        // Verify shape and spot-check values
        REQUIRE(result.num_elements() == VECTOR_SIZE);
        auto view = result.read<float>();
        float expected_first = input_data[0] * 2.0f + 1.0f;
        REQUIRE_CLOSE(view[0], expected_first, 0.001f);
        
        progress.tick();
    }
    
    progress.done();
}

// ============================================================================
// Soak Test: Model Load/Unload (100 iterations)
// ============================================================================
//
// Purpose: Catch leaks in model loading/unloading.
// Each Model::Load creates Session, Graph, and internal TF resources.
// Destructor must clean up everything.

TEST(soak_model_reload_100) {
    constexpr int ITERATIONS = 100;
    
    ProgressReporter progress(ITERATIONS, 20);
    
    for (int i = 0; i < ITERATIONS; ++i) {
        // Load model
        auto model = Model::Load(TEST_MODEL_PATH);
        REQUIRE(model.valid());
        
        // Do a quick inference to exercise full path
        auto input_op = find_input_op(model.graph());
        auto output_op = find_output_op(model.graph());
        
        auto input = Tensor::FromScalar<float>(static_cast<float>(i));
        auto result = model.runner()
            .feed(input_op, input)
            .fetch(output_op)
            .run_one();
        
        REQUIRE(result.valid());
        
        // Model destroyed here - ASan will catch leaks
        progress.tick();
    }
    
    progress.done();
}

// ============================================================================
// Soak Test: Tensor Allocation Churn (100,000 iterations)
// ============================================================================
//
// Purpose: Catch leaks in tensor creation/destruction.
// This is pure Tensor operations without inference.

TEST(soak_tensor_churn_100k) {
    constexpr int ITERATIONS = 100'000;
    
    ProgressReporter progress(ITERATIONS, 20000);
    
    for (int i = 0; i < ITERATIONS; ++i) {
        // Scalar tensor
        auto t1 = Tensor::FromScalar<float>(static_cast<float>(i));
        REQUIRE(t1.valid());
        
        // Vector tensor
        auto t2 = Tensor::FromVector<float>({10}, 
            std::vector<float>(10, static_cast<float>(i)));
        REQUIRE(t2.valid());
        
        // Clone (exercises internal copy)
        auto t3 = t2.Clone();
        REQUIRE(t3.valid());
        
        // Read access (exercises view)
        auto view = t3.read<float>();
        REQUIRE(view.size() == 10);
        
        // All tensors destroyed here
        progress.tick();
    }
    
    progress.done();
}

// ============================================================================
// Soak Test: Large Tensor Allocation (1,000 iterations)
// ============================================================================
//
// Purpose: Catch leaks with larger memory allocations.

TEST(soak_large_tensors_1k) {
    constexpr int ITERATIONS = 1'000;
    constexpr int TENSOR_SIZE = 10'000;  // 10K floats = 40KB each
    
    ProgressReporter progress(ITERATIONS, 200);
    
    for (int i = 0; i < ITERATIONS; ++i) {
        // Create large tensor
        auto t = Tensor::Zeros<float>({TENSOR_SIZE});
        REQUIRE(t.valid());
        REQUIRE(t.num_elements() == TENSOR_SIZE);
        REQUIRE(t.byte_size() == TENSOR_SIZE * sizeof(float));
        
        // Write to it
        auto view = t.write<float>();
        view[0] = static_cast<float>(i);
        view[TENSOR_SIZE - 1] = static_cast<float>(i);
        
        // Verify
        auto read_view = t.read<float>();
        REQUIRE_CLOSE(read_view[0], static_cast<float>(i), 0.001f);
        
        progress.tick();
    }
    
    progress.done();
}

// ============================================================================
// Soak Test: Runner Reuse (5,000 iterations)
// ============================================================================
//
// Purpose: Verify Runner::clear() properly releases resources.

TEST(soak_runner_reuse_5k) {
    constexpr int ITERATIONS = 5'000;
    
    auto model = Model::Load(TEST_MODEL_PATH);
    REQUIRE(model.valid());
    
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    // Single runner, reused
    auto runner = model.runner();
    
    ProgressReporter progress(ITERATIONS, 1000);
    
    for (int i = 0; i < ITERATIONS; ++i) {
        runner.clear();  // Reset for reuse
        
        auto input = Tensor::FromScalar<float>(static_cast<float>(i % 1000));
        
        auto result = runner
            .feed(input_op, input)
            .fetch(output_op)
            .run_one();
        
        REQUIRE(result.valid());
        
        progress.tick();
    }
    
    progress.done();
}

// ============================================================================
// Soak Test: String Tensors (1,000 iterations)
// ============================================================================
//
// Purpose: String tensors have different memory management in TF.

TEST(soak_string_tensors_1k) {
    constexpr int ITERATIONS = 1'000;
    
    ProgressReporter progress(ITERATIONS, 200);
    
    for (int i = 0; i < ITERATIONS; ++i) {
        // Create string tensor with varying length
        std::string content = "test_string_" + std::to_string(i) + 
            std::string(i % 100, 'x');  // Variable padding
        
        auto t = Tensor::FromString(content);
        REQUIRE(t.valid());
        REQUIRE(t.dtype() == TF_STRING);
        
        // Clone string tensor
        auto t2 = t.Clone();
        REQUIRE(t2.valid());
        
        progress.tick();
    }
    
    progress.done();
}

// ============================================================================
// Soak Test: Mixed Operations (2,000 iterations)
// ============================================================================
//
// Purpose: Interleave different operations to catch interaction bugs.

TEST(soak_mixed_operations_2k) {
    constexpr int ITERATIONS = 2'000;
    
    auto model = Model::Load(TEST_MODEL_PATH);
    REQUIRE(model.valid());
    
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    ProgressReporter progress(ITERATIONS, 400);
    
    for (int i = 0; i < ITERATIONS; ++i) {
        // Create multiple tensors
        auto t1 = Tensor::FromScalar<float>(static_cast<float>(i));
        auto t2 = Tensor::FromVector<float>({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
        auto t3 = Tensor::Zeros<float>({10});
        
        // Clone one
        auto t4 = t2.Clone();
        
        // Run inference with scalar
        auto result = model.runner()
            .feed(input_op, t1)
            .fetch(output_op)
            .run_one();
        
        REQUIRE(result.valid());
        
        // Run inference with vector
        auto result2 = model.runner()
            .feed(input_op, t2)
            .fetch(output_op)
            .run_one();
        
        REQUIRE(result2.valid());
        REQUIRE(result2.num_elements() == 5);
        
        progress.tick();
    }
    
    progress.done();
}

// ============================================================================
// Soak Test: Error Paths (1,000 iterations)
// ============================================================================
//
// Purpose: Ensure error handling paths don't leak memory.
// Exceptions must clean up properly.

TEST(soak_error_paths_1k) {
    constexpr int ITERATIONS = 1'000;
    
    ProgressReporter progress(ITERATIONS, 200);
    
    for (int i = 0; i < ITERATIONS; ++i) {
        // Try to load nonexistent model - should throw
        try {
            auto model = Model::Load("/nonexistent/path/model_" + std::to_string(i));
            // Should not reach here
            REQUIRE(false);
        } catch (const Error&) {
            // Expected - error path must not leak
        }
        
        // Create tensor, then try invalid operations
        auto t = Tensor::FromScalar<float>(1.0f);
        
        // Try to read as wrong type - should throw
        try {
            auto view = t.read<int32_t>();
            REQUIRE(false);
        } catch (const Error&) {
            // Expected
        }
        
        // Try to read as wrong type for write
        try {
            auto view = t.write<double>();
            REQUIRE(false);
        } catch (const Error&) {
            // Expected
        }
        
        progress.tick();
    }
    
    progress.done();
}

// ============================================================================
// Soak Test: Rapid Session Creation (50 iterations)
// ============================================================================
//
// Purpose: Stress test session creation/destruction.
// Sessions hold significant TF resources.

TEST(soak_rapid_sessions_50) {
    constexpr int ITERATIONS = 50;
    
    ProgressReporter progress(ITERATIONS, 10);
    
    for (int i = 0; i < ITERATIONS; ++i) {
        // Create graph and session
        Graph graph;
        SessionOptions opts;
        Session session(graph, opts);
        
        REQUIRE(session.valid());
        REQUIRE(graph.valid());
        
        // Session and graph destroyed here
        progress.tick();
    }
    
    progress.done();
}

// ============================================================================
// Soak Test: Tensor Reshape (10,000 iterations)
// ============================================================================
//
// Purpose: Reshape creates views that share data.
// Must not leak the underlying tensor.

TEST(soak_tensor_reshape_10k) {
    constexpr int ITERATIONS = 10'000;
    
    ProgressReporter progress(ITERATIONS, 2000);
    
    for (int i = 0; i < ITERATIONS; ++i) {
        // Create 2D tensor
        auto t = Tensor::FromVector<float>({2, 3}, {1, 2, 3, 4, 5, 6});
        REQUIRE(t.valid());
        
        // Reshape to 1D
        auto reshaped = t.reshape({6});
        REQUIRE(reshaped.valid());
        REQUIRE(reshaped.num_elements() == 6);
        
        // Reshape to 3D
        auto reshaped2 = t.reshape({1, 2, 3});
        REQUIRE(reshaped2.valid());
        
        // Original and reshapes destroyed here
        progress.tick();
    }
    
    progress.done();
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== TensorFlowWrap Soak Tests (Real TF + ASan) ===\n";
    std::cout << "These tests run many iterations to detect memory leaks.\n";
    std::cout << "Ensure ASAN_OPTIONS=detect_leaks=1 is set.\n\n";
    
    auto start = std::chrono::steady_clock::now();
    
    // Tests run automatically via static initialization
    
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    
    std::cout << "\n=== Results: " << tests_passed << "/" << tests_run << " passed ===\n";
    std::cout << "Total time: " << duration.count() << " seconds\n";
    
    if (tests_passed == tests_run) {
        std::cout << "✓ All soak tests passed - no memory leaks detected by ASan\n";
    }
    
    return (tests_passed == tests_run) ? 0 : 1;
}
