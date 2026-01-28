// test_soak_tf.cpp
// Soak tests for TensorFlowWrap with real TensorFlow
//
// These tests run many iterations to detect memory leaks when used with ASan.
// They exercise various code paths repeatedly to ensure RAII cleanup is correct.
//
// Run with: ASAN_OPTIONS=detect_leaks=1 ./test_soak_tf

#include "tf_wrap/core.hpp"
#include "tf_wrap/facade.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace tf_wrap;
using namespace tf_wrap::facade;

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

// Progress reporter for long-running tests
class ProgressReporter {
public:
    ProgressReporter(int total) : total_(total), last_reported_(0) {}
    
    void update(int current) {
        int pct = (current * 100) / total_;
        int milestone = (pct / 20) * 20;  // Report at 20%, 40%, 60%, 80%, 100%
        if (milestone > last_reported_ && milestone <= 100) {
            std::cout << "[" << current << "/" << total_ << "] " << std::flush;
            last_reported_ = milestone;
        }
    }
    
    void complete(int current) {
        std::cout << "[" << current << "/" << total_ << " complete] " << std::flush;
    }
    
private:
    int total_;
    int last_reported_;
};

// Global savedmodel path
static std::string g_savedmodel_path = "test_savedmodel";

// Helper: Find output operation (handles TF version differences)
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
// Soak Test: Model reload stress test
// ============================================================================

TEST(soak_model_reload_100) {
    const int iterations = 100;
    ProgressReporter progress(iterations);
    
    for (int i = 0; i < iterations; ++i) {
        progress.update(i);
        
        // Load model
        auto model = Model::Load(g_savedmodel_path);
        REQUIRE(model.valid());
        
        auto input_op = find_input_op(model.graph());
        auto output_op = find_output_op(model.graph());
        
        // Run a simple inference
        auto input = Tensor::FromScalar<float>(static_cast<float>(i));
        auto outputs = model.runner()
            .feed(input_op, input)
            .fetch(output_op)
            .run();
        REQUIRE(outputs.size() == 1);
        REQUIRE(outputs[0].valid());
        
        // Model destroyed here - should cleanup properly
    }
    
    progress.complete(iterations);
}

// ============================================================================
// Soak Test: Tensor creation/destruction churn
// ============================================================================

TEST(soak_tensor_churn_100k) {
    const int iterations = 100000;
    ProgressReporter progress(iterations);
    
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(1, 100);
    
    for (int i = 0; i < iterations; ++i) {
        progress.update(i);
        
        // Create tensors of various shapes
        int size = dist(rng);
        std::vector<float> data(size, static_cast<float>(i));
        
        auto tensor = Tensor::FromVector<float>({static_cast<std::int64_t>(size)}, data);
        REQUIRE(tensor.valid());
        REQUIRE(tensor.num_elements() == size);
        
        // Read back
        auto view = tensor.read<float>();
        REQUIRE(view.size() == static_cast<std::size_t>(size));
        
        // Tensor destroyed here
    }
    
    progress.complete(iterations);
}

// ============================================================================
// Soak Test: Large tensor allocation
// ============================================================================

TEST(soak_large_tensors_1k) {
    const int iterations = 1000;
    ProgressReporter progress(iterations);
    
    for (int i = 0; i < iterations; ++i) {
        progress.update(i);
        
        // Create a reasonably large tensor (1MB)
        const std::int64_t size = 256 * 1024;  // 256K floats = 1MB
        std::vector<float> data(size, 1.0f);
        
        auto tensor = Tensor::FromVector<float>({size}, data);
        REQUIRE(tensor.valid());
        REQUIRE(tensor.byte_size() == static_cast<std::size_t>(size) * sizeof(float));
        
        // Tensor destroyed here - should free 1MB
    }
    
    progress.complete(iterations);
}

// ============================================================================
// Soak Test: Runner reuse
// ============================================================================

TEST(soak_runner_reuse_5k) {
    const int iterations = 5000;
    ProgressReporter progress(iterations);
    
    auto model = Model::Load(g_savedmodel_path);
    REQUIRE(model.valid());
    
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    for (int i = 0; i < iterations; ++i) {
        progress.update(i);
        
        auto input = Tensor::FromScalar<float>(static_cast<float>(i));
        auto outputs = model.runner()
            .feed(input_op, input)
            .fetch(output_op)
            .run();
        
        REQUIRE(outputs.size() == 1);
        REQUIRE(outputs[0].valid());
    }
    
    progress.complete(iterations);
}

// ============================================================================
// Soak Test: String tensor creation (complex allocation)
// ============================================================================

TEST(soak_string_tensors_1k) {
    const int iterations = 1000;
    ProgressReporter progress(iterations);
    
    for (int i = 0; i < iterations; ++i) {
        progress.update(i);
        
        std::string str = "test_string_" + std::to_string(i);
        auto tensor = Tensor::FromString(str);
        
        REQUIRE(tensor.valid());
        REQUIRE(tensor.dtype() == TF_STRING);
        
        auto result = tensor.ToString();
        REQUIRE(result == str);
    }
    
    progress.complete(iterations);
}

// ============================================================================
// Soak Test: Mixed operations
// ============================================================================

TEST(soak_mixed_operations_2k) {
    const int iterations = 2000;
    ProgressReporter progress(iterations);
    
    auto model = Model::Load(g_savedmodel_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    for (int i = 0; i < iterations; ++i) {
        progress.update(i);
        
        // Alternate between different tensor types and operations
        switch (i % 4) {
            case 0: {
                auto t = Tensor::FromScalar<float>(1.0f);
                REQUIRE(t.dtype() == TF_FLOAT);
                break;
            }
            case 1: {
                auto t = Tensor::FromScalar<double>(2.0);
                REQUIRE(t.dtype() == TF_DOUBLE);
                break;
            }
            case 2: {
                auto t = Tensor::FromScalar<std::int32_t>(42);
                REQUIRE(t.dtype() == TF_INT32);
                break;
            }
            case 3: {
                // Run inference
                auto input = Tensor::FromScalar<float>(static_cast<float>(i));
                auto outputs = model.runner()
                    .feed(input_op, input)
                    .fetch(output_op)
                    .run();
                REQUIRE(outputs.size() == 1);
                break;
            }
        }
    }
    
    progress.complete(iterations);
}

// ============================================================================
// Soak Test: Graph operations
// ============================================================================

TEST(soak_graph_operations_10k) {
    const int iterations = 10000;
    ProgressReporter progress(iterations);
    
    for (int i = 0; i < iterations; ++i) {
        progress.update(i);
        
        // Create and destroy graphs repeatedly
        Graph graph;
        REQUIRE(graph.handle() != nullptr);
        
        // Test freeze
        graph.freeze();
        REQUIRE(graph.is_frozen());
        
        // Graph destroyed here
    }
    
    progress.complete(iterations);
}

// ============================================================================
// Soak Test: Rapid session creation
// ============================================================================

TEST(soak_rapid_sessions_50) {
    const int iterations = 50;
    ProgressReporter progress(iterations);
    
    for (int i = 0; i < iterations; ++i) {
        progress.update(i);
        
        Graph graph;
        SessionOptions opts;
        Session session(graph, opts);
        REQUIRE(session.handle() != nullptr);
        
        // Session destroyed here
    }
    
    progress.complete(iterations);
}

// ============================================================================
// Soak Test: Tensor reshape operations
// ============================================================================

TEST(soak_tensor_reshape_10k) {
    const int iterations = 10000;
    ProgressReporter progress(iterations);
    
    for (int i = 0; i < iterations; ++i) {
        progress.update(i);
        
        // Create tensor with known size
        std::vector<float> data(24, static_cast<float>(i));
        
        // Create with different shapes that have same total size
        switch (i % 4) {
            case 0: {
                auto t = Tensor::FromVector<float>({24}, data);
                REQUIRE(t.num_elements() == 24);
                break;
            }
            case 1: {
                auto t = Tensor::FromVector<float>({2, 12}, data);
                REQUIRE(t.num_elements() == 24);
                break;
            }
            case 2: {
                auto t = Tensor::FromVector<float>({3, 8}, data);
                REQUIRE(t.num_elements() == 24);
                break;
            }
            case 3: {
                auto t = Tensor::FromVector<float>({2, 3, 4}, data);
                REQUIRE(t.num_elements() == 24);
                break;
            }
        }
    }
    
    progress.complete(iterations);
}

// ============================================================================
// Soak Test: Copy tensor data
// ============================================================================

TEST(soak_tensor_copy_5k) {
    const int iterations = 5000;
    ProgressReporter progress(iterations);
    
    for (int i = 0; i < iterations; ++i) {
        progress.update(i);
        
        // Create source tensor
        std::vector<float> data(100, static_cast<float>(i));
        auto src = Tensor::FromVector<float>({100}, data);
        
        // Copy to vector
        auto copy = src.ToVector<float>();
        REQUIRE(copy.size() == 100);
        REQUIRE(copy[0] == static_cast<float>(i));
    }
    
    progress.complete(iterations);
}

// ============================================================================
// Soak Test: Status operations
// ============================================================================

TEST(soak_status_operations_50k) {
    const int iterations = 50000;
    ProgressReporter progress(iterations);
    
    for (int i = 0; i < iterations; ++i) {
        progress.update(i);
        
        // Create and destroy status objects
        Status st;
        REQUIRE(st.ok());
        
        // Set error and check
        st.set(TF_INVALID_ARGUMENT, "test error");
        REQUIRE(!st.ok());
        REQUIRE(st.code() == TF_INVALID_ARGUMENT);
        
        // Reset
        st.reset();
        REQUIRE(st.ok());
    }
    
    progress.complete(iterations);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc > 1) {
        g_savedmodel_path = argv[1];
    }
    
    auto start = std::chrono::steady_clock::now();
    
    std::cout << "=== TensorFlowWrap Soak Tests (Real TF + ASan) ===" << std::endl;
    std::cout << "These tests run many iterations to detect memory leaks." << std::endl;
    std::cout << "Ensure ASAN_OPTIONS=detect_leaks=1 is set." << std::endl;
    std::cout << std::endl;
    
    // Tests run automatically via static initialization
    
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    
    std::cout << std::endl;
    std::cout << "=== Results: " << tests_passed << "/" << tests_run << " passed ===" << std::endl;
    std::cout << "Total time: " << elapsed << " seconds" << std::endl;
    
    return (tests_passed == tests_run) ? 0 : 1;
}
