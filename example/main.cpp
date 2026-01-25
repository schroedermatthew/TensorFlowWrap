// example/main.cpp
// Comprehensive example demonstrating the TensorFlow C++20 wrapper
//
// This example shows:
// 1. Basic graph construction and execution
// 2. Tensor views and lifetime safety
// 3. Multi-threaded session execution
// 4. Callback-based access
// 5. Error handling
// 6. Different tensor factories

#include "tf_wrap/core.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <thread>
#include <vector>

// ============================================================================
// Example 1: Basic graph construction and execution
// ============================================================================

void example_basic() {
    std::cout << "=== Example 1: Basic Graph Construction ===\n";
    
    // Create a simple graph: Const -> Identity
    tf_wrap::Graph graph;
    
    // Create a constant tensor [1, 8] with value 0.5
    std::vector<std::int64_t> shape = {1, 8};
    std::vector<float> values(8, 0.5f);
    auto const_tensor = tf_wrap::Tensor::FromVector<float>(shape, values);
    
    // Add Const operation - using rvalue chaining
    auto const_op = graph.NewOperation("Const", "my_constant")
        .SetAttrTensor("value", const_tensor.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Add Identity operation
    (void)graph.NewOperation("Identity", "output")
        .AddInput(const_op, 0)
        .Finish();
    
    // Create session and run
    tf_wrap::Session session(graph);
    auto result = session.Run("output", 0);
    
    // Access results safely
    auto view = result.read<float>();
    std::cout << "Output (" << view.size() << " elements): ";
    for (float x : view) {
        std::cout << x << " ";
    }
    std::cout << "\n\n";
}

// ============================================================================
// Example 2: Multi-threaded session execution
// ============================================================================

void example_threaded_session() {
    std::cout << "=== Example 2: Multi-threaded Session ===\n";
    
    // Build graph (single-threaded)
    tf_wrap::Graph graph;
    
    std::vector<std::int64_t> shape = {1, 4};
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};
    auto tensor = tf_wrap::Tensor::FromVector<float>(shape, values);
    
    auto const_op = graph.NewOperation("Const", "data")
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    (void)graph.NewOperation("Identity", "result")
        .AddInput(const_op, 0)
        .Finish();
    
    // Session::Run() is thread-safe (TensorFlow's guarantee)
    tf_wrap::Session session(graph);
    
    std::atomic<int> total_runs{0};
    constexpr int num_threads = 4;
    constexpr int runs_per_thread = 5;
    
    std::vector<std::thread> workers;
    for (int i = 0; i < num_threads; ++i) {
        workers.emplace_back([&session, &total_runs, i]() {
            for (int j = 0; j < runs_per_thread; ++j) {
                auto result = session.Run("result", 0);
                
                auto view = result.read<float>();
                float sum = std::accumulate(view.begin(), view.end(), 0.0f);
                
                if (j == 0) {
                    std::cout << "  Thread " << i << " got sum: " << sum << "\n";
                }
                ++total_runs;
            }
        });
    }
    
    for (auto& t : workers) {
        t.join();
    }
    
    std::cout << "Total successful runs: " << total_runs << "\n\n";
}

// ============================================================================
// Example 3: Tensor view lifetime safety
// ============================================================================

void example_view_lifetime() {
    std::cout << "=== Example 3: View Lifetime Safety ===\n";
    
    // Views keep tensor data alive even after Tensor object is destroyed
    std::optional<tf_wrap::Tensor::ReadView<float>> view;
    
    {
        auto tensor = tf_wrap::Tensor::FromVector<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
        view = tensor.read<float>();
        std::cout << "  Inside scope, tensor exists\n";
    }  // tensor destroyed here
    
    // View still valid - it holds shared_ptr to TensorState
    std::cout << "  Outside scope, view data: ";
    for (float x : *view) {
        std::cout << x << " ";
    }
    std::cout << "\n\n";
}

// ============================================================================
// Example 4: Callback-based access
// ============================================================================

void example_callback() {
    std::cout << "=== Example 4: Callback-based Access ===\n";
    
    std::vector<std::int64_t> shape = {100};
    std::vector<float> data(100);
    std::iota(data.begin(), data.end(), 0.0f);  // 0, 1, 2, ..., 99
    
    auto tensor = tf_wrap::Tensor::FromVector<float>(shape, data);
    
    // Callback receives a span - scope is clear
    float sum = tensor.with_read<float>([](std::span<const float> data) {
        return std::accumulate(data.begin(), data.end(), 0.0f);
    });
    
    std::cout << "  Sum of 0..99: " << sum << " (expected: 4950)\n";
    
    // Mutation with callback
    tensor.with_write<float>([](std::span<float> data) {
        for (float& x : data) {
            x *= 2.0f;  // Double all values
        }
    });
    
    float new_sum = tensor.with_read<float>([](std::span<const float> data) {
        return std::accumulate(data.begin(), data.end(), 0.0f);
    });
    
    std::cout << "  After doubling: " << new_sum << " (expected: 9900)\n\n";
}

// ============================================================================
// Example 5: Tensor factories
// ============================================================================

void example_factories() {
    std::cout << "=== Example 5: Tensor Factories ===\n";
    
    // FromVector - most common for initializing with data
    auto t1 = tf_wrap::Tensor::FromVector<float>({2, 2}, {1, 2, 3, 4});
    std::cout << "  FromVector: " << t1.num_elements() << " elements, dtype=" 
              << t1.dtype_name() << "\n";
    
    // FromScalar - for single values
    auto t2 = tf_wrap::Tensor::FromScalar<double>(3.14159);
    std::cout << "  FromScalar: " << t2.ToScalar<double>() << "\n";
    
    // Allocate - uninitialized (faster when you'll overwrite)
    auto t3 = tf_wrap::Tensor::Allocate<std::int32_t>({100});
    std::cout << "  Allocate: " << t3.byte_size() << " bytes\n";
    
    // Zeros - initialized to zero
    auto t4 = tf_wrap::Tensor::Zeros<float>({3, 3});
    auto view = t4.read<float>();
    bool all_zero = std::all_of(view.begin(), view.end(), 
                                 [](float x) { return x == 0.0f; });
    std::cout << "  Zeros: all zeros = " << (all_zero ? "yes" : "no") << "\n";
    
    // Clone - deep copy
    auto t5 = t1.Clone();
    std::cout << "  Clone: " << t5.num_elements() << " elements (copy of t1)\n\n";
}

// ============================================================================
// Example 6: Error handling
// ============================================================================

void example_errors() {
    std::cout << "=== Example 6: Error Handling ===\n";
    
    auto tensor = tf_wrap::Tensor::FromScalar<float>(1.0f);
    
    // Type mismatch throws
    try {
        auto view = tensor.read<int>();  // Wrong type!
    } catch (const std::runtime_error& e) {
        std::cout << "  Caught type mismatch: " << e.what() << "\n";
    }
    
    // Null tensor throws
    try {
        (void)tf_wrap::Tensor::FromRaw(nullptr);
    } catch (const std::invalid_argument& e) {
        std::cout << "  Caught null tensor: " << e.what() << "\n";
    }
    
    // Shape mismatch throws
    try {
        (void)tf_wrap::Tensor::FromVector<float>({2, 3}, {1, 2, 3});  // Need 6, got 3
    } catch (const std::invalid_argument& e) {
        std::cout << "  Caught shape mismatch: " << e.what() << "\n";
    }
    
    std::cout << "\n";
}

// ============================================================================
// Example 7: Graph with multiple operations
// ============================================================================

void example_complex_graph() {
    std::cout << "=== Example 7: Complex Graph ===\n";
    
    tf_wrap::Graph graph;
    
    auto tensor = tf_wrap::Tensor::FromScalar<float>(42.0f);
    
    // Build: Const(42) -> Add(x, x) -> Identity
    auto const_op = graph.NewOperation("Const", "x")
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto add_op = graph.NewOperation("AddV2", "doubled")
        .AddInput(const_op, 0)
        .AddInput(const_op, 0)  // x + x
        .SetAttrType("T", TF_FLOAT)
        .Finish();
    
    (void)graph.NewOperation("Identity", "output")
        .AddInput(add_op, 0)
        .Finish();
    
    // Run multiple times
    tf_wrap::Session session(graph);
    
    for (int i = 0; i < 3; ++i) {
        auto result = session.Run("output", 0);
        std::cout << "  Run " << i << ": " << result.ToScalar<float>() 
                  << " (expected: 84)\n";
    }
    
    std::cout << "\n";
}

// ============================================================================
// Example 8: Device enumeration
// ============================================================================

void example_devices() {
    std::cout << "=== Example 8: Device Enumeration ===\n";
    
    tf_wrap::Graph graph;
    auto tensor = tf_wrap::Tensor::FromScalar<float>(1.0f);
    
    auto const_op = graph.NewOperation("Const", "x")
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    (void)graph.NewOperation("Identity", "y")
        .AddInput(const_op, 0)
        .Finish();
    
    tf_wrap::Session session(graph);
    
    auto devices = session.ListDevices();
    std::cout << "  Available devices: " << devices.count() << "\n";
    
    for (int i = 0; i < devices.count(); ++i) {
        auto dev = devices.at(i);
        std::cout << "    " << dev.name << " (" << dev.type << ")";
        if (dev.memory_bytes > 0) {
            std::cout << " - " << (dev.memory_bytes / 1024 / 1024) << " MB";
        }
        std::cout << "\n";
    }
    
    std::cout << "  Has GPU: " << (session.HasGPU() ? "yes" : "no") << "\n\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "TensorFlow C++20 Wrapper Examples\n";
    std::cout << "==================================\n\n";
    
    try {
        example_basic();
        example_threaded_session();
        example_view_lifetime();
        example_callback();
        example_factories();
        example_errors();
        example_complex_graph();
        example_devices();
        
        std::cout << "All examples completed successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << "\n";
        return 1;
    }
}
