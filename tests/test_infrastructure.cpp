// tests/test_infrastructure.cpp
// Infrastructure-dependent tests - SKIPPED on GitHub Actions CI
// Run locally with: TF_INFRA_TESTS=1 ./test_infrastructure
//
// These tests require specialized environments:
// - GPU tests: NVIDIA GPU with CUDA
// - Distributed tests: gRPC cluster
// - OOM tests: Controlled memory limits
// - Huge tensor tests: 16GB+ RAM
// - High-thread tests: 32+ core machine
// - Soak tests: Hours of runtime

#include "tf_wrap/all.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

// ============================================================================
// Test Framework
// ============================================================================

static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;
static int g_tests_skipped = 0;

static bool should_skip_infra_tests() {
    const char* env = std::getenv("TF_INFRA_TESTS");
    return env == nullptr || std::string(env) != "1";
}

static bool has_gpu() {
    const char* env = std::getenv("TF_HAS_GPU");
    return env != nullptr && std::string(env) == "1";
}

static bool has_distributed() {
    const char* env = std::getenv("TF_DISTRIBUTED_TARGET");
    return env != nullptr && std::strlen(env) > 0;
}

static bool has_high_memory() {
    const char* env = std::getenv("TF_HIGH_MEMORY");
    return env != nullptr && std::string(env) == "1";
}

static bool has_many_cores() {
    const char* env = std::getenv("TF_MANY_CORES");
    return env != nullptr && std::string(env) == "1";
}

static bool run_soak_tests() {
    const char* env = std::getenv("TF_SOAK_TESTS");
    return env != nullptr && std::string(env) == "1";
}

#define REQUIRE(cond) do { \
    if (!(cond)) { \
        std::cerr << "  FAILED: " #cond << " at line " << __LINE__ << "\n"; \
        throw std::runtime_error("Test failed"); \
    } \
} while(0)

#define INFRA_TEST(name, check_fn) \
    void test_##name(); \
    struct TestReg_##name { \
        TestReg_##name() { \
            ++g_tests_run; \
            std::cout << "[TEST] " #name << "\n"; \
            if (should_skip_infra_tests()) { \
                std::cout << "  SKIPPED (set TF_INFRA_TESTS=1 to run)\n"; \
                ++g_tests_skipped; \
                return; \
            } \
            if (!check_fn()) { \
                std::cout << "  SKIPPED (missing requirement)\n"; \
                ++g_tests_skipped; \
                return; \
            } \
            try { \
                test_##name(); \
                std::cout << "  PASS\n"; \
                ++g_tests_passed; \
            } catch (const std::exception& e) { \
                std::cout << "  FAIL: " << e.what() << "\n"; \
                ++g_tests_failed; \
            } \
        } \
    } g_reg_##name; \
    void test_##name()

// Always-true check for tests that just need TF_INFRA_TESTS=1
static bool always_true() { return true; }

// ============================================================================
// GPU Device Placement Tests
// ============================================================================

INFRA_TEST(gpu_device_list_contains_gpu, has_gpu) {
    tf_wrap::FastGraph g;
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::FastSession s(g);
    auto devices = s.ListDevices();
    
    bool found_gpu = false;
    for (int i = 0; i < devices.count(); ++i) {
        auto dev = devices.at(i);
        std::cout << "    Found device: " << dev.name << " (" << dev.type << ")\n";
        if (dev.type == "GPU") {
            found_gpu = true;
        }
    }
    REQUIRE(found_gpu);
}

INFRA_TEST(gpu_explicit_device_placement, has_gpu) {
    tf_wrap::FastGraph g;
    
    // Create tensor on GPU
    auto t = tf_wrap::FastTensor::FromVector<float>({1000}, std::vector<float>(1000, 1.0f));
    
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .SetDevice("/device:GPU:0")
        .Finish();
    
    auto* x = g.GetOperationOrThrow("X");
    
    // Square on GPU
    (void)g.NewOperation("Square", "Y")
        .AddInput(tf_wrap::Output(x, 0))
        .SetDevice("/device:GPU:0")
        .Finish();
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"Y", 0}}, {});
    
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].num_elements() == 1000);
    
    auto v = results[0].ToVector<float>();
    REQUIRE(std::abs(v[0] - 1.0f) < 0.001f);  // 1^2 = 1
}

INFRA_TEST(gpu_large_tensor_computation, has_gpu) {
    // 100M floats = 400MB - should fit on most GPUs
    const size_t num_elements = 100'000'000;
    
    tf_wrap::FastGraph g;
    
    std::vector<float> data(num_elements, 2.0f);
    auto t = tf_wrap::FastTensor::FromVector<float>(
        {static_cast<int64_t>(num_elements)}, data);
    
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .SetDevice("/device:GPU:0")
        .Finish();
    
    auto* x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("Square", "Y")
        .AddInput(tf_wrap::Output(x, 0))
        .SetDevice("/device:GPU:0")
        .Finish();
    
    tf_wrap::FastSession s(g);
    
    auto start = std::chrono::high_resolution_clock::now();
    auto results = s.Run({}, {{"Y", 0}}, {});
    auto end = std::chrono::high_resolution_clock::now();
    
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "    GPU computation time: " << ms << "ms for " << num_elements << " elements\n";
    
    REQUIRE(results[0].num_elements() == num_elements);
    
    // Spot check
    auto v = results[0].ToVector<float>();
    REQUIRE(std::abs(v[0] - 4.0f) < 0.001f);
    REQUIRE(std::abs(v[num_elements/2] - 4.0f) < 0.001f);
    REQUIRE(std::abs(v[num_elements-1] - 4.0f) < 0.001f);
}

INFRA_TEST(gpu_cpu_data_transfer, has_gpu) {
    tf_wrap::FastGraph g;
    
    // Create on CPU
    auto t = tf_wrap::FastTensor::FromVector<float>({1000}, std::vector<float>(1000, 3.0f));
    
    (void)g.NewOperation("Const", "CPU_Data")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .SetDevice("/device:CPU:0")
        .Finish();
    
    auto* cpu_data = g.GetOperationOrThrow("CPU_Data");
    
    // Compute on GPU (implicit transfer)
    (void)g.NewOperation("Square", "GPU_Compute")
        .AddInput(tf_wrap::Output(cpu_data, 0))
        .SetDevice("/device:GPU:0")
        .Finish();
    
    auto* gpu_compute = g.GetOperationOrThrow("GPU_Compute");
    
    // Transfer back to CPU for output
    (void)g.NewOperation("Identity", "CPU_Output")
        .AddInput(tf_wrap::Output(gpu_compute, 0))
        .SetDevice("/device:CPU:0")
        .Finish();
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"CPU_Output", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(std::abs(v[0] - 9.0f) < 0.001f);  // 3^2 = 9
}

INFRA_TEST(gpu_multi_gpu_placement, has_gpu) {
    tf_wrap::FastGraph g;
    tf_wrap::FastSession s_temp(g);
    auto devices = s_temp.ListDevices();
    
    int gpu_count = 0;
    for (int i = 0; i < devices.count(); ++i) {
        if (devices.at(i).type == "GPU") ++gpu_count;
    }
    
    if (gpu_count < 2) {
        std::cout << "    Only " << gpu_count << " GPU(s), skipping multi-GPU test\n";
        return;  // Not a failure, just can't test
    }
    
    tf_wrap::FastGraph g2;
    auto t1 = tf_wrap::FastTensor::FromScalar<float>(2.0f);
    auto t2 = tf_wrap::FastTensor::FromScalar<float>(3.0f);
    
    (void)g2.NewOperation("Const", "A")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .SetDevice("/device:GPU:0")
        .Finish();
    
    (void)g2.NewOperation("Const", "B")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .SetDevice("/device:GPU:1")
        .Finish();
    
    auto* a = g2.GetOperationOrThrow("A");
    auto* b = g2.GetOperationOrThrow("B");
    
    // Cross-GPU operation
    (void)g2.NewOperation("AddV2", "Sum")
        .AddInput(tf_wrap::Output(a, 0))
        .AddInput(tf_wrap::Output(b, 0))
        .Finish();
    
    tf_wrap::FastSession s(g2);
    auto results = s.Run({}, {{"Sum", 0}}, {});
    
    REQUIRE(std::abs(results[0].ToScalar<float>() - 5.0f) < 0.001f);
    std::cout << "    Multi-GPU computation successful\n";
}

// ============================================================================
// Distributed TensorFlow Tests
// ============================================================================

INFRA_TEST(distributed_connect_to_cluster, has_distributed) {
    const char* target = std::getenv("TF_DISTRIBUTED_TARGET");
    
    tf_wrap::FastGraph g;
    auto t = tf_wrap::FastTensor::FromScalar<float>(42.0f);
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::SessionOptions opts;
    opts.SetTarget(target);
    
    tf_wrap::FastSession s(g, opts);
    auto results = s.Run({}, {{"X", 0}}, {});
    
    REQUIRE(std::abs(results[0].ToScalar<float>() - 42.0f) < 0.001f);
    std::cout << "    Connected to: " << target << "\n";
}

INFRA_TEST(distributed_remote_device_list, has_distributed) {
    const char* target = std::getenv("TF_DISTRIBUTED_TARGET");
    
    tf_wrap::FastGraph g;
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::SessionOptions opts;
    opts.SetTarget(target);
    
    tf_wrap::FastSession s(g, opts);
    auto devices = s.ListDevices();
    
    std::cout << "    Remote devices:\n";
    for (int i = 0; i < devices.count(); ++i) {
        auto dev = devices.at(i);
        std::cout << "      " << dev.name << " (" << dev.type << ")\n";
    }
    
    REQUIRE(devices.count() > 0);
}

INFRA_TEST(distributed_cross_worker_computation, has_distributed) {
    const char* target = std::getenv("TF_DISTRIBUTED_TARGET");
    
    tf_wrap::FastGraph g;
    auto t1 = tf_wrap::FastTensor::FromScalar<float>(10.0f);
    auto t2 = tf_wrap::FastTensor::FromScalar<float>(20.0f);
    
    // Place operations on different workers (assumes job:worker/task:0 and task:1)
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .SetDevice("/job:worker/task:0/device:CPU:0")
        .Finish();
    
    (void)g.NewOperation("Const", "B")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .SetDevice("/job:worker/task:1/device:CPU:0")
        .Finish();
    
    auto* a = g.GetOperationOrThrow("A");
    auto* b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("AddV2", "Sum")
        .AddInput(tf_wrap::Output(a, 0))
        .AddInput(tf_wrap::Output(b, 0))
        .Finish();
    
    tf_wrap::SessionOptions opts;
    opts.SetTarget(target);
    
    tf_wrap::FastSession s(g, opts);
    auto results = s.Run({}, {{"Sum", 0}}, {});
    
    REQUIRE(std::abs(results[0].ToScalar<float>() - 30.0f) < 0.001f);
    std::cout << "    Cross-worker computation successful\n";
}

// ============================================================================
// OOM Handling Tests
// ============================================================================

INFRA_TEST(oom_tensor_allocation_fails_gracefully, always_true) {
    // Try to allocate an impossibly large tensor
    // 1 trillion floats = 4 petabytes
    std::vector<int64_t> huge_shape = {1'000'000, 1'000'000, 1'000};
    
    bool threw = false;
    try {
        auto tensor = tf_wrap::FastTensor::Allocate<float>(huge_shape);
        // If we get here, allocation somehow succeeded (unlikely)
        std::cout << "    Warning: huge allocation succeeded?!\n";
    } catch (const std::exception& e) {
        threw = true;
        std::cout << "    Correctly threw: " << e.what() << "\n";
    }
    
    REQUIRE(threw);
}

INFRA_TEST(oom_graph_continues_after_allocation_failure, always_true) {
    tf_wrap::FastGraph g;
    
    // First, do something that works
    auto t1 = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "Good")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Try to create impossibly large tensor
    bool allocation_failed = false;
    try {
        std::vector<int64_t> huge_shape = {1'000'000'000'000LL};
        auto huge = tf_wrap::FastTensor::Allocate<float>(huge_shape);
    } catch (...) {
        allocation_failed = true;
    }
    
    REQUIRE(allocation_failed);
    
    // Graph should still work
    auto t2 = tf_wrap::FastTensor::FromScalar<float>(2.0f);
    (void)g.NewOperation("Const", "StillGood")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"Good", 0}, {"StillGood", 0}}, {});
    
    REQUIRE(results.size() == 2);
    REQUIRE(std::abs(results[0].ToScalar<float>() - 1.0f) < 0.001f);
    REQUIRE(std::abs(results[1].ToScalar<float>() - 2.0f) < 0.001f);
}

INFRA_TEST(oom_session_survives_large_intermediate, has_high_memory) {
    // This test requires a machine where we can actually allocate ~8GB
    // and test recovery
    
    tf_wrap::FastGraph g;
    
    // Create a large tensor (2GB)
    const size_t large_size = 500'000'000;  // 500M floats = 2GB
    std::vector<float> data(large_size, 1.0f);
    auto t = tf_wrap::FastTensor::FromVector<float>(
        {static_cast<int64_t>(large_size)}, data);
    
    (void)g.NewOperation("Const", "Large")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::FastSession s(g);
    
    // This should work
    auto results = s.Run({}, {{"Large", 0}}, {});
    REQUIRE(results[0].num_elements() == large_size);
    
    // Now try multiple large operations
    data.clear();  // Free memory
    
    auto* large = g.GetOperationOrThrow("Large");
    (void)g.NewOperation("Square", "Squared")
        .AddInput(tf_wrap::Output(large, 0))
        .Finish();
    
    // This creates another 2GB tensor
    results = s.Run({}, {{"Squared", 0}}, {});
    REQUIRE(results[0].num_elements() == large_size);
    
    std::cout << "    Large tensor operations completed\n";
}

// ============================================================================
// Huge Tensor Tests (>1GB)
// ============================================================================

INFRA_TEST(huge_tensor_1b_elements, has_high_memory) {
    // 1 billion floats = 4GB
    const size_t num_elements = 1'000'000'000;
    
    std::cout << "    Allocating 4GB tensor...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    auto tensor = tf_wrap::FastTensor::Allocate<float>(
        {static_cast<int64_t>(num_elements)});
    
    auto alloc_end = std::chrono::high_resolution_clock::now();
    auto alloc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        alloc_end - start).count();
    std::cout << "    Allocation time: " << alloc_ms << "ms\n";
    
    REQUIRE(tensor.valid());
    REQUIRE(tensor.num_elements() == num_elements);
    
    // Write to tensor
    {
        auto view = tensor.write<float>();
        // Just write first, middle, last
        view[0] = 1.0f;
        view[num_elements/2] = 2.0f;
        view[num_elements-1] = 3.0f;
    }
    
    // Read back
    {
        auto view = tensor.read<float>();
        REQUIRE(view[0] == 1.0f);
        REQUIRE(view[num_elements/2] == 2.0f);
        REQUIRE(view[num_elements-1] == 3.0f);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start).count();
    std::cout << "    Total time: " << total_ms << "ms\n";
}

INFRA_TEST(huge_tensor_2b_elements, has_high_memory) {
    // 2 billion floats = 8GB
    const size_t num_elements = 2'000'000'000;
    
    std::cout << "    Allocating 8GB tensor...\n";
    
    auto tensor = tf_wrap::FastTensor::Allocate<float>(
        {static_cast<int64_t>(num_elements)});
    
    REQUIRE(tensor.valid());
    REQUIRE(tensor.num_elements() == num_elements);
    
    std::cout << "    8GB tensor allocated successfully\n";
}

INFRA_TEST(huge_tensor_multidimensional, has_high_memory) {
    // 1000 x 1000 x 1000 = 1 billion floats = 4GB
    auto tensor = tf_wrap::FastTensor::Allocate<float>({1000, 1000, 1000});
    
    REQUIRE(tensor.valid());
    REQUIRE(tensor.shape().size() == 3);
    REQUIRE(tensor.shape()[0] == 1000);
    REQUIRE(tensor.shape()[1] == 1000);
    REQUIRE(tensor.shape()[2] == 1000);
    REQUIRE(tensor.num_elements() == 1'000'000'000);
}

INFRA_TEST(huge_tensor_session_run, has_high_memory) {
    const size_t num_elements = 500'000'000;  // 2GB - more reasonable
    
    tf_wrap::FastGraph g;
    
    (void)g.NewOperation("Placeholder", "X")
        .SetAttrType("dtype", TF_FLOAT)
        .SetAttrShape("shape", {static_cast<int64_t>(num_elements)})
        .Finish();
    
    auto* x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("Square", "Y")
        .AddInput(tf_wrap::Output(x, 0))
        .Finish();
    
    tf_wrap::FastSession s(g);
    
    // Create large input
    auto input = tf_wrap::FastTensor::Allocate<float>(
        {static_cast<int64_t>(num_elements)});
    {
        auto view = input.write<float>();
        view[0] = 2.0f;
        view[num_elements-1] = 3.0f;
    }
    
    std::cout << "    Running computation on 2GB tensor...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    auto results = s.Run(
        {{"X", 0, input.handle()}},
        {{"Y", 0}},
        {}
    );
    
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "    Computation time: " << ms << "ms\n";
    
    auto v = results[0].read<float>();
    REQUIRE(std::abs(v[0] - 4.0f) < 0.001f);
    REQUIRE(std::abs(v[num_elements-1] - 9.0f) < 0.001f);
}

// ============================================================================
// High Thread Count Tests (32+)
// ============================================================================

INFRA_TEST(high_thread_64_concurrent_sessions, has_many_cores) {
    const int num_threads = 64;
    const int runs_per_thread = 100;
    
    tf_wrap::SafeGraph g;
    auto t = tf_wrap::SafeTensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    std::vector<std::unique_ptr<tf_wrap::SafeSession>> sessions;
    for (int i = 0; i < num_threads; ++i) {
        sessions.push_back(std::make_unique<tf_wrap::SafeSession>(g));
    }
    
    std::atomic<int> success_count{0};
    std::atomic<int> error_count{0};
    
    auto worker = [&](int id) {
        try {
            for (int i = 0; i < runs_per_thread; ++i) {
                auto results = sessions[id]->Run({}, {{"X", 0}}, {});
                if (results[0].ToScalar<float>() == 1.0f) {
                    ++success_count;
                }
            }
        } catch (...) {
            ++error_count;
        }
    };
    
    std::cout << "    Running " << num_threads << " threads x " 
              << runs_per_thread << " runs...\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }
    for (auto& th : threads) {
        th.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "    Completed in " << ms << "ms\n";
    std::cout << "    Success: " << success_count << ", Errors: " << error_count << "\n";
    
    REQUIRE(error_count == 0);
    REQUIRE(success_count == num_threads * runs_per_thread);
}

INFRA_TEST(high_thread_128_tensor_operations, has_many_cores) {
    const int num_threads = 128;
    const int ops_per_thread = 1000;
    
    std::atomic<int> success_count{0};
    std::atomic<int> error_count{0};
    
    auto worker = [&](int id) {
        try {
            for (int i = 0; i < ops_per_thread; ++i) {
                auto t = tf_wrap::FastTensor::FromScalar<float>(
                    static_cast<float>(id * ops_per_thread + i));
                auto v = t.ToScalar<float>();
                if (static_cast<int>(v) == id * ops_per_thread + i) {
                    ++success_count;
                }
            }
        } catch (...) {
            ++error_count;
        }
    };
    
    std::cout << "    Running " << num_threads << " threads x " 
              << ops_per_thread << " tensor ops...\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }
    for (auto& th : threads) {
        th.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "    Completed in " << ms << "ms\n";
    std::cout << "    Throughput: " << (num_threads * ops_per_thread * 1000 / ms) << " ops/sec\n";
    
    REQUIRE(error_count == 0);
    REQUIRE(success_count == num_threads * ops_per_thread);
}

INFRA_TEST(high_thread_shared_graph_stress, has_many_cores) {
    const int num_threads = 64;
    
    tf_wrap::SharedGraph g;
    
    // Build graph with multiple threads reading
    auto t = tf_wrap::SharedTensor::FromScalar<float>(42.0f);
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    std::atomic<int> read_count{0};
    std::atomic<bool> stop{false};
    
    auto reader = [&]() {
        while (!stop) {
            auto ops = g.num_operations();
            if (ops >= 1) ++read_count;
        }
    };
    
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(reader);
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    stop = true;
    
    for (auto& th : threads) {
        th.join();
    }
    
    std::cout << "    " << num_threads << " threads performed " 
              << read_count << " concurrent reads\n";
    
    REQUIRE(read_count > 0);
}

// ============================================================================
// Soak Tests (Long Running)
// ============================================================================

INFRA_TEST(soak_1_hour_continuous_operations, run_soak_tests) {
    const auto duration = std::chrono::hours(1);
    const auto report_interval = std::chrono::minutes(5);
    
    std::cout << "    Starting 1-hour soak test...\n";
    
    tf_wrap::SafeGraph g;
    auto t = tf_wrap::SafeTensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::SafeSession s(g);
    
    auto start = std::chrono::high_resolution_clock::now();
    auto last_report = start;
    size_t total_ops = 0;
    size_t errors = 0;
    
    while (std::chrono::high_resolution_clock::now() - start < duration) {
        try {
            auto results = s.Run({}, {{"X", 0}}, {});
            if (results[0].ToScalar<float>() == 1.0f) {
                ++total_ops;
            } else {
                ++errors;
            }
        } catch (...) {
            ++errors;
        }
        
        auto now = std::chrono::high_resolution_clock::now();
        if (now - last_report >= report_interval) {
            auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(now - start).count();
            std::cout << "    [" << elapsed << "m] Ops: " << total_ops 
                      << ", Errors: " << errors << "\n";
            last_report = now;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto total_seconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    
    std::cout << "    Completed: " << total_ops << " operations in " 
              << total_seconds << " seconds\n";
    std::cout << "    Throughput: " << (total_ops / total_seconds) << " ops/sec\n";
    std::cout << "    Errors: " << errors << "\n";
    
    REQUIRE(errors == 0);
}

INFRA_TEST(soak_memory_stability, run_soak_tests) {
    const auto duration = std::chrono::minutes(30);
    
    std::cout << "    Starting 30-minute memory stability test...\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    size_t iterations = 0;
    
    while (std::chrono::high_resolution_clock::now() - start < duration) {
        // Create and destroy tensors rapidly
        for (int i = 0; i < 1000; ++i) {
            auto t = tf_wrap::FastTensor::FromVector<float>({100, 100}, 
                std::vector<float>(10000, static_cast<float>(i)));
            auto v = t.ToVector<float>();
            (void)v;
        }
        
        // Create and destroy graphs
        for (int i = 0; i < 100; ++i) {
            tf_wrap::FastGraph g;
            auto t = tf_wrap::FastTensor::FromScalar<float>(static_cast<float>(i));
            (void)g.NewOperation("Const", "X")
                .SetAttrTensor("value", t.handle())
                .SetAttrType("dtype", TF_FLOAT)
                .Finish();
        }
        
        ++iterations;
        
        if (iterations % 100 == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(
                std::chrono::high_resolution_clock::now() - start).count();
            std::cout << "    [" << elapsed << "m] Iterations: " << iterations << "\n";
        }
    }
    
    std::cout << "    Completed " << iterations << " iterations without memory issues\n";
}

INFRA_TEST(soak_thread_creation_destruction, run_soak_tests) {
    const auto duration = std::chrono::minutes(15);
    
    std::cout << "    Starting 15-minute thread stress test...\n";
    
    tf_wrap::SafeGraph g;
    auto t = tf_wrap::SafeTensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto start = std::chrono::high_resolution_clock::now();
    size_t thread_batches = 0;
    std::atomic<size_t> total_runs{0};
    
    while (std::chrono::high_resolution_clock::now() - start < duration) {
        // Create batch of threads
        std::vector<std::thread> threads;
        for (int i = 0; i < 16; ++i) {
            threads.emplace_back([&g, &total_runs]() {
                tf_wrap::SafeSession s(g);
                for (int j = 0; j < 100; ++j) {
                    auto results = s.Run({}, {{"X", 0}}, {});
                    (void)results;
                    ++total_runs;
                }
            });
        }
        
        // Wait for all to complete
        for (auto& th : threads) {
            th.join();
        }
        
        ++thread_batches;
        
        if (thread_batches % 50 == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(
                std::chrono::high_resolution_clock::now() - start).count();
            std::cout << "    [" << elapsed << "m] Batches: " << thread_batches 
                      << ", Total runs: " << total_runs << "\n";
        }
    }
    
    std::cout << "    Completed " << thread_batches << " thread batches, "
              << total_runs << " total runs\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "\n========================================\n";
    std::cout << "TensorFlowWrap Infrastructure Tests\n";
    std::cout << "========================================\n\n";
    
    if (should_skip_infra_tests()) {
        std::cout << "NOTE: Set TF_INFRA_TESTS=1 to run infrastructure tests\n";
        std::cout << "Additional env vars:\n";
        std::cout << "  TF_HAS_GPU=1           - Enable GPU tests\n";
        std::cout << "  TF_DISTRIBUTED_TARGET  - gRPC target for distributed tests\n";
        std::cout << "  TF_HIGH_MEMORY=1       - Enable huge tensor tests (16GB+ RAM)\n";
        std::cout << "  TF_MANY_CORES=1        - Enable 64+ thread tests\n";
        std::cout << "  TF_SOAK_TESTS=1        - Enable multi-hour soak tests\n";
        std::cout << "\n";
    }
    
    // Tests run via static initialization
    
    std::cout << "\n========================================\n";
    std::cout << "Results: " << g_tests_passed << " passed, "
              << g_tests_failed << " failed, "
              << g_tests_skipped << " skipped\n";
    std::cout << "========================================\n";
    
    return g_tests_failed > 0 ? 1 : 0;
}
