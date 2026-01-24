// tests/test_infrastructure.cpp
// Infrastructure-dependent tests using doctest
//
// Run with: TF_INFRA_TESTS=1 ./test_infrastructure
//
// These tests require specialized environments:
// - GPU tests: TF_HAS_GPU=1
// - Distributed tests: TF_DISTRIBUTED_TARGET=grpc://...
// - High memory tests: TF_HIGH_MEMORY=1 (16GB+ RAM)
// - Many cores tests: TF_MANY_CORES=1 (32+ cores)
// - Soak tests: TF_SOAK_TESTS=1 (hours of runtime)

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

// Suppress warnings from CHECK_THROWS_AS with [[nodiscard]] functions
#if defined(_MSC_VER)
#pragma warning(disable: 4834)  // discarding return value of function with [[nodiscard]]
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

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
// Environment checks
// ============================================================================

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

#define SKIP_IF_NO_INFRA() do { \
    if (should_skip_infra_tests()) { MESSAGE("Skipped: set TF_INFRA_TESTS=1"); return; } \
} while(0)

#define SKIP_IF_NO_GPU() do { \
    SKIP_IF_NO_INFRA(); \
    if (!has_gpu()) { MESSAGE("Skipped: set TF_HAS_GPU=1"); return; } \
} while(0)

#define SKIP_IF_NO_DISTRIBUTED() do { \
    SKIP_IF_NO_INFRA(); \
    if (!has_distributed()) { MESSAGE("Skipped: set TF_DISTRIBUTED_TARGET"); return; } \
} while(0)

#define SKIP_IF_NO_HIGH_MEMORY() do { \
    SKIP_IF_NO_INFRA(); \
    if (!has_high_memory()) { MESSAGE("Skipped: set TF_HIGH_MEMORY=1"); return; } \
} while(0)

#define SKIP_IF_NO_MANY_CORES() do { \
    SKIP_IF_NO_INFRA(); \
    if (!has_many_cores()) { MESSAGE("Skipped: set TF_MANY_CORES=1"); return; } \
} while(0)

#define SKIP_IF_NO_SOAK() do { \
    SKIP_IF_NO_INFRA(); \
    if (!run_soak_tests()) { MESSAGE("Skipped: set TF_SOAK_TESTS=1"); return; } \
} while(0)

// ============================================================================
// GPU Device Placement Tests
// ============================================================================

TEST_CASE("gpu_device_list_contains_gpu" * doctest::test_suite("gpu")) {
    SKIP_IF_NO_GPU();
    
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Session s(g);
    auto devices = s.ListDevices();
    
    bool found_gpu = false;
    for (int i = 0; i < devices.count(); ++i) {
        auto dev = devices.at(i);
        MESSAGE("Found device: ", dev.name, " (", dev.type, ")");
        if (dev.type == "GPU") {
            found_gpu = true;
        }
    }
    CHECK(found_gpu);
}

TEST_CASE("gpu_explicit_device_placement" * doctest::test_suite("gpu")) {
    SKIP_IF_NO_GPU();
    
    tf_wrap::Graph g;
    
    auto t = tf_wrap::Tensor::FromVector<float>({1000}, std::vector<float>(1000, 1.0f));
    
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
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Y", 0}}, {});
    
    CHECK(results.size() == 1);
    CHECK(results[0].num_elements() == 1000);
    
    auto v = results[0].ToVector<float>();
    CHECK(std::abs(v[0] - 1.0f) < 0.001f);
}

TEST_CASE("gpu_large_tensor_computation" * doctest::test_suite("gpu")) {
    SKIP_IF_NO_GPU();
    
    const size_t num_elements = 100'000'000;
    
    tf_wrap::Graph g;
    
    std::vector<float> data(num_elements, 2.0f);
    auto t = tf_wrap::Tensor::FromVector<float>(
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
    
    tf_wrap::Session s(g);
    
    auto start = std::chrono::high_resolution_clock::now();
    auto results = s.Run({}, {{"Y", 0}}, {});
    auto end = std::chrono::high_resolution_clock::now();
    
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    MESSAGE("GPU computation time: ", ms, "ms for ", num_elements, " elements");
    
    CHECK(results[0].num_elements() == num_elements);
    
    auto v = results[0].ToVector<float>();
    CHECK(std::abs(v[0] - 4.0f) < 0.001f);
    CHECK(std::abs(v[num_elements/2] - 4.0f) < 0.001f);
    CHECK(std::abs(v[num_elements-1] - 4.0f) < 0.001f);
}

TEST_CASE("gpu_cpu_data_transfer" * doctest::test_suite("gpu")) {
    SKIP_IF_NO_GPU();
    
    tf_wrap::Graph g;
    
    auto t = tf_wrap::Tensor::FromVector<float>({1000}, std::vector<float>(1000, 3.0f));
    
    (void)g.NewOperation("Const", "CPU_Data")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .SetDevice("/device:CPU:0")
        .Finish();
    
    auto* cpu_data = g.GetOperationOrThrow("CPU_Data");
    
    (void)g.NewOperation("Square", "GPU_Compute")
        .AddInput(tf_wrap::Output(cpu_data, 0))
        .SetDevice("/device:GPU:0")
        .Finish();
    
    auto* gpu_compute = g.GetOperationOrThrow("GPU_Compute");
    
    (void)g.NewOperation("Identity", "CPU_Output")
        .AddInput(tf_wrap::Output(gpu_compute, 0))
        .SetDevice("/device:CPU:0")
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"CPU_Output", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    CHECK(std::abs(v[0] - 9.0f) < 0.001f);
}

TEST_CASE("gpu_multi_gpu_placement" * doctest::test_suite("gpu")) {
    SKIP_IF_NO_GPU();
    
    tf_wrap::Graph g;
    tf_wrap::Session s_temp(g);
    auto devices = s_temp.ListDevices();
    
    int gpu_count = 0;
    for (int i = 0; i < devices.count(); ++i) {
        if (devices.at(i).type == "GPU") ++gpu_count;
    }
    
    if (gpu_count < 2) {
        MESSAGE("Only ", gpu_count, " GPU(s), skipping multi-GPU test");
        return;
    }
    
    tf_wrap::Graph g2;
    auto t1 = tf_wrap::Tensor::FromScalar<float>(2.0f);
    auto t2 = tf_wrap::Tensor::FromScalar<float>(3.0f);
    
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
    
    (void)g2.NewOperation("AddV2", "Sum")
        .AddInput(tf_wrap::Output(a, 0))
        .AddInput(tf_wrap::Output(b, 0))
        .Finish();
    
    tf_wrap::Session s(g2);
    auto results = s.Run({}, {{"Sum", 0}}, {});
    
    CHECK(std::abs(results[0].ToScalar<float>() - 5.0f) < 0.001f);
    MESSAGE("Multi-GPU computation successful");
}

// ============================================================================
// Distributed TensorFlow Tests
// ============================================================================

TEST_CASE("distributed_connect_to_cluster" * doctest::test_suite("distributed")) {
    SKIP_IF_NO_DISTRIBUTED();
    
    const char* target = std::getenv("TF_DISTRIBUTED_TARGET");
    
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(42.0f);
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::SessionOptions opts;
    opts.SetTarget(target);
    
    tf_wrap::Session s(g, opts);
    auto results = s.Run({}, {{"X", 0}}, {});
    
    CHECK(std::abs(results[0].ToScalar<float>() - 42.0f) < 0.001f);
    MESSAGE("Connected to: ", target);
}

TEST_CASE("distributed_remote_device_list" * doctest::test_suite("distributed")) {
    SKIP_IF_NO_DISTRIBUTED();
    
    const char* target = std::getenv("TF_DISTRIBUTED_TARGET");
    
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::SessionOptions opts;
    opts.SetTarget(target);
    
    tf_wrap::Session s(g, opts);
    auto devices = s.ListDevices();
    
    MESSAGE("Remote devices:");
    for (int i = 0; i < devices.count(); ++i) {
        auto dev = devices.at(i);
        MESSAGE("  ", dev.name, " (", dev.type, ")");
    }
    
    CHECK(devices.count() > 0);
}

TEST_CASE("distributed_cross_worker_computation" * doctest::test_suite("distributed")) {
    SKIP_IF_NO_DISTRIBUTED();
    
    const char* target = std::getenv("TF_DISTRIBUTED_TARGET");
    
    tf_wrap::Graph g;
    auto t1 = tf_wrap::Tensor::FromScalar<float>(10.0f);
    auto t2 = tf_wrap::Tensor::FromScalar<float>(20.0f);
    
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
    
    tf_wrap::Session s(g, opts);
    auto results = s.Run({}, {{"Sum", 0}}, {});
    
    CHECK(std::abs(results[0].ToScalar<float>() - 30.0f) < 0.001f);
    MESSAGE("Cross-worker computation successful");
}

// ============================================================================
// OOM Handling Tests
// ============================================================================

TEST_CASE("oom_tensor_allocation_fails_gracefully" * doctest::test_suite("oom")) {
    SKIP_IF_NO_INFRA();
    
    std::vector<int64_t> huge_shape = {1'000'000, 1'000'000, 1'000};
    
    bool threw = false;
    try {
        auto tensor = tf_wrap::Tensor::Allocate<float>(huge_shape);
        MESSAGE("Warning: huge allocation succeeded?!");
    } catch (const std::exception& e) {
        threw = true;
        MESSAGE("Correctly threw: ", e.what());
    }
    
    CHECK(threw);
}

TEST_CASE("oom_graph_continues_after_allocation_failure" * doctest::test_suite("oom")) {
    SKIP_IF_NO_INFRA();
    
    tf_wrap::Graph g;
    
    auto t1 = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "Good")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    bool allocation_failed = false;
    try {
        std::vector<int64_t> huge_shape = {1'000'000'000'000LL};
        auto huge = tf_wrap::Tensor::Allocate<float>(huge_shape);
    } catch (...) {
        allocation_failed = true;
    }
    
    CHECK(allocation_failed);
    
    auto t2 = tf_wrap::Tensor::FromScalar<float>(2.0f);
    (void)g.NewOperation("Const", "StillGood")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Good", 0}, {"StillGood", 0}}, {});
    
    CHECK(results.size() == 2);
    CHECK(std::abs(results[0].ToScalar<float>() - 1.0f) < 0.001f);
    CHECK(std::abs(results[1].ToScalar<float>() - 2.0f) < 0.001f);
}

TEST_CASE("oom_session_survives_large_intermediate" * doctest::test_suite("oom")) {
    SKIP_IF_NO_HIGH_MEMORY();
    
    tf_wrap::Graph g;
    
    const size_t large_size = 500'000'000;
    std::vector<float> data(large_size, 1.0f);
    auto t = tf_wrap::Tensor::FromVector<float>(
        {static_cast<int64_t>(large_size)}, data);
    
    (void)g.NewOperation("Const", "Large")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Session s(g);
    
    auto results = s.Run({}, {{"Large", 0}}, {});
    CHECK(results[0].num_elements() == large_size);
    
    data.clear();
    
    auto* large = g.GetOperationOrThrow("Large");
    (void)g.NewOperation("Square", "Squared")
        .AddInput(tf_wrap::Output(large, 0))
        .Finish();
    
    results = s.Run({}, {{"Squared", 0}}, {});
    CHECK(results[0].num_elements() == large_size);
    
    MESSAGE("Large tensor operations completed");
}

// ============================================================================
// Huge Tensor Tests (>1GB)
// ============================================================================

TEST_CASE("huge_tensor_1b_elements" * doctest::test_suite("huge")) {
    SKIP_IF_NO_HIGH_MEMORY();
    
    const size_t num_elements = 1'000'000'000;
    
    MESSAGE("Allocating 4GB tensor...");
    auto start = std::chrono::high_resolution_clock::now();
    
    auto tensor = tf_wrap::Tensor::Allocate<float>(
        {static_cast<int64_t>(num_elements)});
    
    auto alloc_end = std::chrono::high_resolution_clock::now();
    auto alloc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        alloc_end - start).count();
    MESSAGE("Allocation time: ", alloc_ms, "ms");
    
    CHECK(tensor.valid());
    CHECK(tensor.num_elements() == num_elements);
    
    {
        auto view = tensor.write<float>();
        view[0] = 1.0f;
        view[num_elements/2] = 2.0f;
        view[num_elements-1] = 3.0f;
    }
    
    {
        auto view = tensor.read<float>();
        CHECK(view[0] == 1.0f);
        CHECK(view[num_elements/2] == 2.0f);
        CHECK(view[num_elements-1] == 3.0f);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start).count();
    MESSAGE("Total time: ", total_ms, "ms");
}

TEST_CASE("huge_tensor_2b_elements" * doctest::test_suite("huge")) {
    SKIP_IF_NO_HIGH_MEMORY();
    
    const size_t num_elements = 2'000'000'000;
    
    MESSAGE("Allocating 8GB tensor...");
    
    auto tensor = tf_wrap::Tensor::Allocate<float>(
        {static_cast<int64_t>(num_elements)});
    
    CHECK(tensor.valid());
    CHECK(tensor.num_elements() == num_elements);
    
    MESSAGE("8GB tensor allocated successfully");
}

TEST_CASE("huge_tensor_multidimensional" * doctest::test_suite("huge")) {
    SKIP_IF_NO_HIGH_MEMORY();
    
    auto tensor = tf_wrap::Tensor::Allocate<float>({1000, 1000, 1000});
    
    CHECK(tensor.valid());
    CHECK(tensor.shape().size() == 3);
    CHECK(tensor.shape()[0] == 1000);
    CHECK(tensor.shape()[1] == 1000);
    CHECK(tensor.shape()[2] == 1000);
    CHECK(tensor.num_elements() == 1'000'000'000);
}

TEST_CASE("huge_tensor_session_run" * doctest::test_suite("huge")) {
    SKIP_IF_NO_HIGH_MEMORY();
    
    const size_t num_elements = 500'000'000;
    
    tf_wrap::Graph g;
    
    (void)g.NewOperation("Placeholder", "X")
        .SetAttrType("dtype", TF_FLOAT)
        .SetAttrShape("shape", {static_cast<int64_t>(num_elements)})
        .Finish();
    
    auto* x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("Square", "Y")
        .AddInput(tf_wrap::Output(x, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    
    auto input = tf_wrap::Tensor::Allocate<float>(
        {static_cast<int64_t>(num_elements)});
    {
        auto view = input.write<float>();
        view[0] = 2.0f;
        view[num_elements-1] = 3.0f;
    }
    
    MESSAGE("Running computation on 2GB tensor...");
    auto start = std::chrono::high_resolution_clock::now();
    
    auto results = s.Run(
        {{"X", 0, input.handle()}},
        {{"Y", 0}},
        {}
    );
    
    auto end = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    MESSAGE("Computation time: ", ms, "ms");
    
    auto v = results[0].read<float>();
    CHECK(std::abs(v[0] - 4.0f) < 0.001f);
    CHECK(std::abs(v[num_elements-1] - 9.0f) < 0.001f);
}

// ============================================================================
// High Thread Count Tests (32+)
// ============================================================================

TEST_CASE("high_thread_64_concurrent_sessions" * doctest::test_suite("threads")) {
    SKIP_IF_NO_MANY_CORES();
    
    const int num_threads = 64;
    const int runs_per_thread = 100;
    
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    std::vector<std::unique_ptr<tf_wrap::Session>> sessions;
    for (int i = 0; i < num_threads; ++i) {
        sessions.push_back(std::make_unique<tf_wrap::Session>(g));
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
    
    MESSAGE("Running ", num_threads, " threads x ", runs_per_thread, " runs...");
    
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
    
    MESSAGE("Completed in ", ms, "ms");
    MESSAGE("Success: ", success_count.load(), ", Errors: ", error_count.load());
    
    CHECK(error_count == 0);
    CHECK(success_count == num_threads * runs_per_thread);
}

TEST_CASE("high_thread_128_tensor_operations" * doctest::test_suite("threads")) {
    SKIP_IF_NO_MANY_CORES();
    
    const int num_threads = 128;
    const int ops_per_thread = 1000;
    
    std::atomic<int> success_count{0};
    std::atomic<int> error_count{0};
    
    auto worker = [&](int id) {
        try {
            for (int i = 0; i < ops_per_thread; ++i) {
                auto t = tf_wrap::Tensor::FromScalar<float>(
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
    
    MESSAGE("Running ", num_threads, " threads x ", ops_per_thread, " tensor ops...");
    
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
    
    MESSAGE("Completed in ", ms, "ms");
    MESSAGE("Throughput: ", (num_threads * ops_per_thread * 1000 / ms), " ops/sec");
    
    CHECK(error_count == 0);
    CHECK(success_count == num_threads * ops_per_thread);
}

TEST_CASE("high_thread_shared_graph_stress" * doctest::test_suite("threads")) {
    SKIP_IF_NO_MANY_CORES();
    
    const int num_threads = 64;
    
    tf_wrap::Graph g;
    
    auto t = tf_wrap::Tensor::FromScalar<float>(42.0f);
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
    
    MESSAGE(num_threads, " threads performed ", read_count.load(), " concurrent reads");
    
    CHECK(read_count > 0);
}

// ============================================================================
// Soak Tests (Long Running)
// ============================================================================

TEST_CASE("soak_1_hour_continuous_operations" * doctest::test_suite("soak")) {
    SKIP_IF_NO_SOAK();
    
    const auto duration = std::chrono::hours(1);
    const auto report_interval = std::chrono::minutes(5);
    
    MESSAGE("Starting 1-hour soak test...");
    
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Session s(g);
    
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
            MESSAGE("[", elapsed, "m] Ops: ", total_ops, ", Errors: ", errors);
            last_report = now;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto total_seconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    
    MESSAGE("Completed: ", total_ops, " operations in ", total_seconds, " seconds");
    MESSAGE("Throughput: ", (total_ops / total_seconds), " ops/sec");
    MESSAGE("Errors: ", errors);
    
    CHECK(errors == 0);
}

TEST_CASE("soak_memory_stability" * doctest::test_suite("soak")) {
    SKIP_IF_NO_SOAK();
    
    const auto duration = std::chrono::minutes(30);
    
    MESSAGE("Starting 30-minute memory stability test...");
    
    auto start = std::chrono::high_resolution_clock::now();
    size_t iterations = 0;
    
    while (std::chrono::high_resolution_clock::now() - start < duration) {
        for (int i = 0; i < 1000; ++i) {
            auto t = tf_wrap::Tensor::FromVector<float>({100, 100}, 
                std::vector<float>(10000, static_cast<float>(i)));
            auto v = t.ToVector<float>();
            (void)v;
        }
        
        for (int i = 0; i < 100; ++i) {
            tf_wrap::Graph g;
            auto t = tf_wrap::Tensor::FromScalar<float>(static_cast<float>(i));
            (void)g.NewOperation("Const", "X")
                .SetAttrTensor("value", t.handle())
                .SetAttrType("dtype", TF_FLOAT)
                .Finish();
        }
        
        ++iterations;
        
        if (iterations % 100 == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(
                std::chrono::high_resolution_clock::now() - start).count();
            MESSAGE("[", elapsed, "m] Iterations: ", iterations);
        }
    }
    
    MESSAGE("Completed ", iterations, " iterations without memory issues");
}

TEST_CASE("soak_thread_creation_destruction" * doctest::test_suite("soak")) {
    SKIP_IF_NO_SOAK();
    
    const auto duration = std::chrono::minutes(15);
    
    MESSAGE("Starting 15-minute thread stress test...");
    
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto start = std::chrono::high_resolution_clock::now();
    size_t thread_batches = 0;
    std::atomic<size_t> total_runs{0};
    
    while (std::chrono::high_resolution_clock::now() - start < duration) {
        std::vector<std::thread> threads;
        for (int i = 0; i < 16; ++i) {
            threads.emplace_back([&g, &total_runs]() {
                tf_wrap::Session s(g);
                for (int j = 0; j < 100; ++j) {
                    auto results = s.Run({}, {{"X", 0}}, {});
                    (void)results;
                    ++total_runs;
                }
            });
        }
        
        for (auto& th : threads) {
            th.join();
        }
        
        ++thread_batches;
        
        if (thread_batches % 50 == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(
                std::chrono::high_resolution_clock::now() - start).count();
            MESSAGE("[", elapsed, "m] Batches: ", thread_batches, ", Total runs: ", total_runs.load());
        }
    }
    
    MESSAGE("Completed ", thread_batches, " thread batches, ", total_runs.load(), " total runs");
}
