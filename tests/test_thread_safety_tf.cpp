// test_thread_safety_tf.cpp
// Thread safety tests for tf_wrap::Session::Run
//
// Framework: Custom (see below)
// Runs with: Real TensorFlow (Linux CI only)
//
// These tests verify that Session::Run is thread-safe as per TensorFlow's
// guarantee. Multiple threads can call Run() concurrently on the same
// Session with different input tensors.
//
// Test scenarios:
// - Concurrent inference with same model
// - High contention (many threads, same session)
// - Mixed tensor sizes
// - Stress test with repeated iterations

#include "tf_wrap/core.hpp"
#include "tf_wrap/facade.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
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

#define REQUIRE_CLOSE(a, b, eps) \
    do { \
        if (std::abs((a) - (b)) > (eps)) { \
            throw std::runtime_error("REQUIRE_CLOSE failed: " #a " vs " #b); \
        } \
    } while (0)

// Global test model path
static std::string g_model_path = "test_savedmodel";

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
// Thread Safety Tests
// ============================================================================

// Test: Basic concurrent inference
// Multiple threads run inference simultaneously on the same model
TEST(concurrent_inference_basic) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    const int num_threads = 4;
    const int iterations_per_thread = 100;
    
    std::atomic<int> success_count{0};
    std::atomic<int> error_count{0};
    
    auto worker = [&](int thread_id) {
        for (int i = 0; i < iterations_per_thread; ++i) {
            try {
                float input_val = static_cast<float>(thread_id * 1000 + i);
                auto input = Tensor::FromScalar<float>(input_val);
                
                auto results = model.runner()
                    .feed(input_op, input)
                    .fetch(output_op)
                    .run();
                
                REQUIRE(results.size() == 1);
                REQUIRE(results[0].valid());
                
                // Verify result: y = x * 2.0 + 1.0
                float expected = input_val * 2.0f + 1.0f;
                float actual = results[0].ToScalar<float>();
                REQUIRE_CLOSE(actual, expected, 1e-5f);
                
                success_count.fetch_add(1, std::memory_order_relaxed);
            } catch (...) {
                error_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };
    
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker, t);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    REQUIRE(error_count.load() == 0);
    REQUIRE(success_count.load() == num_threads * iterations_per_thread);
}

// Test: High contention with many threads
TEST(concurrent_inference_high_contention) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    const int num_threads = 16;
    const int iterations_per_thread = 50;
    
    std::atomic<int> success_count{0};
    std::atomic<int> error_count{0};
    std::atomic<bool> start_flag{false};
    
    auto worker = [&](int thread_id) {
        // Wait for all threads to be ready
        while (!start_flag.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        
        for (int i = 0; i < iterations_per_thread; ++i) {
            try {
                float input_val = static_cast<float>(thread_id * 1000 + i);
                auto input = Tensor::FromScalar<float>(input_val);
                
                auto results = model.runner()
                    .feed(input_op, input)
                    .fetch(output_op)
                    .run();
                
                REQUIRE(results.size() == 1);
                
                float expected = input_val * 2.0f + 1.0f;
                float actual = results[0].ToScalar<float>();
                REQUIRE_CLOSE(actual, expected, 1e-5f);
                
                success_count.fetch_add(1, std::memory_order_relaxed);
            } catch (...) {
                error_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };
    
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker, t);
    }
    
    // Release all threads at once for maximum contention
    start_flag.store(true, std::memory_order_release);
    
    for (auto& t : threads) {
        t.join();
    }
    
    REQUIRE(error_count.load() == 0);
    REQUIRE(success_count.load() == num_threads * iterations_per_thread);
}

// Test: Concurrent inference with varying tensor sizes
TEST(concurrent_inference_varying_sizes) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    const int num_threads = 8;
    const int iterations_per_thread = 25;
    
    std::atomic<int> success_count{0};
    std::atomic<int> error_count{0};
    
    auto worker = [&](int thread_id) {
        std::mt19937 rng(thread_id);
        std::uniform_int_distribution<int> size_dist(1, 100);
        
        for (int i = 0; i < iterations_per_thread; ++i) {
            try {
                int size = size_dist(rng);
                std::vector<float> data(size);
                for (int j = 0; j < size; ++j) {
                    data[j] = static_cast<float>(thread_id * 10000 + i * 100 + j);
                }
                
                auto input = Tensor::FromVector<float>({static_cast<int64_t>(size)}, data);
                
                auto results = model.runner()
                    .feed(input_op, input)
                    .fetch(output_op)
                    .run();
                
                REQUIRE(results.size() == 1);
                REQUIRE(results[0].num_elements() == size);
                
                // Verify results
                auto view = results[0].read<float>();
                for (int j = 0; j < size; ++j) {
                    float expected = data[j] * 2.0f + 1.0f;
                    REQUIRE_CLOSE(view[j], expected, 1e-4f);
                }
                
                success_count.fetch_add(1, std::memory_order_relaxed);
            } catch (...) {
                error_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };
    
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker, t);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    REQUIRE(error_count.load() == 0);
    REQUIRE(success_count.load() == num_threads * iterations_per_thread);
}

// Test: Stress test - sustained concurrent load
TEST(concurrent_inference_stress) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    const int num_threads = 8;
    const int iterations_per_thread = 500;
    
    std::atomic<int> success_count{0};
    std::atomic<int> error_count{0};
    
    auto start_time = std::chrono::steady_clock::now();
    
    auto worker = [&](int thread_id) {
        for (int i = 0; i < iterations_per_thread; ++i) {
            try {
                float input_val = static_cast<float>(thread_id * 10000 + i);
                auto input = Tensor::FromScalar<float>(input_val);
                
                auto results = model.runner()
                    .feed(input_op, input)
                    .fetch(output_op)
                    .run();
                
                REQUIRE(results.size() == 1);
                
                float expected = input_val * 2.0f + 1.0f;
                float actual = results[0].ToScalar<float>();
                REQUIRE_CLOSE(actual, expected, 1e-5f);
                
                success_count.fetch_add(1, std::memory_order_relaxed);
            } catch (...) {
                error_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };
    
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker, t);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    int total = num_threads * iterations_per_thread;
    double throughput = static_cast<double>(total) / (duration_ms / 1000.0);
    
    std::cout << "[" << total << " inferences in " << duration_ms << "ms, " 
              << static_cast<int>(throughput) << " inf/s] ";
    
    REQUIRE(error_count.load() == 0);
    REQUIRE(success_count.load() == total);
}

// Test: Multiple models concurrently
TEST(concurrent_multiple_models) {
    const int num_models = 4;
    const int num_threads_per_model = 2;
    const int iterations_per_thread = 50;
    
    // Load multiple model instances
    std::vector<Model> models;
    std::vector<TF_Output> input_ops;
    std::vector<TF_Output> output_ops;
    
    for (int i = 0; i < num_models; ++i) {
        models.push_back(Model::Load(g_model_path));
        input_ops.push_back(find_input_op(models.back().graph()));
        output_ops.push_back(find_output_op(models.back().graph()));
    }
    
    std::atomic<int> success_count{0};
    std::atomic<int> error_count{0};
    
    auto worker = [&](int model_idx, int thread_id) {
        Model& model = models[model_idx];
        TF_Output input_op = input_ops[model_idx];
        TF_Output output_op = output_ops[model_idx];
        
        for (int i = 0; i < iterations_per_thread; ++i) {
            try {
                float input_val = static_cast<float>(model_idx * 100000 + thread_id * 1000 + i);
                auto input = Tensor::FromScalar<float>(input_val);
                
                auto results = model.runner()
                    .feed(input_op, input)
                    .fetch(output_op)
                    .run();
                
                REQUIRE(results.size() == 1);
                
                float expected = input_val * 2.0f + 1.0f;
                float actual = results[0].ToScalar<float>();
                REQUIRE_CLOSE(actual, expected, 1e-5f);
                
                success_count.fetch_add(1, std::memory_order_relaxed);
            } catch (...) {
                error_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };
    
    std::vector<std::thread> threads;
    for (int m = 0; m < num_models; ++m) {
        for (int t = 0; t < num_threads_per_model; ++t) {
            threads.emplace_back(worker, m, t);
        }
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    int total = num_models * num_threads_per_model * iterations_per_thread;
    REQUIRE(error_count.load() == 0);
    REQUIRE(success_count.load() == total);
}

// Test: Concurrent inference with Runner reuse
TEST(concurrent_runner_reuse) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    const int num_threads = 4;
    const int iterations_per_thread = 100;
    
    std::atomic<int> success_count{0};
    std::atomic<int> error_count{0};
    
    auto worker = [&](int thread_id) {
        // Each thread has its own runner that it reuses
        auto runner = model.runner();
        
        for (int i = 0; i < iterations_per_thread; ++i) {
            try {
                runner.clear();
                
                float input_val = static_cast<float>(thread_id * 1000 + i);
                auto input = Tensor::FromScalar<float>(input_val);
                
                runner.feed(input_op, input).fetch(output_op);
                auto results = runner.run();
                
                REQUIRE(results.size() == 1);
                
                float expected = input_val * 2.0f + 1.0f;
                float actual = results[0].ToScalar<float>();
                REQUIRE_CLOSE(actual, expected, 1e-5f);
                
                success_count.fetch_add(1, std::memory_order_relaxed);
            } catch (...) {
                error_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };
    
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker, t);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    REQUIRE(error_count.load() == 0);
    REQUIRE(success_count.load() == num_threads * iterations_per_thread);
}

// Test: Data race detection - verify no corruption
TEST(concurrent_data_integrity) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    const int num_threads = 8;
    const int iterations_per_thread = 100;
    
    std::atomic<int> success_count{0};
    std::atomic<int> mismatch_count{0};
    
    auto worker = [&](int thread_id) {
        // Use a unique pattern for this thread to detect corruption
        float base = static_cast<float>(thread_id * 1000000);
        
        for (int i = 0; i < iterations_per_thread; ++i) {
            float input_val = base + static_cast<float>(i);
            auto input = Tensor::FromScalar<float>(input_val);
            
            auto results = model.runner()
                .feed(input_op, input)
                .fetch(output_op)
                .run();
            
            float expected = input_val * 2.0f + 1.0f;
            float actual = results[0].ToScalar<float>();
            
            // Strict check for data corruption
            if (std::abs(actual - expected) > 1e-5f) {
                mismatch_count.fetch_add(1, std::memory_order_relaxed);
            } else {
                success_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };
    
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker, t);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // Any mismatch indicates potential data race
    REQUIRE(mismatch_count.load() == 0);
    REQUIRE(success_count.load() == num_threads * iterations_per_thread);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc > 1) {
        g_model_path = argv[1];
    }
    
    std::cout << "=== TensorFlowWrap Thread Safety Tests (Real TF) ===" << std::endl;
    std::cout << "Testing concurrent Session::Run() calls" << std::endl;
    std::cout << "Hardware concurrency: " << std::thread::hardware_concurrency() << " threads" << std::endl;
    std::cout << std::endl;
    
    // Tests run automatically via static initialization
    
    std::cout << std::endl;
    std::cout << "=== Results: " << tests_passed << "/" << tests_run << " passed ===" << std::endl;
    
    return (tests_passed == tests_run) ? 0 : 1;
}
