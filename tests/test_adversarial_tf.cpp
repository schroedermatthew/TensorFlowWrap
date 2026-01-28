// test_adversarial_tf.cpp
// Adversarial tests with real TensorFlow
//
// Framework: Custom
// Runs with: Real TensorFlow (Linux CI only)
//
// These tests attempt to break the system under real TF conditions

#include "tf_wrap/core.hpp"
#include "tf_wrap/facade.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
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
    do { if (std::abs((a) - (b)) > (eps)) throw std::runtime_error("REQUIRE_CLOSE failed"); } while (0)

static std::string g_model_path = "test_savedmodel";

static TF_Output find_output_op(const Graph& graph) {
    if (auto op = graph.GetOperation("PartitionedCall")) return TF_Output{*op, 0};
    if (auto op = graph.GetOperation("StatefulPartitionedCall")) return TF_Output{*op, 0};
    throw std::runtime_error("Could not find output operation");
}

static TF_Output find_input_op(const Graph& graph) {
    if (auto op = graph.GetOperation("serving_default_x")) return TF_Output{*op, 0};
    throw std::runtime_error("Could not find input operation");
}

// ============================================================================
// Concurrent Stress Tests
// ============================================================================

TEST(concurrent_inference_stress_8_threads) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    std::atomic<int> success{0};
    std::atomic<int> failure{0};
    
    auto worker = [&](int thread_id) {
        std::mt19937 rng(thread_id);
        std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
        
        for (int i = 0; i < 200; ++i) {
            try {
                float val = dist(rng);
                auto input = Tensor::FromScalar<float>(val);
                
                auto result = model.runner()
                    .feed(input_op, input)
                    .fetch(output_op)
                    .run_one();
                
                float expected = val * 2.0f + 1.0f;
                float actual = result.ToScalar<float>();
                
                if (std::abs(actual - expected) < 1e-4f) {
                    success++;
                } else {
                    failure++;
                }
            } catch (...) {
                failure++;
            }
        }
    };
    
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back(worker, t);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    REQUIRE(failure.load() == 0);
    REQUIRE(success.load() == 8 * 200);
}

TEST(concurrent_model_reload_while_running) {
    std::atomic<bool> keep_running{true};
    std::atomic<int> inference_count{0};
    std::atomic<int> error_count{0};
    
    auto model = std::make_shared<Model>(Model::Load(g_model_path));
    
    // Inference thread
    std::thread inference_thread([&]() {
        while (keep_running) {
            try {
                auto local_model = model;  // Copy shared_ptr
                auto input_op = find_input_op(local_model->graph());
                auto output_op = find_output_op(local_model->graph());
                
                auto input = Tensor::FromScalar<float>(1.0f);
                auto result = local_model->runner()
                    .feed(input_op, input)
                    .fetch(output_op)
                    .run_one();
                
                inference_count++;
            } catch (...) {
                error_count++;
            }
        }
    });
    
    // Main thread reloads model periodically
    for (int i = 0; i < 5; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        auto new_model = std::make_shared<Model>(Model::Load(g_model_path));
        std::atomic_store(&model, new_model);
    }
    
    keep_running = false;
    inference_thread.join();
    
    std::cout << "[" << inference_count << " inferences, " << error_count << " errors] ";
    
    REQUIRE(inference_count > 0);
}

TEST(concurrent_batch_and_single_inference) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    std::atomic<int> single_count{0};
    std::atomic<int> batch_count{0};
    std::atomic<bool> failed{false};
    
    // Single inference thread
    std::thread single_thread([&]() {
        for (int i = 0; i < 100 && !failed; ++i) {
            try {
                auto input = Tensor::FromScalar<float>(static_cast<float>(i));
                auto result = model.runner()
                    .feed(input_op, input)
                    .fetch(output_op)
                    .run_one();
                
                float expected = static_cast<float>(i) * 2.0f + 1.0f;
                if (std::abs(result.ToScalar<float>() - expected) > 1e-4f) {
                    failed = true;
                }
                single_count++;
            } catch (...) {
                failed = true;
            }
        }
    });
    
    // Batch inference thread
    std::thread batch_thread([&]() {
        for (int i = 0; i < 20 && !failed; ++i) {
            try {
                std::vector<Tensor> inputs;
                for (int j = 0; j < 5; ++j) {
                    inputs.push_back(Tensor::FromScalar<float>(static_cast<float>(j)));
                }
                
                auto results = model.BatchRun(input_op, inputs, output_op);
                
                for (int j = 0; j < 5; ++j) {
                    float expected = static_cast<float>(j) * 2.0f + 1.0f;
                    if (std::abs(results[j].ToScalar<float>() - expected) > 1e-4f) {
                        failed = true;
                    }
                }
                batch_count++;
            } catch (...) {
                failed = true;
            }
        }
    });
    
    single_thread.join();
    batch_thread.join();
    
    REQUIRE(!failed);
    REQUIRE(single_count == 100);
    REQUIRE(batch_count == 20);
}

// ============================================================================
// Resource Stress Tests
// ============================================================================

TEST(rapid_model_load_unload) {
    for (int i = 0; i < 20; ++i) {
        auto model = Model::Load(g_model_path);
        auto input_op = find_input_op(model.graph());
        auto output_op = find_output_op(model.graph());
        
        auto input = Tensor::FromScalar<float>(1.0f);
        auto result = model.runner()
            .feed(input_op, input)
            .fetch(output_op)
            .run_one();
        
        REQUIRE_CLOSE(result.ToScalar<float>(), 3.0f, 1e-5f);
    }
}

TEST(many_tensors_single_session) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    for (int i = 0; i < 5000; ++i) {
        auto input = Tensor::FromScalar<float>(static_cast<float>(i));
        auto result = model.runner()
            .feed(input_op, input)
            .fetch(output_op)
            .run_one();
        
        float expected = static_cast<float>(i) * 2.0f + 1.0f;
        REQUIRE_CLOSE(result.ToScalar<float>(), expected, 1e-4f);
    }
}

TEST(large_tensor_inference) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    // 1MB tensor
    std::vector<float> data(262144, 1.0f);  // 256K floats = 1MB
    auto input = Tensor::FromVector<float>({262144}, data);
    
    auto result = model.runner()
        .feed(input_op, input)
        .fetch(output_op)
        .run_one();
    
    REQUIRE(result.num_elements() == 262144);
    
    auto view = result.read<float>();
    REQUIRE_CLOSE(view[0], 3.0f, 1e-5f);  // 1*2+1
}

TEST(varying_tensor_sizes_rapid) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> size_dist(1, 10000);
    
    for (int i = 0; i < 100; ++i) {
        int size = size_dist(rng);
        std::vector<float> data(size, static_cast<float>(i));
        auto input = Tensor::FromVector<float>({static_cast<int64_t>(size)}, data);
        
        auto result = model.runner()
            .feed(input_op, input)
            .fetch(output_op)
            .run_one();
        
        REQUIRE(result.num_elements() == size);
    }
}

// ============================================================================
// Error Recovery Tests
// ============================================================================

TEST(recover_from_bad_input) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    // Bad input (wrong dtype)
    bool threw = false;
    try {
        auto bad_input = Tensor::FromScalar<int32_t>(42);
        model.require_valid_input(input_op, bad_input);
    } catch (...) {
        threw = true;
    }
    REQUIRE(threw);
    
    // Good input should still work
    auto good_input = Tensor::FromScalar<float>(1.0f);
    auto result = model.runner()
        .feed(input_op, good_input)
        .fetch(output_op)
        .run_one();
    
    REQUIRE_CLOSE(result.ToScalar<float>(), 3.0f, 1e-5f);
}

TEST(recover_from_bad_operation) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    // Try to resolve bad operation
    bool threw = false;
    try {
        model.session().resolve("nonexistent_op:0");
    } catch (...) {
        threw = true;
    }
    REQUIRE(threw);
    
    // Good operations should still work
    auto input = Tensor::FromScalar<float>(2.0f);
    auto result = model.runner()
        .feed(input_op, input)
        .fetch(output_op)
        .run_one();
    
    REQUIRE_CLOSE(result.ToScalar<float>(), 5.0f, 1e-5f);
}

// ============================================================================
// Edge Case Inputs
// ============================================================================

TEST(special_float_values) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    // Infinity
    {
        auto input = Tensor::FromScalar<float>(std::numeric_limits<float>::infinity());
        auto result = model.runner()
            .feed(input_op, input)
            .fetch(output_op)
            .run_one();
        REQUIRE(std::isinf(result.ToScalar<float>()));
    }
    
    // Negative infinity
    {
        auto input = Tensor::FromScalar<float>(-std::numeric_limits<float>::infinity());
        auto result = model.runner()
            .feed(input_op, input)
            .fetch(output_op)
            .run_one();
        REQUIRE(std::isinf(result.ToScalar<float>()));
    }
    
    // NaN propagates
    {
        auto input = Tensor::FromScalar<float>(std::numeric_limits<float>::quiet_NaN());
        auto result = model.runner()
            .feed(input_op, input)
            .fetch(output_op)
            .run_one();
        REQUIRE(std::isnan(result.ToScalar<float>()));
    }
    
    // Denormalized numbers
    {
        auto input = Tensor::FromScalar<float>(std::numeric_limits<float>::denorm_min());
        auto result = model.runner()
            .feed(input_op, input)
            .fetch(output_op)
            .run_one();
        REQUIRE(result.valid());
    }
}

TEST(empty_tensor_handling) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    // Empty tensor (0 elements)
    std::vector<float> empty;
    auto input = Tensor::FromVector<float>({0}, empty);
    
    auto result = model.runner()
        .feed(input_op, input)
        .fetch(output_op)
        .run_one();
    
    REQUIRE(result.num_elements() == 0);
}

TEST(extreme_values) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    // Max float
    {
        auto input = Tensor::FromScalar<float>(std::numeric_limits<float>::max());
        auto result = model.runner()
            .feed(input_op, input)
            .fetch(output_op)
            .run_one();
        REQUIRE(std::isinf(result.ToScalar<float>()));  // Overflow expected
    }
    
    // Min positive float
    {
        auto input = Tensor::FromScalar<float>(std::numeric_limits<float>::min());
        auto result = model.runner()
            .feed(input_op, input)
            .fetch(output_op)
            .run_one();
        REQUIRE_CLOSE(result.ToScalar<float>(), 1.0f, 1e-5f);  // ~0*2+1
    }
}

// ============================================================================
// RunContext Stress
// ============================================================================

TEST(run_context_rapid_reuse) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    RunContext ctx;
    
    for (int i = 0; i < 1000; ++i) {
        ctx.reset();
        
        auto input = Tensor::FromScalar<float>(static_cast<float>(i % 100));
        ctx.add_feed(input_op, input);
        ctx.add_fetch(output_op);
        
        auto results = model.session().Run(ctx);
        
        float expected = static_cast<float>(i % 100) * 2.0f + 1.0f;
        REQUIRE_CLOSE(results[0].ToScalar<float>(), expected, 1e-4f);
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc > 1) {
        g_model_path = argv[1];
    }
    
    std::cout << "=== TensorFlowWrap Adversarial Tests (Real TF) ===" << std::endl;
    std::cout << "Model: " << g_model_path << std::endl;
    std::cout << std::endl;
    
    // Tests run automatically via static initialization
    
    std::cout << std::endl;
    std::cout << "=== Results: " << tests_passed << "/" << tests_run << " passed ===" << std::endl;
    
    return (tests_passed == tests_run) ? 0 : 1;
}
