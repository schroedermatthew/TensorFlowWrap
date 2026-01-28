// test_comprehensive_tf.cpp
// Comprehensive integration tests with real TensorFlow
//
// Framework: Custom (see below)
// Runs with: Real TensorFlow (Linux CI only)
//
// Tests:
// - BatchRun / BatchRunStacked
// - RunContext zero-allocation path
// - Device enumeration with real hardware
// - Error paths with real TF

#include "tf_wrap/core.hpp"
#include "tf_wrap/facade.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
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

#define REQUIRE_CLOSE(a, b, eps) \
    do { \
        if (std::abs((a) - (b)) > (eps)) { \
            throw std::runtime_error("REQUIRE_CLOSE failed: " #a " vs " #b); \
        } \
    } while (0)

// Global test model path
static std::string g_model_path = "test_savedmodel";

// Helper: Find ops
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
// BatchRun Tests
// ============================================================================

TEST(batch_run_single_input) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    std::vector<Tensor> inputs = {
        Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f})
    };
    
    auto results = model.BatchRun(input_op, inputs, output_op);
    
    REQUIRE(results.size() == 1);
    REQUIRE(results[0].num_elements() == 3);
    
    // y = x * 2 + 1
    auto view = results[0].read<float>();
    REQUIRE_CLOSE(view[0], 3.0f, 1e-5f);
    REQUIRE_CLOSE(view[1], 5.0f, 1e-5f);
    REQUIRE_CLOSE(view[2], 7.0f, 1e-5f);
}

TEST(batch_run_multiple_inputs) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    std::vector<Tensor> inputs;
    for (int i = 0; i < 5; ++i) {
        inputs.push_back(Tensor::FromScalar<float>(static_cast<float>(i)));
    }
    
    auto results = model.BatchRun(input_op, inputs, output_op);
    
    REQUIRE(results.size() == 5);
    
    for (int i = 0; i < 5; ++i) {
        float expected = static_cast<float>(i) * 2.0f + 1.0f;
        REQUIRE_CLOSE(results[i].ToScalar<float>(), expected, 1e-5f);
    }
}

TEST(batch_run_varying_sizes) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    std::vector<Tensor> inputs;
    inputs.push_back(Tensor::FromVector<float>({1}, {1.0f}));
    inputs.push_back(Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f}));
    inputs.push_back(Tensor::FromVector<float>({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}));
    
    auto results = model.BatchRun(input_op, inputs, output_op);
    
    REQUIRE(results.size() == 3);
    REQUIRE(results[0].num_elements() == 1);
    REQUIRE(results[1].num_elements() == 3);
    REQUIRE(results[2].num_elements() == 5);
}

TEST(batch_run_stacked_basic) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    // Create same-size inputs for stacking
    std::vector<Tensor> inputs;
    for (int i = 0; i < 4; ++i) {
        inputs.push_back(Tensor::FromVector<float>({3}, 
            {static_cast<float>(i), static_cast<float>(i+1), static_cast<float>(i+2)}));
    }
    
    auto results = model.BatchRunStacked(input_op, std::span<const Tensor>(inputs), output_op);
    
    REQUIRE(results.size() == 4);
    
    for (std::size_t i = 0; i < 4; ++i) {
        auto view = results[i].read<float>();
        REQUIRE(view.size() == 3);
    }
}

TEST(batch_run_large_batch) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    std::vector<Tensor> inputs;
    for (int i = 0; i < 100; ++i) {
        inputs.push_back(Tensor::FromScalar<float>(static_cast<float>(i)));
    }
    
    auto results = model.BatchRun(input_op, inputs, output_op);
    
    REQUIRE(results.size() == 100);
    
    // Spot check
    REQUIRE_CLOSE(results[0].ToScalar<float>(), 1.0f, 1e-5f);   // 0*2+1
    REQUIRE_CLOSE(results[50].ToScalar<float>(), 101.0f, 1e-5f); // 50*2+1
    REQUIRE_CLOSE(results[99].ToScalar<float>(), 199.0f, 1e-5f); // 99*2+1
}

// ============================================================================
// RunContext Tests (Zero-allocation path)
// ============================================================================

TEST(run_context_basic) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    RunContext ctx;
    
    auto input = Tensor::FromScalar<float>(5.0f);
    ctx.add_feed(input_op, input);
    ctx.add_fetch(output_op);
    
    auto results = model.session().Run(ctx);
    
    REQUIRE(results.size() == 1);
    REQUIRE_CLOSE(results[0].ToScalar<float>(), 11.0f, 1e-5f); // 5*2+1
}

TEST(run_context_reuse) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    RunContext ctx(4, 2);  // Pre-allocate for 4 feeds, 2 fetches
    
    for (int i = 0; i < 10; ++i) {
        ctx.reset();
        
        auto input = Tensor::FromScalar<float>(static_cast<float>(i));
        ctx.add_feed(input_op, input);
        ctx.add_fetch(output_op);
        
        auto results = model.session().Run(ctx);
        
        float expected = static_cast<float>(i) * 2.0f + 1.0f;
        REQUIRE_CLOSE(results[0].ToScalar<float>(), expected, 1e-5f);
    }
}

TEST(run_context_performance) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    RunContext ctx;
    const int iterations = 100;
    
    auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        ctx.reset();
        
        auto input = Tensor::FromScalar<float>(1.0f);
        ctx.add_feed(input_op, input);
        ctx.add_fetch(output_op);
        
        auto results = model.session().Run(ctx);
        (void)results;
    }
    
    auto end = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "[" << iterations << " iterations in " << duration_ms << "ms] ";
    
    REQUIRE(duration_ms < 10000);  // Should complete in reasonable time
}

// ============================================================================
// Device Enumeration Tests
// ============================================================================

TEST(device_list_has_cpu) {
    auto model = Model::Load(g_model_path);
    auto devices = model.session().ListDevices();
    
    bool found_cpu = false;
    for (int i = 0; i < devices.count(); ++i) {
        auto dev = devices.at(i);
        if (dev.type == "CPU") {
            found_cpu = true;
            break;
        }
    }
    
    REQUIRE(found_cpu);
}

TEST(device_attributes) {
    auto model = Model::Load(g_model_path);
    auto devices = model.session().ListDevices();
    
    REQUIRE(devices.count() >= 1);
    
    auto dev = devices.at(0);
    REQUIRE(dev.name.length() > 0);
    REQUIRE(dev.type.length() > 0);
    // Memory bytes can be 0 for some devices
}

TEST(has_gpu_check) {
    auto model = Model::Load(g_model_path);
    
    // Just verify it doesn't crash
    bool has_gpu = model.session().HasGPU();
    std::cout << "[GPU: " << (has_gpu ? "yes" : "no") << "] ";
    
    REQUIRE(true);  // Pass as long as it doesn't crash
}

// ============================================================================
// Error Path Tests
// ============================================================================

TEST(load_nonexistent_model_throws) {
    bool threw = false;
    try {
        auto model = Model::Load("/nonexistent/path/to/model");
    } catch (const Error& e) {
        threw = true;
        REQUIRE(e.code() != TF_OK);
    } catch (const std::exception&) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST(resolve_nonexistent_op_throws) {
    auto model = Model::Load(g_model_path);
    
    bool threw = false;
    try {
        auto op = model.session().resolve("nonexistent_operation:0");
    } catch (const Error&) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST(validate_input_wrong_dtype) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    
    // Create int32 tensor for float input
    auto tensor = Tensor::FromScalar<int32_t>(42);
    
    auto error = model.validate_input(input_op, tensor);
    REQUIRE(!error.empty());  // Should report error
}

TEST(require_valid_input_throws_on_mismatch) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    
    auto tensor = Tensor::FromScalar<int32_t>(42);
    
    bool threw = false;
    try {
        model.require_valid_input(input_op, tensor);
    } catch (const Error&) {
        threw = true;
    }
    REQUIRE(threw);
}

// ============================================================================
// Graph Introspection with Real Model
// ============================================================================

TEST(graph_operations_nonempty) {
    auto model = Model::Load(g_model_path);
    auto& graph = model.graph();
    
    auto ops = graph.GetAllOperations();
    REQUIRE(ops.size() > 0);
}

TEST(graph_num_operations_matches) {
    auto model = Model::Load(g_model_path);
    auto& graph = model.graph();
    
    auto ops = graph.GetAllOperations();
    REQUIRE(graph.num_operations() == ops.size());
}

TEST(graph_is_frozen_after_session) {
    auto model = Model::Load(g_model_path);
    
    REQUIRE(model.graph().is_frozen());
}

TEST(graph_to_graphdef_nonempty) {
    auto model = Model::Load(g_model_path);
    
    auto graphdef = model.graph().ToGraphDef();
    REQUIRE(graphdef.size() > 0);
}

TEST(graph_debug_string_nonempty) {
    auto model = Model::Load(g_model_path);
    
    auto debug = model.graph().DebugString();
    REQUIRE(debug.length() > 0);
}

// ============================================================================
// Warmup Tests
// ============================================================================

TEST(warmup_single_inference) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    auto dummy = Tensor::FromVector<float>({10}, std::vector<float>(10, 0.0f));
    
    // Should not throw
    model.warmup(input_op, dummy, output_op);
    
    REQUIRE(true);
}

TEST(warmup_multiple_iterations) {
    auto model = Model::Load(g_model_path);
    auto input_op = find_input_op(model.graph());
    auto output_op = find_output_op(model.graph());
    
    auto dummy = Tensor::FromVector<float>({10}, std::vector<float>(10, 0.0f));
    
    // Warmup multiple times
    for (int i = 0; i < 5; ++i) {
        model.warmup(input_op, dummy, output_op);
    }
    
    REQUIRE(true);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc > 1) {
        g_model_path = argv[1];
    }
    
    std::cout << "=== TensorFlowWrap Comprehensive Tests (Real TF) ===" << std::endl;
    std::cout << "Model: " << g_model_path << std::endl;
    std::cout << std::endl;
    
    // Tests run automatically via static initialization
    
    std::cout << std::endl;
    std::cout << "=== Results: " << tests_passed << "/" << tests_run << " passed ===" << std::endl;
    
    return (tests_passed == tests_run) ? 0 : 1;
}
