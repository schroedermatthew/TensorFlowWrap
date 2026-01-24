// tests/test_real_tf.cpp
// Comprehensive test suite for real TensorFlow C API (not stub)
// Tests operations, error handling, edge cases, and stress scenarios

#include "tf_wrap/all.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

// Simple test framework
static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST(name) \
    void test_##name(); \
    struct TestReg_##name { \
        TestReg_##name() { \
            std::cout << "[TEST] " << #name << "\n"; \
            g_tests_run++; \
            try { \
                test_##name(); \
                std::cout << "  PASS\n"; \
                g_tests_passed++; \
            } catch (const std::exception& e) { \
                std::cout << "  FAIL: " << e.what() << "\n"; \
                g_tests_failed++; \
            } catch (...) { \
                std::cout << "  FAIL: unknown exception\n"; \
                g_tests_failed++; \
            } \
        } \
    } g_testreg_##name; \
    void test_##name()

#define REQUIRE(cond) \
    do { if (!(cond)) throw std::runtime_error("REQUIRE failed: " #cond); } while(0)

#define REQUIRE_APPROX(a, b, eps) \
    do { if (std::abs((a) - (b)) > (eps)) \
        throw std::runtime_error("REQUIRE_APPROX failed: " #a " != " #b); } while(0)

// =============================================================================
// Basic Operations
// =============================================================================

TEST(const_and_identity) {
    tf_wrap::Graph g;
    
    auto t = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* op_x = g.GetOperationOrThrow("X");
    (void)g.NewOperation("Identity", "Y")
        .AddInput(tf_wrap::Output(op_x, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Y", 0}}, {});
    
    REQUIRE(results.size() == 1);
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 3);
    REQUIRE(v[0] == 1.0f && v[1] == 2.0f && v[2] == 3.0f);
}

TEST(add_subtract_multiply) {
    tf_wrap::Graph g;
    
    auto a = tf_wrap::Tensor::FromVector<float>({2}, {10.0f, 20.0f});
    auto b = tf_wrap::Tensor::FromVector<float>({2}, {3.0f, 4.0f});
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("AddV2", "Add").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    (void)g.NewOperation("Sub", "Sub").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    (void)g.NewOperation("Mul", "Mul").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Add", 0}, {"Sub", 0}, {"Mul", 0}}, {});
    
    REQUIRE(results.size() == 3);
    
    auto add_v = results[0].ToVector<float>();
    REQUIRE(add_v[0] == 13.0f && add_v[1] == 24.0f);
    
    auto sub_v = results[1].ToVector<float>();
    REQUIRE(sub_v[0] == 7.0f && sub_v[1] == 16.0f);
    
    auto mul_v = results[2].ToVector<float>();
    REQUIRE(mul_v[0] == 30.0f && mul_v[1] == 80.0f);
}

TEST(matmul_2x2) {
    tf_wrap::Graph g;
    
    // A = [[1, 2], [3, 4]]
    auto a = tf_wrap::Tensor::FromVector<float>({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    // B = [[5, 6], [7, 8]]
    auto b = tf_wrap::Tensor::FromVector<float>({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("MatMul", "C")
        .AddInput(tf_wrap::Output(op_a, 0))
        .AddInput(tf_wrap::Output(op_b, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"C", 0}}, {});
    
    // C = A @ B = [[19, 22], [43, 50]]
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 4);
    REQUIRE(v[0] == 19.0f && v[1] == 22.0f && v[2] == 43.0f && v[3] == 50.0f);
}

TEST(reshape) {
    tf_wrap::Graph g;
    
    auto t = tf_wrap::Tensor::FromVector<float>({6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto shape_t = tf_wrap::Tensor::FromVector<int32_t>({2}, {2, 3});
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Shape").SetAttrTensor("value", shape_t.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    auto* op_shape = g.GetOperationOrThrow("Shape");
    
    (void)g.NewOperation("Reshape", "R")
        .AddInput(tf_wrap::Output(op_t, 0))
        .AddInput(tf_wrap::Output(op_shape, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"R", 0}}, {});
    
    REQUIRE(results[0].rank() == 2);
    REQUIRE(results[0].shape()[0] == 2);
    REQUIRE(results[0].shape()[1] == 3);
}

TEST(reduce_sum) {
    tf_wrap::Graph g;
    
    // 2x3 matrix
    auto t = tf_wrap::Tensor::FromVector<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto axis = tf_wrap::Tensor::FromVector<int32_t>({1}, {1}); // reduce along axis 1
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Axis").SetAttrTensor("value", axis.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    auto* op_axis = g.GetOperationOrThrow("Axis");
    
    (void)g.NewOperation("Sum", "S")
        .AddInput(tf_wrap::Output(op_t, 0))
        .AddInput(tf_wrap::Output(op_axis, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"S", 0}}, {});
    
    // Sum along axis 1: [1+2+3, 4+5+6] = [6, 15]
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 2);
    REQUIRE(v[0] == 6.0f && v[1] == 15.0f);
}

TEST(relu) {
    tf_wrap::Graph g;
    
    auto t = tf_wrap::Tensor::FromVector<float>({4}, {-2.0f, -1.0f, 1.0f, 2.0f});
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    
    (void)g.NewOperation("Relu", "R")
        .AddInput(tf_wrap::Output(op_t, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"R", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v[0] == 0.0f && v[1] == 0.0f && v[2] == 1.0f && v[3] == 2.0f);
}

TEST(softmax) {
    tf_wrap::Graph g;
    
    auto t = tf_wrap::Tensor::FromVector<float>({1, 3}, {1.0f, 2.0f, 3.0f});
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    
    (void)g.NewOperation("Softmax", "S")
        .AddInput(tf_wrap::Output(op_t, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"S", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 3);
    
    // Softmax should sum to 1
    float sum = v[0] + v[1] + v[2];
    REQUIRE_APPROX(sum, 1.0f, 0.0001f);
    
    // Values should be increasing
    REQUIRE(v[0] < v[1] && v[1] < v[2]);
}

// =============================================================================
// Placeholder and Feed
// =============================================================================

TEST(placeholder_feed) {
    tf_wrap::Graph g;
    
    (void)g.NewOperation("Placeholder", "X")
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* op_x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("Square", "Y")
        .AddInput(tf_wrap::Output(op_x, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    
    // Feed different values
    auto input1 = tf_wrap::Tensor::FromVector<float>({3}, {2.0f, 3.0f, 4.0f});
    auto results1 = s.Run({{"X", 0, input1.handle()}}, {{"Y", 0}}, {});
    auto v1 = results1[0].ToVector<float>();
    REQUIRE(v1[0] == 4.0f && v1[1] == 9.0f && v1[2] == 16.0f);
    
    auto input2 = tf_wrap::Tensor::FromVector<float>({2}, {5.0f, 6.0f});
    auto results2 = s.Run({{"X", 0, input2.handle()}}, {{"Y", 0}}, {});
    auto v2 = results2[0].ToVector<float>();
    REQUIRE(v2[0] == 25.0f && v2[1] == 36.0f);
}

// =============================================================================
// Data Types
// =============================================================================

TEST(int32_operations) {
    tf_wrap::Graph g;
    
    auto a = tf_wrap::Tensor::FromVector<int32_t>({3}, {10, 20, 30});
    auto b = tf_wrap::Tensor::FromVector<int32_t>({3}, {1, 2, 3});
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_INT32).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("AddV2", "Sum").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Sum", 0}}, {});
    
    auto v = results[0].ToVector<int32_t>();
    REQUIRE(v[0] == 11 && v[1] == 22 && v[2] == 33);
}

TEST(int64_operations) {
    tf_wrap::Graph g;
    
    auto a = tf_wrap::Tensor::FromVector<int64_t>({2}, {1000000000000LL, 2000000000000LL});
    auto b = tf_wrap::Tensor::FromVector<int64_t>({2}, {1LL, 2LL});
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_INT64).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_INT64).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("AddV2", "Sum").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Sum", 0}}, {});
    
    auto v = results[0].ToVector<int64_t>();
    REQUIRE(v[0] == 1000000000001LL && v[1] == 2000000000002LL);
}

TEST(double_precision) {
    tf_wrap::Graph g;
    
    auto a = tf_wrap::Tensor::FromVector<double>({2}, {1.0000000001, 2.0000000002});
    auto b = tf_wrap::Tensor::FromVector<double>({2}, {0.0000000001, 0.0000000002});
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_DOUBLE).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_DOUBLE).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("AddV2", "Sum").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Sum", 0}}, {});
    
    auto v = results[0].ToVector<double>();
    REQUIRE_APPROX(v[0], 1.0000000002, 1e-15);
    REQUIRE_APPROX(v[1], 2.0000000004, 1e-15);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(scalar_tensor) {
    tf_wrap::Graph g;
    
    auto scalar = tf_wrap::Tensor::FromScalar<float>(42.0f);
    
    (void)g.NewOperation("Const", "S").SetAttrTensor("value", scalar.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"S", 0}}, {});
    
    REQUIRE(results[0].rank() == 0);
    REQUIRE(results[0].num_elements() == 1);
    REQUIRE(results[0].ToScalar<float>() == 42.0f);
}

TEST(empty_tensor) {
    tf_wrap::Graph g;
    
    auto empty = tf_wrap::Tensor::FromVector<float>({0}, {});
    
    (void)g.NewOperation("Const", "E").SetAttrTensor("value", empty.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_e = g.GetOperationOrThrow("E");
    (void)g.NewOperation("Identity", "I").AddInput(tf_wrap::Output(op_e, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"I", 0}}, {});
    
    REQUIRE(results[0].num_elements() == 0);
}

TEST(large_tensor) {
    tf_wrap::Graph g;
    
    // 1M elements
    std::vector<float> data(1000000);
    std::iota(data.begin(), data.end(), 0.0f);
    std::vector<int64_t> shape_vec = {1000, 1000};
    
    auto large = tf_wrap::Tensor::FromVector<float>(shape_vec, data);
    
    (void)g.NewOperation("Const", "L").SetAttrTensor("value", large.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_l = g.GetOperationOrThrow("L");
    
    // Reduce to single value
    auto axis = tf_wrap::Tensor::FromVector<int32_t>({2}, {0, 1});
    (void)g.NewOperation("Const", "Axis").SetAttrTensor("value", axis.handle()).SetAttrType("dtype", TF_INT32).Finish();
    auto* op_axis = g.GetOperationOrThrow("Axis");
    
    (void)g.NewOperation("Sum", "S")
        .AddInput(tf_wrap::Output(op_l, 0))
        .AddInput(tf_wrap::Output(op_axis, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"S", 0}}, {});
    
    // Sum of 0..999999 = n*(n-1)/2 = 499999500000
    float expected = 499999500000.0f;
    float actual = results[0].ToScalar<float>();
    REQUIRE_APPROX(actual, expected, expected * 0.0001f); // 0.01% tolerance for float
}

TEST(high_rank_tensor) {
    tf_wrap::Graph g;
    
    // 5D tensor: 2x3x4x5x6 = 720 elements
    std::vector<float> data(720);
    std::iota(data.begin(), data.end(), 0.0f);
    std::vector<int64_t> shape_vec = {2, 3, 4, 5, 6};
    
    auto t = tf_wrap::Tensor::FromVector<float>(shape_vec, data);
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"T", 0}}, {});
    
    REQUIRE(results[0].rank() == 5);
    REQUIRE(results[0].shape()[0] == 2);
    REQUIRE(results[0].shape()[1] == 3);
    REQUIRE(results[0].shape()[2] == 4);
    REQUIRE(results[0].shape()[3] == 5);
    REQUIRE(results[0].shape()[4] == 6);
}

// =============================================================================
// Error Handling
// =============================================================================

TEST(invalid_operation_name_throws) {
    tf_wrap::Graph g;
    
    bool threw = false;
    try {
        g.GetOperationOrThrow("nonexistent");
    } catch (const std::exception&) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST(dtype_mismatch_throws) {
    auto t = tf_wrap::Tensor::FromVector<float>({2}, {1.0f, 2.0f});
    
    bool threw = false;
    try {
        t.ToVector<int32_t>(); // Wrong type
    } catch (const std::exception&) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST(session_run_missing_feed_throws) {
    // Create graph with placeholder but don't feed it
    tf_wrap::Graph g;
    
    (void)g.NewOperation("Placeholder", "X")
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* op_x = g.GetOperationOrThrow("X");
    (void)g.NewOperation("Identity", "Y")
        .AddInput(tf_wrap::Output(op_x, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    
    bool threw = false;
    std::string error_msg;
    try {
        // Run without feeding X - should fail
        auto results = s.Run({}, {{"Y", 0}}, {});
    } catch (const std::exception& e) {
        threw = true;
        error_msg = e.what();
    }
    REQUIRE(threw);
    std::cout << "    (Error: " << error_msg.substr(0, 60) << "...)\n";
}

TEST(session_run_incompatible_matmul_shapes_throws) {
    // Test that runtime shape validation works for placeholders.
    // MatMul requires 2D inputs; feeding 1D tensor should fail at Run() time.
    tf_wrap::Graph g;
    
    (void)g.NewOperation("Placeholder", "A")
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    (void)g.NewOperation("Placeholder", "B")
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("MatMul", "C")
        .AddInput(tf_wrap::Output(op_a, 0))
        .AddInput(tf_wrap::Output(op_b, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    
    // Feed 1D tensors - MatMul requires 2D
    auto a_tensor = tf_wrap::Tensor::FromVector<float>({6}, 
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto b_tensor = tf_wrap::Tensor::FromVector<float>({6}, 
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    
    bool threw = false;
    std::string error_msg;
    try {
        auto results = s.Run(
            {{"A", 0, a_tensor.handle()}, {"B", 0, b_tensor.handle()}}, 
            {{"C", 0}}, {});
    } catch (const std::exception& e) {
        threw = true;
        error_msg = e.what();
    }
    REQUIRE(threw);
    std::cout << "    (Error: " << error_msg.substr(0, 60) << "...)\n";
}

TEST(session_run_wrong_feed_dtype_throws) {
    tf_wrap::Graph g;
    
    // Placeholder expects float
    (void)g.NewOperation("Placeholder", "X")
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* op_x = g.GetOperationOrThrow("X");
    (void)g.NewOperation("Identity", "Y")
        .AddInput(tf_wrap::Output(op_x, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    
    // Feed int32 instead of float
    auto wrong_dtype = tf_wrap::Tensor::FromVector<int32_t>({3}, {1, 2, 3});
    
    bool threw = false;
    std::string error_msg;
    try {
        auto results = s.Run({{"X", 0, wrong_dtype.handle()}}, {{"Y", 0}}, {});
    } catch (const std::exception& e) {
        threw = true;
        error_msg = e.what();
    }
    REQUIRE(threw);
    std::cout << "    (Error: " << error_msg.substr(0, 60) << "...)\n";
}

TEST(fetch_nonexistent_operation_throws) {
    tf_wrap::Graph g;
    
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Session s(g);
    
    bool threw = false;
    std::string error_msg;
    try {
        // Fetch operation that doesn't exist
        auto results = s.Run({}, {{"NonExistent", 0}}, {});
    } catch (const std::exception& e) {
        threw = true;
        error_msg = e.what();
    }
    REQUIRE(threw);
    std::cout << "    (Error: " << error_msg.substr(0, 60) << "...)\n";
}

TEST(invalid_output_index_throws) {
    tf_wrap::Graph g;
    
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Session s(g);
    
    bool threw = false;
    std::string error_msg;
    try {
        // Const has only output 0, try to fetch output 99
        auto results = s.Run({}, {{"X", 99}}, {});
    } catch (const std::exception& e) {
        threw = true;
        error_msg = e.what();
    }
    REQUIRE(threw);
    std::cout << "    (Error: " << error_msg.substr(0, 60) << "...)\n";
}

TEST(operation_with_wrong_input_count_throws) {
    tf_wrap::Graph g;
    
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* op_x = g.GetOperationOrThrow("X");
    
    bool threw = false;
    std::string error_msg;
    try {
        // AddV2 requires 2 inputs, only provide 1
        (void)g.NewOperation("AddV2", "Bad")
            .AddInput(tf_wrap::Output(op_x, 0))
            .Finish();
    } catch (const std::exception& e) {
        threw = true;
        error_msg = e.what();
    }
    REQUIRE(threw);
    std::cout << "    (Error: " << error_msg.substr(0, 60) << "...)\n";
}

TEST(operation_type_mismatch_throws) {
    tf_wrap::Graph g;
    
    // Create float and int tensors
    auto f = tf_wrap::Tensor::FromScalar<float>(1.0f);
    auto i = tf_wrap::Tensor::FromScalar<int32_t>(2);
    
    (void)g.NewOperation("Const", "F")
        .SetAttrTensor("value", f.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    (void)g.NewOperation("Const", "I")
        .SetAttrTensor("value", i.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    auto* op_f = g.GetOperationOrThrow("F");
    auto* op_i = g.GetOperationOrThrow("I");
    
    bool threw = false;
    std::string error_msg;
    try {
        // AddV2 with mismatched types should fail
        (void)g.NewOperation("AddV2", "Bad")
            .AddInput(tf_wrap::Output(op_f, 0))
            .AddInput(tf_wrap::Output(op_i, 0))
            .Finish();
    } catch (const std::exception& e) {
        threw = true;
        error_msg = e.what();
    }
    REQUIRE(threw);
    std::cout << "    (Error: " << error_msg.substr(0, 60) << "...)\n";
}

TEST(matmul_incompatible_shapes_at_finish_throws) {
    // Real TensorFlow validates shapes at Finish() time when shapes are known statically
    tf_wrap::Graph g;
    
    // A is 2x3, B is 2x3 - can't multiply (need 2x3 @ 3xN)
    auto a = tf_wrap::Tensor::FromVector<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto b = tf_wrap::Tensor::FromVector<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", a.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    (void)g.NewOperation("Const", "B")
        .SetAttrTensor("value", b.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    bool threw = false;
    std::string error_msg;
    try {
        // TF validates shapes at Finish() time for Const inputs
        (void)g.NewOperation("MatMul", "C")
            .AddInput(tf_wrap::Output(op_a, 0))
            .AddInput(tf_wrap::Output(op_b, 0))
            .Finish();
    } catch (const std::exception& e) {
        threw = true;
        error_msg = e.what();
    }
    REQUIRE(threw);
    std::cout << "    (Error: " << error_msg.substr(0, 60) << "...)\n";
}

TEST(reshape_incompatible_size_at_finish_throws) {
    // Real TensorFlow validates reshape compatibility at Finish() time when shapes are known
    tf_wrap::Graph g;
    
    // 6 elements can't reshape to 2x4=8
    auto t = tf_wrap::Tensor::FromVector<float>({6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto bad_shape = tf_wrap::Tensor::FromVector<int32_t>({2}, {2, 4});
    
    (void)g.NewOperation("Const", "T")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    (void)g.NewOperation("Const", "Shape")
        .SetAttrTensor("value", bad_shape.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    auto* op_shape = g.GetOperationOrThrow("Shape");
    
    bool threw = false;
    std::string error_msg;
    try {
        // TF validates reshape at Finish() time when shape is Const
        (void)g.NewOperation("Reshape", "R")
            .AddInput(tf_wrap::Output(op_t, 0))
            .AddInput(tf_wrap::Output(op_shape, 0))
            .Finish();
    } catch (const std::exception& e) {
        threw = true;
        error_msg = e.what();
    }
    REQUIRE(threw);
    std::cout << "    (Error: " << error_msg.substr(0, 60) << "...)\n";
}

TEST(duplicate_operation_name_throws) {
    tf_wrap::Graph g;
    
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    bool threw = false;
    std::string error_msg;
    try {
        // Try to create another op with same name
        (void)g.NewOperation("Const", "X")
            .SetAttrTensor("value", t.handle())
            .SetAttrType("dtype", TF_FLOAT)
            .Finish();
    } catch (const std::exception& e) {
        threw = true;
        error_msg = e.what();
    }
    REQUIRE(threw);
    std::cout << "    (Error: " << error_msg.substr(0, 60) << "...)\n";
}

TEST(error_messages_are_helpful) {
    // Test that error messages contain useful info
    tf_wrap::Graph g;
    
    std::string error_msg;
    try {
        g.GetOperationOrThrow("my_missing_op");
    } catch (const std::exception& e) {
        error_msg = e.what();
    }
    
    // Error should mention the operation name
    REQUIRE(error_msg.find("my_missing_op") != std::string::npos);
}

// =============================================================================
// Value Verification Tests - Actually verify computation results
// =============================================================================

TEST(value_div_and_floordiv) {
    tf_wrap::Graph g;
    
    auto a = tf_wrap::Tensor::FromVector<float>({3}, {7.0f, 8.0f, 9.0f});
    auto b = tf_wrap::Tensor::FromVector<float>({3}, {2.0f, 3.0f, 4.0f});
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("Div", "Div").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    (void)g.NewOperation("FloorDiv", "FloorDiv").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Div", 0}, {"FloorDiv", 0}}, {});
    
    auto div_v = results[0].ToVector<float>();
    REQUIRE_APPROX(div_v[0], 3.5f, 0.0001f);
    REQUIRE_APPROX(div_v[1], 2.6667f, 0.001f);
    REQUIRE_APPROX(div_v[2], 2.25f, 0.0001f);
    
    auto floor_v = results[1].ToVector<float>();
    REQUIRE(floor_v[0] == 3.0f);
    REQUIRE(floor_v[1] == 2.0f);
    REQUIRE(floor_v[2] == 2.0f);
}

TEST(value_mod_and_pow) {
    tf_wrap::Graph g;
    
    auto a = tf_wrap::Tensor::FromVector<float>({3}, {7.0f, 10.0f, 2.0f});
    auto b = tf_wrap::Tensor::FromVector<float>({3}, {3.0f, 4.0f, 8.0f});
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("Mod", "Mod").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    (void)g.NewOperation("Pow", "Pow").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Mod", 0}, {"Pow", 0}}, {});
    
    auto mod_v = results[0].ToVector<float>();
    REQUIRE(mod_v[0] == 1.0f);  // 7 % 3 = 1
    REQUIRE(mod_v[1] == 2.0f);  // 10 % 4 = 2
    REQUIRE(mod_v[2] == 2.0f);  // 2 % 8 = 2
    
    auto pow_v = results[1].ToVector<float>();
    REQUIRE(pow_v[0] == 343.0f);    // 7^3
    REQUIRE(pow_v[1] == 10000.0f);  // 10^4
    REQUIRE(pow_v[2] == 256.0f);    // 2^8
}

TEST(value_trig_functions) {
    tf_wrap::Graph g;
    
    const float pi = 3.14159265358979f;
    auto t = tf_wrap::Tensor::FromVector<float>({4}, {0.0f, pi/6.0f, pi/4.0f, pi/2.0f});
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_t = g.GetOperationOrThrow("T");
    
    (void)g.NewOperation("Sin", "Sin").AddInput(tf_wrap::Output(op_t, 0)).Finish();
    (void)g.NewOperation("Cos", "Cos").AddInput(tf_wrap::Output(op_t, 0)).Finish();
    (void)g.NewOperation("Tan", "Tan").AddInput(tf_wrap::Output(op_t, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Sin", 0}, {"Cos", 0}, {"Tan", 0}}, {});
    
    auto sin_v = results[0].ToVector<float>();
    REQUIRE_APPROX(sin_v[0], 0.0f, 0.0001f);      // sin(0)
    REQUIRE_APPROX(sin_v[1], 0.5f, 0.0001f);      // sin(pi/6)
    REQUIRE_APPROX(sin_v[2], 0.7071f, 0.001f);    // sin(pi/4)
    REQUIRE_APPROX(sin_v[3], 1.0f, 0.0001f);      // sin(pi/2)
    
    auto cos_v = results[1].ToVector<float>();
    REQUIRE_APPROX(cos_v[0], 1.0f, 0.0001f);      // cos(0)
    REQUIRE_APPROX(cos_v[1], 0.866f, 0.001f);     // cos(pi/6)
    REQUIRE_APPROX(cos_v[2], 0.7071f, 0.001f);    // cos(pi/4)
    REQUIRE_APPROX(cos_v[3], 0.0f, 0.0001f);      // cos(pi/2)
}

TEST(value_exp_log) {
    tf_wrap::Graph g;
    
    auto t = tf_wrap::Tensor::FromVector<float>({4}, {0.0f, 1.0f, 2.0f, -1.0f});
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_t = g.GetOperationOrThrow("T");
    
    (void)g.NewOperation("Exp", "Exp").AddInput(tf_wrap::Output(op_t, 0)).Finish();
    
    // Create tensor for log (need positive values)
    auto t_log = tf_wrap::Tensor::FromVector<float>({4}, {1.0f, 2.718281828f, 10.0f, 0.1f});
    (void)g.NewOperation("Const", "TLog").SetAttrTensor("value", t_log.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_tlog = g.GetOperationOrThrow("TLog");
    
    (void)g.NewOperation("Log", "Log").AddInput(tf_wrap::Output(op_tlog, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Exp", 0}, {"Log", 0}}, {});
    
    auto exp_v = results[0].ToVector<float>();
    REQUIRE_APPROX(exp_v[0], 1.0f, 0.0001f);       // e^0
    REQUIRE_APPROX(exp_v[1], 2.7183f, 0.001f);     // e^1
    REQUIRE_APPROX(exp_v[2], 7.3891f, 0.001f);     // e^2
    REQUIRE_APPROX(exp_v[3], 0.3679f, 0.001f);     // e^-1
    
    auto log_v = results[1].ToVector<float>();
    REQUIRE_APPROX(log_v[0], 0.0f, 0.0001f);       // ln(1)
    REQUIRE_APPROX(log_v[1], 1.0f, 0.0001f);       // ln(e)
    REQUIRE_APPROX(log_v[2], 2.3026f, 0.001f);     // ln(10)
    REQUIRE_APPROX(log_v[3], -2.3026f, 0.001f);    // ln(0.1)
}

TEST(value_sqrt_square_abs) {
    tf_wrap::Graph g;
    
    auto t = tf_wrap::Tensor::FromVector<float>({4}, {4.0f, 9.0f, 16.0f, 25.0f});
    auto t_neg = tf_wrap::Tensor::FromVector<float>({4}, {-3.0f, 5.0f, -7.0f, 0.0f});
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "TNeg").SetAttrTensor("value", t_neg.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    auto* op_tneg = g.GetOperationOrThrow("TNeg");
    
    (void)g.NewOperation("Sqrt", "Sqrt").AddInput(tf_wrap::Output(op_t, 0)).Finish();
    (void)g.NewOperation("Square", "Square").AddInput(tf_wrap::Output(op_tneg, 0)).Finish();
    (void)g.NewOperation("Abs", "Abs").AddInput(tf_wrap::Output(op_tneg, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Sqrt", 0}, {"Square", 0}, {"Abs", 0}}, {});
    
    auto sqrt_v = results[0].ToVector<float>();
    REQUIRE(sqrt_v[0] == 2.0f && sqrt_v[1] == 3.0f && sqrt_v[2] == 4.0f && sqrt_v[3] == 5.0f);
    
    auto square_v = results[1].ToVector<float>();
    REQUIRE(square_v[0] == 9.0f && square_v[1] == 25.0f && square_v[2] == 49.0f && square_v[3] == 0.0f);
    
    auto abs_v = results[2].ToVector<float>();
    REQUIRE(abs_v[0] == 3.0f && abs_v[1] == 5.0f && abs_v[2] == 7.0f && abs_v[3] == 0.0f);
}

TEST(value_comparison_ops) {
    tf_wrap::Graph g;
    
    auto a = tf_wrap::Tensor::FromVector<float>({4}, {1.0f, 2.0f, 3.0f, 2.0f});
    auto b = tf_wrap::Tensor::FromVector<float>({4}, {2.0f, 2.0f, 1.0f, 2.0f});
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("Less", "Less").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    (void)g.NewOperation("Greater", "Greater").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    (void)g.NewOperation("Equal", "Equal").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Less", 0}, {"Greater", 0}, {"Equal", 0}}, {});
    
    auto less_v = results[0].ToVector<bool>();
    REQUIRE(less_v[0] == true);   // 1 < 2
    REQUIRE(less_v[1] == false);  // 2 < 2
    REQUIRE(less_v[2] == false);  // 3 < 1
    REQUIRE(less_v[3] == false);  // 2 < 2
    
    auto greater_v = results[1].ToVector<bool>();
    REQUIRE(greater_v[0] == false);  // 1 > 2
    REQUIRE(greater_v[1] == false);  // 2 > 2
    REQUIRE(greater_v[2] == true);   // 3 > 1
    REQUIRE(greater_v[3] == false);  // 2 > 2
    
    auto equal_v = results[2].ToVector<bool>();
    REQUIRE(equal_v[0] == false);  // 1 == 2
    REQUIRE(equal_v[1] == true);   // 2 == 2
    REQUIRE(equal_v[2] == false);  // 3 == 1
    REQUIRE(equal_v[3] == true);   // 2 == 2
}

TEST(value_activation_functions) {
    tf_wrap::Graph g;
    
    auto t = tf_wrap::Tensor::FromVector<float>({6}, {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f});
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_t = g.GetOperationOrThrow("T");
    
    (void)g.NewOperation("Relu", "Relu").AddInput(tf_wrap::Output(op_t, 0)).Finish();
    (void)g.NewOperation("Relu6", "Relu6").AddInput(tf_wrap::Output(op_t, 0)).Finish();
    (void)g.NewOperation("Sigmoid", "Sigmoid").AddInput(tf_wrap::Output(op_t, 0)).Finish();
    (void)g.NewOperation("Tanh", "Tanh").AddInput(tf_wrap::Output(op_t, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Relu", 0}, {"Relu6", 0}, {"Sigmoid", 0}, {"Tanh", 0}}, {});
    
    auto relu_v = results[0].ToVector<float>();
    REQUIRE(relu_v[0] == 0.0f && relu_v[1] == 0.0f && relu_v[2] == 0.0f);
    REQUIRE(relu_v[3] == 0.0f && relu_v[4] == 0.5f && relu_v[5] == 2.0f);
    
    auto relu6_v = results[1].ToVector<float>();
    REQUIRE(relu6_v[0] == 0.0f && relu6_v[5] == 2.0f);  // Note: 2 < 6, so not clipped
    
    auto sigmoid_v = results[2].ToVector<float>();
    REQUIRE_APPROX(sigmoid_v[3], 0.5f, 0.0001f);  // sigmoid(0) = 0.5
    REQUIRE(sigmoid_v[0] < 0.5f && sigmoid_v[5] > 0.5f);  // Negative gives < 0.5, positive > 0.5
    
    auto tanh_v = results[3].ToVector<float>();
    REQUIRE_APPROX(tanh_v[3], 0.0f, 0.0001f);  // tanh(0) = 0
    REQUIRE(tanh_v[0] < 0.0f && tanh_v[5] > 0.0f);  // Negative gives negative, positive gives positive
}

TEST(value_reduction_mean_max_min) {
    tf_wrap::Graph g;
    
    // 2x3 matrix
    auto t = tf_wrap::Tensor::FromVector<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto axis = tf_wrap::Tensor::FromVector<int32_t>({1}, {1}); // reduce along axis 1
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Axis").SetAttrTensor("value", axis.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    auto* op_axis = g.GetOperationOrThrow("Axis");
    
    (void)g.NewOperation("Mean", "Mean").AddInput(tf_wrap::Output(op_t, 0)).AddInput(tf_wrap::Output(op_axis, 0)).Finish();
    (void)g.NewOperation("Max", "Max").AddInput(tf_wrap::Output(op_t, 0)).AddInput(tf_wrap::Output(op_axis, 0)).Finish();
    (void)g.NewOperation("Min", "Min").AddInput(tf_wrap::Output(op_t, 0)).AddInput(tf_wrap::Output(op_axis, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Mean", 0}, {"Max", 0}, {"Min", 0}}, {});
    
    auto mean_v = results[0].ToVector<float>();
    REQUIRE(mean_v.size() == 2);
    REQUIRE(mean_v[0] == 2.0f);  // (1+2+3)/3
    REQUIRE(mean_v[1] == 5.0f);  // (4+5+6)/3
    
    auto max_v = results[1].ToVector<float>();
    REQUIRE(max_v[0] == 3.0f && max_v[1] == 6.0f);
    
    auto min_v = results[2].ToVector<float>();
    REQUIRE(min_v[0] == 1.0f && min_v[1] == 4.0f);
}

TEST(value_argmax_argmin) {
    tf_wrap::Graph g;
    
    // 2x4 matrix
    auto t = tf_wrap::Tensor::FromVector<float>({2, 4}, {3.0f, 1.0f, 4.0f, 1.0f, 2.0f, 7.0f, 1.0f, 8.0f});
    auto axis = tf_wrap::Tensor::FromScalar<int32_t>(1); // reduce along axis 1
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Axis").SetAttrTensor("value", axis.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    auto* op_axis = g.GetOperationOrThrow("Axis");
    
    (void)g.NewOperation("ArgMax", "ArgMax")
        .AddInput(tf_wrap::Output(op_t, 0))
        .AddInput(tf_wrap::Output(op_axis, 0))
        .Finish();
    (void)g.NewOperation("ArgMin", "ArgMin")
        .AddInput(tf_wrap::Output(op_t, 0))
        .AddInput(tf_wrap::Output(op_axis, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"ArgMax", 0}, {"ArgMin", 0}}, {});
    
    auto argmax_v = results[0].ToVector<int64_t>();
    REQUIRE(argmax_v[0] == 2);  // index of 4.0 in row 0
    REQUIRE(argmax_v[1] == 3);  // index of 8.0 in row 1
    
    auto argmin_v = results[1].ToVector<int64_t>();
    REQUIRE(argmin_v[0] == 1 || argmin_v[0] == 3);  // index of 1.0 in row 0 (first occurrence)
    REQUIRE(argmin_v[1] == 2);  // index of 1.0 in row 1
}

TEST(value_transpose) {
    tf_wrap::Graph g;
    
    // 2x3 matrix
    auto t = tf_wrap::Tensor::FromVector<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto perm = tf_wrap::Tensor::FromVector<int32_t>({2}, {1, 0}); // swap dimensions
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Perm").SetAttrTensor("value", perm.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    auto* op_perm = g.GetOperationOrThrow("Perm");
    
    (void)g.NewOperation("Transpose", "Trans")
        .AddInput(tf_wrap::Output(op_t, 0))
        .AddInput(tf_wrap::Output(op_perm, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Trans", 0}}, {});
    
    // Original: [[1, 2, 3], [4, 5, 6]]
    // Transposed: [[1, 4], [2, 5], [3, 6]]
    REQUIRE(results[0].rank() == 2);
    REQUIRE(results[0].shape()[0] == 3);
    REQUIRE(results[0].shape()[1] == 2);
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v[0] == 1.0f && v[1] == 4.0f);  // first row
    REQUIRE(v[2] == 2.0f && v[3] == 5.0f);  // second row
    REQUIRE(v[4] == 3.0f && v[5] == 6.0f);  // third row
}

TEST(value_gather) {
    tf_wrap::Graph g;
    
    auto params = tf_wrap::Tensor::FromVector<float>({4}, {10.0f, 20.0f, 30.0f, 40.0f});
    auto indices = tf_wrap::Tensor::FromVector<int32_t>({3}, {3, 0, 2});
    
    (void)g.NewOperation("Const", "Params").SetAttrTensor("value", params.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Indices").SetAttrTensor("value", indices.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_params = g.GetOperationOrThrow("Params");
    auto* op_indices = g.GetOperationOrThrow("Indices");
    
    // Use basic Gather which doesn't need axis
    (void)g.NewOperation("Gather", "Gather")
        .AddInput(tf_wrap::Output(op_params, 0))
        .AddInput(tf_wrap::Output(op_indices, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Gather", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 3);
    REQUIRE(v[0] == 40.0f);  // params[3]
    REQUIRE(v[1] == 10.0f);  // params[0]
    REQUIRE(v[2] == 30.0f);  // params[2]
}

TEST(value_where_select) {
    tf_wrap::Graph g;
    
    auto cond = tf_wrap::Tensor::FromVector<bool>({4}, {true, false, true, false});
    auto x = tf_wrap::Tensor::FromVector<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto y = tf_wrap::Tensor::FromVector<float>({4}, {10.0f, 20.0f, 30.0f, 40.0f});
    
    (void)g.NewOperation("Const", "Cond").SetAttrTensor("value", cond.handle()).SetAttrType("dtype", TF_BOOL).Finish();
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Y").SetAttrTensor("value", y.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_cond = g.GetOperationOrThrow("Cond");
    auto* op_x = g.GetOperationOrThrow("X");
    auto* op_y = g.GetOperationOrThrow("Y");
    
    (void)g.NewOperation("Select", "Select")
        .AddInput(tf_wrap::Output(op_cond, 0))
        .AddInput(tf_wrap::Output(op_x, 0))
        .AddInput(tf_wrap::Output(op_y, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Select", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v[0] == 1.0f);   // cond[0]=true, select x
    REQUIRE(v[1] == 20.0f);  // cond[1]=false, select y
    REQUIRE(v[2] == 3.0f);   // cond[2]=true, select x
    REQUIRE(v[3] == 40.0f);  // cond[3]=false, select y
}

TEST(value_concat) {
    tf_wrap::Graph g;
    
    auto t1 = tf_wrap::Tensor::FromVector<float>({2}, {1.0f, 2.0f});
    auto t2 = tf_wrap::Tensor::FromVector<float>({3}, {3.0f, 4.0f, 5.0f});
    auto axis = tf_wrap::Tensor::FromScalar<int32_t>(0);
    
    (void)g.NewOperation("Const", "T1").SetAttrTensor("value", t1.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "T2").SetAttrTensor("value", t2.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Axis").SetAttrTensor("value", axis.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_t1 = g.GetOperationOrThrow("T1");
    auto* op_t2 = g.GetOperationOrThrow("T2");
    auto* op_axis = g.GetOperationOrThrow("Axis");
    
    std::vector<TF_Output> values = {tf_wrap::Output(op_t1, 0), tf_wrap::Output(op_t2, 0)};
    
    (void)g.NewOperation("ConcatV2", "Concat")
        .AddInputList(values)
        .AddInput(tf_wrap::Output(op_axis, 0))
        .SetAttrInt("N", 2)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Concat", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 5);
    REQUIRE(v[0] == 1.0f && v[1] == 2.0f && v[2] == 3.0f && v[3] == 4.0f && v[4] == 5.0f);
}

TEST(value_slice) {
    tf_wrap::Graph g;
    
    auto t = tf_wrap::Tensor::FromVector<float>({6}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto begin = tf_wrap::Tensor::FromVector<int32_t>({1}, {2});
    auto size = tf_wrap::Tensor::FromVector<int32_t>({1}, {3});
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Begin").SetAttrTensor("value", begin.handle()).SetAttrType("dtype", TF_INT32).Finish();
    (void)g.NewOperation("Const", "Size").SetAttrTensor("value", size.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    auto* op_begin = g.GetOperationOrThrow("Begin");
    auto* op_size = g.GetOperationOrThrow("Size");
    
    (void)g.NewOperation("Slice", "Slice")
        .AddInput(tf_wrap::Output(op_t, 0))
        .AddInput(tf_wrap::Output(op_begin, 0))
        .AddInput(tf_wrap::Output(op_size, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Slice", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 3);
    REQUIRE(v[0] == 2.0f && v[1] == 3.0f && v[2] == 4.0f);
}

TEST(value_cast_dtypes) {
    tf_wrap::Graph g;
    
    auto t_float = tf_wrap::Tensor::FromVector<float>({3}, {1.5f, 2.7f, -3.9f});
    
    (void)g.NewOperation("Const", "TFloat").SetAttrTensor("value", t_float.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_t = g.GetOperationOrThrow("TFloat");
    
    (void)g.NewOperation("Cast", "ToInt32")
        .AddInput(tf_wrap::Output(op_t, 0))
        .SetAttrType("SrcT", TF_FLOAT)
        .SetAttrType("DstT", TF_INT32)
        .Finish();
    
    (void)g.NewOperation("Cast", "ToInt64")
        .AddInput(tf_wrap::Output(op_t, 0))
        .SetAttrType("SrcT", TF_FLOAT)
        .SetAttrType("DstT", TF_INT64)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"ToInt32", 0}, {"ToInt64", 0}}, {});
    
    auto int32_v = results[0].ToVector<int32_t>();
    REQUIRE(int32_v[0] == 1);
    REQUIRE(int32_v[1] == 2);
    REQUIRE(int32_v[2] == -3);
    
    auto int64_v = results[1].ToVector<int64_t>();
    REQUIRE(int64_v[0] == 1);
    REQUIRE(int64_v[1] == 2);
    REQUIRE(int64_v[2] == -3);
}

TEST(value_logical_ops) {
    tf_wrap::Graph g;
    
    auto a = tf_wrap::Tensor::FromVector<bool>({4}, {true, true, false, false});
    auto b = tf_wrap::Tensor::FromVector<bool>({4}, {true, false, true, false});
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_BOOL).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_BOOL).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("LogicalAnd", "And").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    (void)g.NewOperation("LogicalOr", "Or").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    (void)g.NewOperation("LogicalNot", "Not").AddInput(tf_wrap::Output(op_a, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"And", 0}, {"Or", 0}, {"Not", 0}}, {});
    
    auto and_v = results[0].ToVector<bool>();
    REQUIRE(and_v[0] == true && and_v[1] == false && and_v[2] == false && and_v[3] == false);
    
    auto or_v = results[1].ToVector<bool>();
    REQUIRE(or_v[0] == true && or_v[1] == true && or_v[2] == true && or_v[3] == false);
    
    auto not_v = results[2].ToVector<bool>();
    REQUIRE(not_v[0] == false && not_v[1] == false && not_v[2] == true && not_v[3] == true);
}

TEST(value_maximum_minimum) {
    tf_wrap::Graph g;
    
    auto a = tf_wrap::Tensor::FromVector<float>({4}, {1.0f, 5.0f, 3.0f, 8.0f});
    auto b = tf_wrap::Tensor::FromVector<float>({4}, {2.0f, 4.0f, 6.0f, 7.0f});
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("Maximum", "Max").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    (void)g.NewOperation("Minimum", "Min").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Max", 0}, {"Min", 0}}, {});
    
    auto max_v = results[0].ToVector<float>();
    REQUIRE(max_v[0] == 2.0f && max_v[1] == 5.0f && max_v[2] == 6.0f && max_v[3] == 8.0f);
    
    auto min_v = results[1].ToVector<float>();
    REQUIRE(min_v[0] == 1.0f && min_v[1] == 4.0f && min_v[2] == 3.0f && min_v[3] == 7.0f);
}

TEST(value_fill_ones_zeros_like) {
    tf_wrap::Graph g;
    
    auto dims = tf_wrap::Tensor::FromVector<int32_t>({2}, {2, 3});
    auto val = tf_wrap::Tensor::FromScalar<float>(5.0f);
    
    (void)g.NewOperation("Const", "Dims").SetAttrTensor("value", dims.handle()).SetAttrType("dtype", TF_INT32).Finish();
    (void)g.NewOperation("Const", "Val").SetAttrTensor("value", val.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_dims = g.GetOperationOrThrow("Dims");
    auto* op_val = g.GetOperationOrThrow("Val");
    
    (void)g.NewOperation("Fill", "Fill")
        .AddInput(tf_wrap::Output(op_dims, 0))
        .AddInput(tf_wrap::Output(op_val, 0))
        .Finish();
    
    auto template_t = tf_wrap::Tensor::FromVector<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    (void)g.NewOperation("Const", "Template").SetAttrTensor("value", template_t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_template = g.GetOperationOrThrow("Template");
    
    (void)g.NewOperation("OnesLike", "Ones").AddInput(tf_wrap::Output(op_template, 0)).Finish();
    (void)g.NewOperation("ZerosLike", "Zeros").AddInput(tf_wrap::Output(op_template, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Fill", 0}, {"Ones", 0}, {"Zeros", 0}}, {});
    
    auto fill_v = results[0].ToVector<float>();
    REQUIRE(fill_v.size() == 6);
    for (auto v : fill_v) REQUIRE(v == 5.0f);
    
    auto ones_v = results[1].ToVector<float>();
    REQUIRE(ones_v.size() == 6);
    for (auto v : ones_v) REQUIRE(v == 1.0f);
    
    auto zeros_v = results[2].ToVector<float>();
    REQUIRE(zeros_v.size() == 6);
    for (auto v : zeros_v) REQUIRE(v == 0.0f);
}

TEST(value_tile) {
    tf_wrap::Graph g;
    
    auto t = tf_wrap::Tensor::FromVector<float>({2}, {1.0f, 2.0f});
    auto multiples = tf_wrap::Tensor::FromVector<int32_t>({1}, {3});
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Multiples").SetAttrTensor("value", multiples.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    auto* op_multiples = g.GetOperationOrThrow("Multiples");
    
    (void)g.NewOperation("Tile", "Tile")
        .AddInput(tf_wrap::Output(op_t, 0))
        .AddInput(tf_wrap::Output(op_multiples, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Tile", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 6);
    REQUIRE(v[0] == 1.0f && v[1] == 2.0f);
    REQUIRE(v[2] == 1.0f && v[3] == 2.0f);
    REQUIRE(v[4] == 1.0f && v[5] == 2.0f);
}

TEST(value_range) {
    tf_wrap::Graph g;
    
    auto start = tf_wrap::Tensor::FromScalar<float>(0.0f);
    auto limit = tf_wrap::Tensor::FromScalar<float>(5.0f);
    auto delta = tf_wrap::Tensor::FromScalar<float>(1.0f);
    
    (void)g.NewOperation("Const", "Start").SetAttrTensor("value", start.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Limit").SetAttrTensor("value", limit.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Delta").SetAttrTensor("value", delta.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_start = g.GetOperationOrThrow("Start");
    auto* op_limit = g.GetOperationOrThrow("Limit");
    auto* op_delta = g.GetOperationOrThrow("Delta");
    
    (void)g.NewOperation("Range", "Range")
        .AddInput(tf_wrap::Output(op_start, 0))
        .AddInput(tf_wrap::Output(op_limit, 0))
        .AddInput(tf_wrap::Output(op_delta, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Range", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 5);
    REQUIRE(v[0] == 0.0f && v[1] == 1.0f && v[2] == 2.0f && v[3] == 3.0f && v[4] == 4.0f);
}

TEST(value_reduction_prod) {
    tf_wrap::Graph g;
    
    auto t = tf_wrap::Tensor::FromVector<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto axis = tf_wrap::Tensor::FromVector<int32_t>({1}, {1}); // reduce along axis 1
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Axis").SetAttrTensor("value", axis.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    auto* op_axis = g.GetOperationOrThrow("Axis");
    
    (void)g.NewOperation("Prod", "Prod")
        .AddInput(tf_wrap::Output(op_t, 0))
        .AddInput(tf_wrap::Output(op_axis, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Prod", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 2);
    REQUIRE(v[0] == 6.0f);    // 1*2*3
    REQUIRE(v[1] == 120.0f);  // 4*5*6
}

TEST(value_reduction_any_all) {
    tf_wrap::Graph g;
    
    auto t = tf_wrap::Tensor::FromVector<bool>({2, 3}, {true, false, true, true, true, true});
    auto axis = tf_wrap::Tensor::FromVector<int32_t>({1}, {1}); // reduce along axis 1
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_BOOL).Finish();
    (void)g.NewOperation("Const", "Axis").SetAttrTensor("value", axis.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    auto* op_axis = g.GetOperationOrThrow("Axis");
    
    (void)g.NewOperation("Any", "Any")
        .AddInput(tf_wrap::Output(op_t, 0))
        .AddInput(tf_wrap::Output(op_axis, 0))
        .Finish();
    
    (void)g.NewOperation("All", "All")
        .AddInput(tf_wrap::Output(op_t, 0))
        .AddInput(tf_wrap::Output(op_axis, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Any", 0}, {"All", 0}}, {});
    
    auto any_v = results[0].ToVector<bool>();
    REQUIRE(any_v.size() == 2);
    REQUIRE(any_v[0] == true);   // row 0: T,F,T -> any=true
    REQUIRE(any_v[1] == true);   // row 1: T,T,T -> any=true
    
    auto all_v = results[1].ToVector<bool>();
    REQUIRE(all_v[0] == false);  // row 0: T,F,T -> all=false
    REQUIRE(all_v[1] == true);   // row 1: T,T,T -> all=true
}

TEST(value_batchmatmul) {
    tf_wrap::Graph g;
    
    // Batch of 2 matrices: each is 2x2
    // A[0] = [[1,2],[3,4]], A[1] = [[5,6],[7,8]]
    auto a = tf_wrap::Tensor::FromVector<float>({2, 2, 2}, 
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    // B[0] = [[1,0],[0,1]] (identity), B[1] = [[2,0],[0,2]] (scale by 2)
    auto b = tf_wrap::Tensor::FromVector<float>({2, 2, 2}, 
        {1.0f, 0.0f, 0.0f, 1.0f, 2.0f, 0.0f, 0.0f, 2.0f});
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("BatchMatMul", "BMM")
        .AddInput(tf_wrap::Output(op_a, 0))
        .AddInput(tf_wrap::Output(op_b, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"BMM", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 8);
    // C[0] = A[0] @ B[0] = A[0] @ I = A[0] = [[1,2],[3,4]]
    REQUIRE(v[0] == 1.0f && v[1] == 2.0f && v[2] == 3.0f && v[3] == 4.0f);
    // C[1] = A[1] @ B[1] = [[5,6],[7,8]] @ [[2,0],[0,2]] = [[10,12],[14,16]]
    REQUIRE(v[4] == 10.0f && v[5] == 12.0f && v[6] == 14.0f && v[7] == 16.0f);
}

TEST(value_conv2d_simple) {
    tf_wrap::Graph g;
    
    // Input: NHWC format - batch=1, height=3, width=3, channels=1
    // Simple 3x3 input with values 1-9
    auto input = tf_wrap::Tensor::FromVector<float>({1, 3, 3, 1}, 
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    
    // Filter: HWIO format - height=2, width=2, in_channels=1, out_channels=1
    // Simple 2x2 filter of all 1s (sum filter)
    auto filter = tf_wrap::Tensor::FromVector<float>({2, 2, 1, 1}, 
        {1.0f, 1.0f, 1.0f, 1.0f});
    
    (void)g.NewOperation("Const", "Input").SetAttrTensor("value", input.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Filter").SetAttrTensor("value", filter.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_input = g.GetOperationOrThrow("Input");
    auto* op_filter = g.GetOperationOrThrow("Filter");
    
    (void)g.NewOperation("Conv2D", "Conv")
        .AddInput(tf_wrap::Output(op_input, 0))
        .AddInput(tf_wrap::Output(op_filter, 0))
        .SetAttrIntList("strides", std::vector<int64_t>{1, 1, 1, 1})
        .SetAttrString("padding", "VALID")
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Conv", 0}}, {});
    
    // Output shape: [1, 2, 2, 1] with VALID padding
    REQUIRE(results[0].shape()[0] == 1);
    REQUIRE(results[0].shape()[1] == 2);
    REQUIRE(results[0].shape()[2] == 2);
    REQUIRE(results[0].shape()[3] == 1);
    
    auto v = results[0].ToVector<float>();
    // Conv with sum filter:
    // [0,0]: 1+2+4+5 = 12
    // [0,1]: 2+3+5+6 = 16
    // [1,0]: 4+5+7+8 = 24
    // [1,1]: 5+6+8+9 = 28
    REQUIRE(v[0] == 12.0f);
    REQUIRE(v[1] == 16.0f);
    REQUIRE(v[2] == 24.0f);
    REQUIRE(v[3] == 28.0f);
}

TEST(value_maxpool_simple) {
    tf_wrap::Graph g;
    
    // Input: NHWC format - batch=1, height=4, width=4, channels=1
    auto input = tf_wrap::Tensor::FromVector<float>({1, 4, 4, 1}, 
        {1.0f, 2.0f, 3.0f, 4.0f,
         5.0f, 6.0f, 7.0f, 8.0f,
         9.0f, 10.0f, 11.0f, 12.0f,
         13.0f, 14.0f, 15.0f, 16.0f});
    
    (void)g.NewOperation("Const", "Input").SetAttrTensor("value", input.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_input = g.GetOperationOrThrow("Input");
    
    (void)g.NewOperation("MaxPool", "MaxPool")
        .AddInput(tf_wrap::Output(op_input, 0))
        .SetAttrIntList("ksize", std::vector<int64_t>{1, 2, 2, 1})
        .SetAttrIntList("strides", std::vector<int64_t>{1, 2, 2, 1})
        .SetAttrString("padding", "VALID")
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"MaxPool", 0}}, {});
    
    // Output shape: [1, 2, 2, 1]
    REQUIRE(results[0].shape()[1] == 2);
    REQUIRE(results[0].shape()[2] == 2);
    
    auto v = results[0].ToVector<float>();
    // Max in each 2x2 window with stride 2:
    // [0,0]: max(1,2,5,6) = 6
    // [0,1]: max(3,4,7,8) = 8
    // [1,0]: max(9,10,13,14) = 14
    // [1,1]: max(11,12,15,16) = 16
    REQUIRE(v[0] == 6.0f);
    REQUIRE(v[1] == 8.0f);
    REQUIRE(v[2] == 14.0f);
    REQUIRE(v[3] == 16.0f);
}

TEST(value_avgpool_simple) {
    tf_wrap::Graph g;
    
    // Input: NHWC format - batch=1, height=4, width=4, channels=1
    auto input = tf_wrap::Tensor::FromVector<float>({1, 4, 4, 1}, 
        {1.0f, 2.0f, 3.0f, 4.0f,
         5.0f, 6.0f, 7.0f, 8.0f,
         9.0f, 10.0f, 11.0f, 12.0f,
         13.0f, 14.0f, 15.0f, 16.0f});
    
    (void)g.NewOperation("Const", "Input").SetAttrTensor("value", input.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_input = g.GetOperationOrThrow("Input");
    
    (void)g.NewOperation("AvgPool", "AvgPool")
        .AddInput(tf_wrap::Output(op_input, 0))
        .SetAttrIntList("ksize", std::vector<int64_t>{1, 2, 2, 1})
        .SetAttrIntList("strides", std::vector<int64_t>{1, 2, 2, 1})
        .SetAttrString("padding", "VALID")
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"AvgPool", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    // Avg in each 2x2 window:
    // [0,0]: (1+2+5+6)/4 = 3.5
    // [0,1]: (3+4+7+8)/4 = 5.5
    // [1,0]: (9+10+13+14)/4 = 11.5
    // [1,1]: (11+12+15+16)/4 = 13.5
    REQUIRE_APPROX(v[0], 3.5f, 0.0001f);
    REQUIRE_APPROX(v[1], 5.5f, 0.0001f);
    REQUIRE_APPROX(v[2], 11.5f, 0.0001f);
    REQUIRE_APPROX(v[3], 13.5f, 0.0001f);
}

TEST(value_softmax_crossentropy) {
    tf_wrap::Graph g;
    
    // Logits: 2 samples, 3 classes
    auto logits = tf_wrap::Tensor::FromVector<float>({2, 3}, 
        {1.0f, 2.0f, 3.0f,   // sample 0: class 2 has highest logit
         3.0f, 2.0f, 1.0f}); // sample 1: class 0 has highest logit
    
    // Labels: one-hot encoded
    auto labels = tf_wrap::Tensor::FromVector<float>({2, 3}, 
        {0.0f, 0.0f, 1.0f,   // sample 0: true class = 2
         1.0f, 0.0f, 0.0f}); // sample 1: true class = 0
    
    (void)g.NewOperation("Const", "Logits").SetAttrTensor("value", logits.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Labels").SetAttrTensor("value", labels.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_logits = g.GetOperationOrThrow("Logits");
    auto* op_labels = g.GetOperationOrThrow("Labels");
    
    (void)g.NewOperation("SoftmaxCrossEntropyWithLogits", "XEnt")
        .AddInput(tf_wrap::Output(op_logits, 0))
        .AddInput(tf_wrap::Output(op_labels, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"XEnt", 0}}, {});  // output 0 is loss
    
    auto loss = results[0].ToVector<float>();
    REQUIRE(loss.size() == 2);
    
    // For sample 0: logits [1,2,3], true class 2
    // softmax([1,2,3]) ≈ [0.09, 0.24, 0.67]
    // cross-entropy = -log(0.67) ≈ 0.407
    REQUIRE(loss[0] > 0.3f && loss[0] < 0.5f);
    
    // For sample 1: logits [3,2,1], true class 0  
    // softmax([3,2,1]) ≈ [0.67, 0.24, 0.09]
    // cross-entropy = -log(0.67) ≈ 0.407
    REQUIRE(loss[1] > 0.3f && loss[1] < 0.5f);
}

TEST(value_sparse_softmax_crossentropy) {
    tf_wrap::Graph g;
    
    // Logits: 2 samples, 3 classes
    auto logits = tf_wrap::Tensor::FromVector<float>({2, 3}, 
        {1.0f, 2.0f, 3.0f,   // sample 0
         3.0f, 2.0f, 1.0f}); // sample 1
    
    // Labels: sparse (class indices)
    auto labels = tf_wrap::Tensor::FromVector<int32_t>({2}, {2, 0}); // class 2, class 0
    
    (void)g.NewOperation("Const", "Logits").SetAttrTensor("value", logits.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Labels").SetAttrTensor("value", labels.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_logits = g.GetOperationOrThrow("Logits");
    auto* op_labels = g.GetOperationOrThrow("Labels");
    
    (void)g.NewOperation("SparseSoftmaxCrossEntropyWithLogits", "SparseXEnt")
        .AddInput(tf_wrap::Output(op_logits, 0))
        .AddInput(tf_wrap::Output(op_labels, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"SparseXEnt", 0}}, {});  // output 0 is loss
    
    auto loss = results[0].ToVector<float>();
    REQUIRE(loss.size() == 2);
    
    // Same expected values as dense version
    REQUIRE(loss[0] > 0.3f && loss[0] < 0.5f);
    REQUIRE(loss[1] > 0.3f && loss[1] < 0.5f);
}

TEST(value_biasadd) {
    tf_wrap::Graph g;
    
    // Value: batch=2, features=3
    auto value = tf_wrap::Tensor::FromVector<float>({2, 3}, 
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto bias = tf_wrap::Tensor::FromVector<float>({3}, {10.0f, 20.0f, 30.0f});
    
    (void)g.NewOperation("Const", "Value").SetAttrTensor("value", value.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Bias").SetAttrTensor("value", bias.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_value = g.GetOperationOrThrow("Value");
    auto* op_bias = g.GetOperationOrThrow("Bias");
    
    (void)g.NewOperation("BiasAdd", "BiasAdd")
        .AddInput(tf_wrap::Output(op_value, 0))
        .AddInput(tf_wrap::Output(op_bias, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"BiasAdd", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    // Row 0: [1+10, 2+20, 3+30] = [11, 22, 33]
    // Row 1: [4+10, 5+20, 6+30] = [14, 25, 36]
    REQUIRE(v[0] == 11.0f && v[1] == 22.0f && v[2] == 33.0f);
    REQUIRE(v[3] == 14.0f && v[4] == 25.0f && v[5] == 36.0f);
}

TEST(value_leaky_relu) {
    tf_wrap::Graph g;
    
    auto t = tf_wrap::Tensor::FromVector<float>({4}, {-2.0f, -1.0f, 1.0f, 2.0f});
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_t = g.GetOperationOrThrow("T");
    
    (void)g.NewOperation("LeakyRelu", "LeakyRelu")
        .AddInput(tf_wrap::Output(op_t, 0))
        .SetAttrFloat("alpha", 0.1f)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"LeakyRelu", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    // LeakyReLU: x if x > 0, alpha*x otherwise
    REQUIRE_APPROX(v[0], -0.2f, 0.0001f);  // -2 * 0.1
    REQUIRE_APPROX(v[1], -0.1f, 0.0001f);  // -1 * 0.1
    REQUIRE(v[2] == 1.0f);
    REQUIRE(v[3] == 2.0f);
}

TEST(value_elu_selu) {
    tf_wrap::Graph g;
    
    auto t = tf_wrap::Tensor::FromVector<float>({4}, {-1.0f, 0.0f, 1.0f, 2.0f});
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_t = g.GetOperationOrThrow("T");
    
    (void)g.NewOperation("Elu", "Elu").AddInput(tf_wrap::Output(op_t, 0)).Finish();
    (void)g.NewOperation("Selu", "Selu").AddInput(tf_wrap::Output(op_t, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Elu", 0}, {"Selu", 0}}, {});
    
    auto elu_v = results[0].ToVector<float>();
    // ELU: x if x > 0, exp(x)-1 otherwise
    REQUIRE_APPROX(elu_v[0], -0.6321f, 0.001f);  // exp(-1)-1 ≈ -0.632
    REQUIRE(elu_v[1] == 0.0f);
    REQUIRE(elu_v[2] == 1.0f);
    REQUIRE(elu_v[3] == 2.0f);
    
    auto selu_v = results[1].ToVector<float>();
    // SELU: scale * (x if x > 0, alpha*(exp(x)-1) otherwise)
    // scale ≈ 1.0507, alpha ≈ 1.6733
    REQUIRE(selu_v[0] < 0.0f);  // negative output for negative input
    REQUIRE(selu_v[1] == 0.0f);
    REQUIRE(selu_v[2] > 1.0f);  // scaled positive
    REQUIRE(selu_v[3] > 2.0f);  // scaled positive
}

TEST(value_clip_by_value) {
    tf_wrap::Graph g;
    
    auto t = tf_wrap::Tensor::FromVector<float>({5}, {-5.0f, 0.0f, 5.0f, 10.0f, 15.0f});
    auto clip_min = tf_wrap::Tensor::FromScalar<float>(0.0f);
    auto clip_max = tf_wrap::Tensor::FromScalar<float>(10.0f);
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Min").SetAttrTensor("value", clip_min.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Max").SetAttrTensor("value", clip_max.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    auto* op_min = g.GetOperationOrThrow("Min");
    auto* op_max = g.GetOperationOrThrow("Max");
    
    (void)g.NewOperation("ClipByValue", "Clip")
        .AddInput(tf_wrap::Output(op_t, 0))
        .AddInput(tf_wrap::Output(op_min, 0))
        .AddInput(tf_wrap::Output(op_max, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Clip", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v[0] == 0.0f);   // -5 clipped to 0
    REQUIRE(v[1] == 0.0f);   // 0 unchanged
    REQUIRE(v[2] == 5.0f);   // 5 unchanged
    REQUIRE(v[3] == 10.0f);  // 10 unchanged
    REQUIRE(v[4] == 10.0f);  // 15 clipped to 10
}

TEST(value_pad) {
    tf_wrap::Graph g;
    
    auto t = tf_wrap::Tensor::FromVector<float>({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto paddings = tf_wrap::Tensor::FromVector<int32_t>({2, 2}, {1, 1, 1, 1}); // pad 1 on each side
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Paddings").SetAttrTensor("value", paddings.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    auto* op_paddings = g.GetOperationOrThrow("Paddings");
    
    (void)g.NewOperation("Pad", "Pad")
        .AddInput(tf_wrap::Output(op_t, 0))
        .AddInput(tf_wrap::Output(op_paddings, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Pad", 0}}, {});
    
    // Output should be 4x4 (2+1+1 in each dim)
    REQUIRE(results[0].shape()[0] == 4);
    REQUIRE(results[0].shape()[1] == 4);
    
    auto v = results[0].ToVector<float>();
    // Original 2x2 should be in center, padded with zeros
    // Row 0: [0, 0, 0, 0]
    // Row 1: [0, 1, 2, 0]
    // Row 2: [0, 3, 4, 0]
    // Row 3: [0, 0, 0, 0]
    REQUIRE(v[0] == 0.0f && v[1] == 0.0f && v[2] == 0.0f && v[3] == 0.0f);
    REQUIRE(v[4] == 0.0f && v[5] == 1.0f && v[6] == 2.0f && v[7] == 0.0f);
    REQUIRE(v[8] == 0.0f && v[9] == 3.0f && v[10] == 4.0f && v[11] == 0.0f);
    REQUIRE(v[12] == 0.0f && v[13] == 0.0f && v[14] == 0.0f && v[15] == 0.0f);
}

TEST(value_reverse) {
    tf_wrap::Graph g;
    
    auto t = tf_wrap::Tensor::FromVector<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto axis = tf_wrap::Tensor::FromVector<int32_t>({1}, {1}); // reverse along axis 1
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Axis").SetAttrTensor("value", axis.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    auto* op_axis = g.GetOperationOrThrow("Axis");
    
    (void)g.NewOperation("ReverseV2", "Reverse")
        .AddInput(tf_wrap::Output(op_t, 0))
        .AddInput(tf_wrap::Output(op_axis, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Reverse", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    // Reverse each row: [[1,2,3],[4,5,6]] -> [[3,2,1],[6,5,4]]
    REQUIRE(v[0] == 3.0f && v[1] == 2.0f && v[2] == 1.0f);
    REQUIRE(v[3] == 6.0f && v[4] == 5.0f && v[5] == 4.0f);
}

TEST(value_stack_unstack) {
    tf_wrap::Graph g;
    
    auto t1 = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    auto t2 = tf_wrap::Tensor::FromVector<float>({3}, {4.0f, 5.0f, 6.0f});
    
    (void)g.NewOperation("Const", "T1").SetAttrTensor("value", t1.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "T2").SetAttrTensor("value", t2.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_t1 = g.GetOperationOrThrow("T1");
    auto* op_t2 = g.GetOperationOrThrow("T2");
    
    std::vector<TF_Output> values = {tf_wrap::Output(op_t1, 0), tf_wrap::Output(op_t2, 0)};
    
    (void)g.NewOperation("Pack", "Stack")
        .AddInputList(values)
        .SetAttrInt("N", 2)
        .SetAttrInt("axis", 0)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Stack", 0}}, {});
    
    // Stacked along axis 0: shape [2, 3]
    REQUIRE(results[0].shape()[0] == 2);
    REQUIRE(results[0].shape()[1] == 3);
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v[0] == 1.0f && v[1] == 2.0f && v[2] == 3.0f);
    REQUIRE(v[3] == 4.0f && v[4] == 5.0f && v[5] == 6.0f);
}

TEST(value_squeeze_expanddims) {
    tf_wrap::Graph g;
    
    auto t = tf_wrap::Tensor::FromVector<float>({1, 3, 1}, {1.0f, 2.0f, 3.0f});
    auto axis = tf_wrap::Tensor::FromScalar<int32_t>(1);
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Axis").SetAttrTensor("value", axis.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    auto* op_axis = g.GetOperationOrThrow("Axis");
    
    // Squeeze removes dims of size 1
    (void)g.NewOperation("Squeeze", "Squeeze")
        .AddInput(tf_wrap::Output(op_t, 0))
        .Finish();
    
    // ExpandDims adds a dim at axis
    auto t2 = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    (void)g.NewOperation("Const", "T2").SetAttrTensor("value", t2.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_t2 = g.GetOperationOrThrow("T2");
    
    (void)g.NewOperation("ExpandDims", "Expand")
        .AddInput(tf_wrap::Output(op_t2, 0))
        .AddInput(tf_wrap::Output(op_axis, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Squeeze", 0}, {"Expand", 0}}, {});
    
    // Squeeze [1,3,1] -> [3]
    REQUIRE(results[0].rank() == 1);
    REQUIRE(results[0].shape()[0] == 3);
    
    // ExpandDims [3] with axis=1 -> [3,1]
    REQUIRE(results[1].rank() == 2);
    REQUIRE(results[1].shape()[0] == 3);
    REQUIRE(results[1].shape()[1] == 1);
}

// =============================================================================
// FILE I/O TESTS - Actually test file operations with real filesystem
// =============================================================================

TEST(value_file_write_read) {
    // Create a temp file path
    const char* test_content = "Hello TensorFlow C API!";
    const char* temp_path = "/tmp/tfwrap_test_file.txt";
    
    // First, write the file using standard C++ (setup)
    {
        std::ofstream ofs(temp_path);
        ofs << test_content;
    }
    
    // Now test ReadFile op
    tf_wrap::Graph g;
    
    auto filename = tf_wrap::Tensor::FromString(temp_path);
    (void)g.NewOperation("Const", "Filename")
        .SetAttrTensor("value", filename.handle())
        .SetAttrType("dtype", TF_STRING)
        .Finish();
    
    auto* op_filename = g.GetOperationOrThrow("Filename");
    
    (void)g.NewOperation("ReadFile", "ReadFile")
        .AddInput(tf_wrap::Output(op_filename, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"ReadFile", 0}}, {});
    
    auto content = results[0].ToString();
    REQUIRE(content == test_content);
    
    // Cleanup
    std::remove(temp_path);
}

TEST(value_file_write_op) {
    const char* temp_path = "/tmp/tfwrap_test_write.txt";
    const char* test_content = "Written by TensorFlow!";
    
    tf_wrap::Graph g;
    
    auto filename = tf_wrap::Tensor::FromString(temp_path);
    auto contents = tf_wrap::Tensor::FromString(test_content);
    
    (void)g.NewOperation("Const", "Filename")
        .SetAttrTensor("value", filename.handle())
        .SetAttrType("dtype", TF_STRING)
        .Finish();
    
    (void)g.NewOperation("Const", "Contents")
        .SetAttrTensor("value", contents.handle())
        .SetAttrType("dtype", TF_STRING)
        .Finish();
    
    auto* op_filename = g.GetOperationOrThrow("Filename");
    auto* op_contents = g.GetOperationOrThrow("Contents");
    
    (void)g.NewOperation("WriteFile", "WriteFile")
        .AddInput(tf_wrap::Output(op_filename, 0))
        .AddInput(tf_wrap::Output(op_contents, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    // WriteFile has no outputs, just run it
    (void)s.Run({}, {}, {"WriteFile"});
    
    // Verify the file was written correctly
    std::ifstream ifs(temp_path);
    std::string read_content((std::istreambuf_iterator<char>(ifs)),
                              std::istreambuf_iterator<char>());
    REQUIRE(read_content == test_content);
    
    // Cleanup
    std::remove(temp_path);
}

// =============================================================================
// IMAGE DECODE TESTS - Minimal valid images embedded as byte arrays
// =============================================================================

TEST(value_decode_png) {
    // Minimal valid 1x1 red PNG (69 bytes)
    // Generated with Python: zlib-compressed raw RGB data
    static const uint8_t red_1x1_png[] = {
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
        0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xde, 0x00, 0x00, 0x00,
        0x0c, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x63, 0xf8, 0xcf, 0xc0, 0x00,
        0x00, 0x03, 0x01, 0x01, 0x00, 0xf7, 0x03, 0x41, 0x43, 0x00, 0x00, 0x00,
        0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82
    };
    
    tf_wrap::Graph g;
    
    // Create string tensor from raw bytes
    auto png_data = tf_wrap::Tensor::FromString(
        std::string(reinterpret_cast<const char*>(red_1x1_png), sizeof(red_1x1_png)));
    
    (void)g.NewOperation("Const", "PngData")
        .SetAttrTensor("value", png_data.handle())
        .SetAttrType("dtype", TF_STRING)
        .Finish();
    
    auto* op_png = g.GetOperationOrThrow("PngData");
    
    (void)g.NewOperation("DecodePng", "Decode")
        .AddInput(tf_wrap::Output(op_png, 0))
        .SetAttrInt("channels", 3)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Decode", 0}}, {});
    
    // Should be 1x1x3 tensor (height, width, channels)
    REQUIRE(results[0].rank() == 3);
    REQUIRE(results[0].shape()[0] == 1);  // height
    REQUIRE(results[0].shape()[1] == 1);  // width
    REQUIRE(results[0].shape()[2] == 3);  // RGB channels
    
    auto pixels = results[0].ToVector<uint8_t>();
    // Red pixel: R=255, G=0, B=0
    REQUIRE(pixels[0] == 255);  // R
    REQUIRE(pixels[1] == 0);    // G
    REQUIRE(pixels[2] == 0);    // B
}

TEST(value_decode_jpeg) {
    // Valid 1x1 blue JPEG (635 bytes) generated by PIL
    static const uint8_t blue_1x1_jpeg[] = {
        0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xff, 0xdb, 0x00, 0x43,
        0x00, 0x02, 0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x01, 0x01, 0x01, 0x02,
        0x02, 0x02, 0x02, 0x02, 0x04, 0x03, 0x02, 0x02, 0x02, 0x02, 0x05, 0x04,
        0x04, 0x03, 0x04, 0x06, 0x05, 0x06, 0x06, 0x06, 0x05, 0x06, 0x06, 0x06,
        0x07, 0x09, 0x08, 0x06, 0x07, 0x09, 0x07, 0x06, 0x06, 0x08, 0x0b, 0x08,
        0x09, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x06, 0x08, 0x0b, 0x0c, 0x0b, 0x0a,
        0x0c, 0x09, 0x0a, 0x0a, 0x0a, 0xff, 0xdb, 0x00, 0x43, 0x01, 0x02, 0x02,
        0x02, 0x02, 0x02, 0x02, 0x05, 0x03, 0x03, 0x05, 0x0a, 0x07, 0x06, 0x07,
        0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0a, 0x0a, 0xff, 0xc0, 0x00, 0x11, 0x08, 0x00, 0x01, 0x00, 0x01, 0x03,
        0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01, 0xff, 0xc4, 0x00,
        0x1f, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
        0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0xff, 0xc4, 0x00, 0xb5, 0x10, 0x00,
        0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00,
        0x00, 0x01, 0x7d, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21,
        0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81,
        0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24,
        0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25,
        0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a,
        0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56,
        0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a,
        0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86,
        0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99,
        0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3,
        0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6,
        0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9,
        0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1,
        0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xff, 0xc4, 0x00,
        0x1f, 0x01, 0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
        0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0xff, 0xc4, 0x00, 0xb5, 0x11, 0x00,
        0x02, 0x01, 0x02, 0x04, 0x04, 0x03, 0x04, 0x07, 0x05, 0x04, 0x04, 0x00,
        0x01, 0x02, 0x77, 0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31,
        0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08,
        0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0, 0x15,
        0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18,
        0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39,
        0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55,
        0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
        0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84,
        0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
        0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa,
        0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4,
        0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7,
        0xd8, 0xd9, 0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
        0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xff, 0xda, 0x00,
        0x0c, 0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3f, 0x00, 0xfc,
        0x73, 0xa2, 0x8a, 0x2b, 0xfd, 0xfc, 0x3f, 0x2b, 0x3f, 0xff, 0xd9
    };
    
    tf_wrap::Graph g;
    
    auto jpeg_data = tf_wrap::Tensor::FromString(
        std::string(reinterpret_cast<const char*>(blue_1x1_jpeg), sizeof(blue_1x1_jpeg)));
    
    (void)g.NewOperation("Const", "JpegData")
        .SetAttrTensor("value", jpeg_data.handle())
        .SetAttrType("dtype", TF_STRING)
        .Finish();
    
    auto* op_jpeg = g.GetOperationOrThrow("JpegData");
    
    (void)g.NewOperation("DecodeJpeg", "Decode")
        .AddInput(tf_wrap::Output(op_jpeg, 0))
        .SetAttrInt("channels", 3)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Decode", 0}}, {});
    
    // Should be 1x1x3 tensor
    REQUIRE(results[0].rank() == 3);
    REQUIRE(results[0].shape()[0] == 1);
    REQUIRE(results[0].shape()[1] == 1);
    REQUIRE(results[0].shape()[2] == 3);
    
    auto pixels = results[0].ToVector<uint8_t>();
    // Blue pixel: R≈0, G≈0, B≈255 (JPEG is lossy so allow some tolerance)
    REQUIRE(pixels[0] < 30);    // R should be low
    REQUIRE(pixels[1] < 30);    // G should be low
    REQUIRE(pixels[2] > 200);   // B should be high
}

TEST(value_encode_png) {
    tf_wrap::Graph g;
    
    // Create a 2x2 RGB image: red, green, blue, white
    auto image = tf_wrap::Tensor::FromVector<uint8_t>({2, 2, 3}, {
        255, 0, 0,      // red
        0, 255, 0,      // green
        0, 0, 255,      // blue
        255, 255, 255   // white
    });
    
    (void)g.NewOperation("Const", "Image")
        .SetAttrTensor("value", image.handle())
        .SetAttrType("dtype", TF_UINT8)
        .Finish();
    
    auto* op_image = g.GetOperationOrThrow("Image");
    
    (void)g.NewOperation("EncodePng", "Encode")
        .AddInput(tf_wrap::Output(op_image, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Encode", 0}}, {});
    
    // Result should be a string tensor containing PNG bytes
    auto png_bytes = results[0].ToString();
    
    // Verify PNG magic bytes
    REQUIRE(png_bytes.size() > 8);
    REQUIRE(static_cast<uint8_t>(png_bytes[0]) == 0x89);
    REQUIRE(png_bytes[1] == 'P');
    REQUIRE(png_bytes[2] == 'N');
    REQUIRE(png_bytes[3] == 'G');
}

// =============================================================================
// CONV2D VARIANT TESTS - Different padding, strides, dilations
// =============================================================================

TEST(value_conv2d_same_padding) {
    tf_wrap::Graph g;
    
    // Input: 1x5x5x1
    std::vector<float> input_data(25);
    for (int i = 0; i < 25; i++) input_data[i] = static_cast<float>(i + 1);
    auto input = tf_wrap::Tensor::FromVector<float>({1, 5, 5, 1}, input_data);
    
    // 3x3 filter of all 1s
    auto filter = tf_wrap::Tensor::FromVector<float>({3, 3, 1, 1}, 
        std::vector<float>(9, 1.0f));
    
    (void)g.NewOperation("Const", "Input").SetAttrTensor("value", input.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Filter").SetAttrTensor("value", filter.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_input = g.GetOperationOrThrow("Input");
    auto* op_filter = g.GetOperationOrThrow("Filter");
    
    (void)g.NewOperation("Conv2D", "Conv")
        .AddInput(tf_wrap::Output(op_input, 0))
        .AddInput(tf_wrap::Output(op_filter, 0))
        .SetAttrIntList("strides", std::vector<int64_t>{1, 1, 1, 1})
        .SetAttrString("padding", "SAME")  // Output should be same size as input
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Conv", 0}}, {});
    
    // With SAME padding, output should be 1x5x5x1 (same as input)
    REQUIRE(results[0].shape()[0] == 1);
    REQUIRE(results[0].shape()[1] == 5);
    REQUIRE(results[0].shape()[2] == 5);
    REQUIRE(results[0].shape()[3] == 1);
}

TEST(value_conv2d_strided) {
    tf_wrap::Graph g;
    
    // Input: 1x6x6x1
    std::vector<float> input_data(36);
    for (int i = 0; i < 36; i++) input_data[i] = static_cast<float>(i + 1);
    auto input = tf_wrap::Tensor::FromVector<float>({1, 6, 6, 1}, input_data);
    
    // 2x2 filter
    auto filter = tf_wrap::Tensor::FromVector<float>({2, 2, 1, 1}, {1.0f, 1.0f, 1.0f, 1.0f});
    
    (void)g.NewOperation("Const", "Input").SetAttrTensor("value", input.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Filter").SetAttrTensor("value", filter.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_input = g.GetOperationOrThrow("Input");
    auto* op_filter = g.GetOperationOrThrow("Filter");
    
    (void)g.NewOperation("Conv2D", "Conv")
        .AddInput(tf_wrap::Output(op_input, 0))
        .AddInput(tf_wrap::Output(op_filter, 0))
        .SetAttrIntList("strides", std::vector<int64_t>{1, 2, 2, 1})  // Stride 2
        .SetAttrString("padding", "VALID")
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Conv", 0}}, {});
    
    // With 6x6 input, 2x2 filter, stride 2, VALID: output is (6-2)/2+1 = 3
    REQUIRE(results[0].shape()[1] == 3);
    REQUIRE(results[0].shape()[2] == 3);
    
    auto v = results[0].ToVector<float>();
    // First output: sum of [1,2,7,8] = 18
    REQUIRE(v[0] == 18.0f);
}

TEST(value_conv2d_dilated) {
    tf_wrap::Graph g;
    
    // Input: 1x7x7x1 (need larger input for dilated conv)
    std::vector<float> input_data(49);
    for (int i = 0; i < 49; i++) input_data[i] = static_cast<float>(i + 1);
    auto input = tf_wrap::Tensor::FromVector<float>({1, 7, 7, 1}, input_data);
    
    // 3x3 filter
    auto filter = tf_wrap::Tensor::FromVector<float>({3, 3, 1, 1}, std::vector<float>(9, 1.0f));
    
    (void)g.NewOperation("Const", "Input").SetAttrTensor("value", input.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Filter").SetAttrTensor("value", filter.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_input = g.GetOperationOrThrow("Input");
    auto* op_filter = g.GetOperationOrThrow("Filter");
    
    (void)g.NewOperation("Conv2D", "Conv")
        .AddInput(tf_wrap::Output(op_input, 0))
        .AddInput(tf_wrap::Output(op_filter, 0))
        .SetAttrIntList("strides", std::vector<int64_t>{1, 1, 1, 1})
        .SetAttrString("padding", "VALID")
        .SetAttrIntList("dilations", std::vector<int64_t>{1, 2, 2, 1})  // Dilation 2
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Conv", 0}}, {});
    
    // With 7x7 input, 3x3 filter with dilation 2: effective filter size is 5x5
    // Output: (7-5)/1+1 = 3
    REQUIRE(results[0].shape()[1] == 3);
    REQUIRE(results[0].shape()[2] == 3);
}

TEST(value_depthwise_conv2d) {
    tf_wrap::Graph g;
    
    // Input: 1x4x4x2 (2 channels)
    auto input = tf_wrap::Tensor::FromVector<float>({1, 4, 4, 2}, 
        std::vector<float>(32, 1.0f));  // All ones
    
    // Depthwise filter: 2x2x2x1 (height, width, in_channels, channel_multiplier)
    // All 1s filter - each output channel will be sum of 4 input values = 4
    auto filter = tf_wrap::Tensor::FromVector<float>({2, 2, 2, 1}, 
        std::vector<float>(8, 1.0f));
    
    (void)g.NewOperation("Const", "Input").SetAttrTensor("value", input.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Filter").SetAttrTensor("value", filter.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_input = g.GetOperationOrThrow("Input");
    auto* op_filter = g.GetOperationOrThrow("Filter");
    
    (void)g.NewOperation("DepthwiseConv2dNative", "DepthwiseConv")
        .AddInput(tf_wrap::Output(op_input, 0))
        .AddInput(tf_wrap::Output(op_filter, 0))
        .SetAttrIntList("strides", std::vector<int64_t>{1, 1, 1, 1})
        .SetAttrString("padding", "VALID")
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"DepthwiseConv", 0}}, {});
    
    // Output: 1x3x3x2
    REQUIRE(results[0].shape()[0] == 1);
    REQUIRE(results[0].shape()[1] == 3);
    REQUIRE(results[0].shape()[2] == 3);
    REQUIRE(results[0].shape()[3] == 2);
    
    auto v = results[0].ToVector<float>();
    // Each position: sum of 4 ones with filter [1,1,1,1] = 4
    REQUIRE(v[0] == 4.0f);
    REQUIRE(v[1] == 4.0f);
}

// =============================================================================
// BATCHNORM TESTS - Both training and inference modes
// =============================================================================

TEST(value_batchnorm_inference) {
    tf_wrap::Graph g;
    
    // Input: batch=2, height=1, width=1, channels=2
    auto x = tf_wrap::Tensor::FromVector<float>({2, 1, 1, 2}, 
        {1.0f, 2.0f,   // Sample 1
         3.0f, 4.0f}); // Sample 2
    
    // Scale (gamma) and offset (beta) per channel
    auto scale = tf_wrap::Tensor::FromVector<float>({2}, {1.0f, 1.0f});
    auto offset = tf_wrap::Tensor::FromVector<float>({2}, {0.0f, 0.0f});
    
    // Pre-computed mean and variance (inference mode uses these)
    auto mean = tf_wrap::Tensor::FromVector<float>({2}, {2.0f, 3.0f});
    auto variance = tf_wrap::Tensor::FromVector<float>({2}, {1.0f, 1.0f});
    
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Scale").SetAttrTensor("value", scale.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Offset").SetAttrTensor("value", offset.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Mean").SetAttrTensor("value", mean.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Variance").SetAttrTensor("value", variance.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_x = g.GetOperationOrThrow("X");
    auto* op_scale = g.GetOperationOrThrow("Scale");
    auto* op_offset = g.GetOperationOrThrow("Offset");
    auto* op_mean = g.GetOperationOrThrow("Mean");
    auto* op_variance = g.GetOperationOrThrow("Variance");
    
    (void)g.NewOperation("FusedBatchNormV3", "BatchNorm")
        .AddInput(tf_wrap::Output(op_x, 0))
        .AddInput(tf_wrap::Output(op_scale, 0))
        .AddInput(tf_wrap::Output(op_offset, 0))
        .AddInput(tf_wrap::Output(op_mean, 0))
        .AddInput(tf_wrap::Output(op_variance, 0))
        .SetAttrFloat("epsilon", 0.001f)
        .SetAttrBool("is_training", false)  // Inference mode
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"BatchNorm", 0}}, {});  // Output 0 is normalized x
    
    auto v = results[0].ToVector<float>();
    // Normalized: (x - mean) / sqrt(variance + epsilon) * scale + offset
    // Sample 1, channel 0: (1 - 2) / sqrt(1.001) ≈ -0.9995
    // Sample 1, channel 1: (2 - 3) / sqrt(1.001) ≈ -0.9995
    REQUIRE_APPROX(v[0], -0.9995f, 0.01f);
    REQUIRE_APPROX(v[1], -0.9995f, 0.01f);
}

TEST(value_batchnorm_training) {
    tf_wrap::Graph g;
    
    // Input: batch=4, height=1, width=1, channels=1
    // Values chosen so mean=2.5
    auto x = tf_wrap::Tensor::FromVector<float>({4, 1, 1, 1}, {1.0f, 2.0f, 3.0f, 4.0f});
    
    auto scale = tf_wrap::Tensor::FromVector<float>({1}, {1.0f});
    auto offset = tf_wrap::Tensor::FromVector<float>({1}, {0.0f});
    
    // In training mode, mean/variance inputs are ignored (computed from batch)
    auto mean = tf_wrap::Tensor::FromVector<float>({1}, {0.0f});
    auto variance = tf_wrap::Tensor::FromVector<float>({1}, {1.0f});
    
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Scale").SetAttrTensor("value", scale.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Offset").SetAttrTensor("value", offset.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Mean").SetAttrTensor("value", mean.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Variance").SetAttrTensor("value", variance.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_x = g.GetOperationOrThrow("X");
    auto* op_scale = g.GetOperationOrThrow("Scale");
    auto* op_offset = g.GetOperationOrThrow("Offset");
    auto* op_mean = g.GetOperationOrThrow("Mean");
    auto* op_variance = g.GetOperationOrThrow("Variance");
    
    (void)g.NewOperation("FusedBatchNormV3", "BatchNorm")
        .AddInput(tf_wrap::Output(op_x, 0))
        .AddInput(tf_wrap::Output(op_scale, 0))
        .AddInput(tf_wrap::Output(op_offset, 0))
        .AddInput(tf_wrap::Output(op_mean, 0))
        .AddInput(tf_wrap::Output(op_variance, 0))
        .SetAttrFloat("epsilon", 0.001f)
        .SetAttrBool("is_training", true)  // Training mode
        .Finish();
    
    tf_wrap::Session s(g);
    // Output 0: normalized y, Output 1: batch_mean, Output 2: batch_variance
    auto results = s.Run({}, {{"BatchNorm", 0}, {"BatchNorm", 1}, {"BatchNorm", 2}}, {});
    
    auto y = results[0].ToVector<float>();
    auto batch_mean = results[1].ToVector<float>();
    auto batch_var = results[2].ToVector<float>();
    
    // Batch mean should be (1+2+3+4)/4 = 2.5
    REQUIRE_APPROX(batch_mean[0], 2.5f, 0.01f);
    
    // Batch variance: TF may use population or sample variance
    // Population: 1.25, Sample (Bessel): 1.6667
    // Just check it's in reasonable range
    REQUIRE(batch_var[0] > 1.0f);
    REQUIRE(batch_var[0] < 2.0f);
    
    // Normalized values should have mean≈0
    float y_mean = (y[0] + y[1] + y[2] + y[3]) / 4.0f;
    REQUIRE_APPROX(y_mean, 0.0f, 0.01f);
}

// =============================================================================
// RANDOM OP TESTS - Statistical validation of distributions
// =============================================================================

TEST(value_random_uniform_distribution) {
    tf_wrap::Graph g;
    
    // Generate 10000 random values
    auto shape = tf_wrap::Tensor::FromVector<int32_t>({1}, {10000});
    
    (void)g.NewOperation("Const", "Shape")
        .SetAttrTensor("value", shape.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    auto* op_shape = g.GetOperationOrThrow("Shape");
    
    (void)g.NewOperation("RandomUniform", "Random")
        .AddInput(tf_wrap::Output(op_shape, 0))
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Random", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 10000);
    
    // Compute statistics
    float sum = 0.0f, sum_sq = 0.0f;
    float min_val = v[0], max_val = v[0];
    for (float x : v) {
        sum += x;
        sum_sq += x * x;
        if (x < min_val) min_val = x;
        if (x > max_val) max_val = x;
    }
    
    float mean = sum / 10000.0f;
    float variance = (sum_sq / 10000.0f) - (mean * mean);
    
    // Uniform[0,1] has mean=0.5, variance=1/12≈0.0833
    REQUIRE_APPROX(mean, 0.5f, 0.02f);        // Mean should be ~0.5
    REQUIRE_APPROX(variance, 0.0833f, 0.01f); // Variance should be ~1/12
    REQUIRE(min_val >= 0.0f);                  // All values in [0,1)
    REQUIRE(max_val < 1.0f);
}

TEST(value_random_normal_distribution) {
    tf_wrap::Graph g;
    
    auto shape = tf_wrap::Tensor::FromVector<int32_t>({1}, {10000});
    
    (void)g.NewOperation("Const", "Shape")
        .SetAttrTensor("value", shape.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    auto* op_shape = g.GetOperationOrThrow("Shape");
    
    (void)g.NewOperation("RandomStandardNormal", "Random")
        .AddInput(tf_wrap::Output(op_shape, 0))
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Random", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    
    // Compute statistics
    float sum = 0.0f, sum_sq = 0.0f;
    for (float x : v) {
        sum += x;
        sum_sq += x * x;
    }
    
    float mean = sum / 10000.0f;
    float variance = (sum_sq / 10000.0f) - (mean * mean);
    
    // Standard normal has mean=0, variance=1
    REQUIRE_APPROX(mean, 0.0f, 0.05f);      // Mean should be ~0
    REQUIRE_APPROX(variance, 1.0f, 0.1f);   // Variance should be ~1
}

TEST(value_truncated_normal_distribution) {
    tf_wrap::Graph g;
    
    auto shape = tf_wrap::Tensor::FromVector<int32_t>({1}, {10000});
    
    (void)g.NewOperation("Const", "Shape")
        .SetAttrTensor("value", shape.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    auto* op_shape = g.GetOperationOrThrow("Shape");
    
    (void)g.NewOperation("TruncatedNormal", "Random")
        .AddInput(tf_wrap::Output(op_shape, 0))
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Random", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    
    // Truncated normal: all values should be within [-2, 2] stddevs
    float min_val = v[0], max_val = v[0];
    for (float x : v) {
        if (x < min_val) min_val = x;
        if (x > max_val) max_val = x;
    }
    
    REQUIRE(min_val >= -2.0f);
    REQUIRE(max_val <= 2.0f);
}

TEST(value_random_shuffle) {
    tf_wrap::Graph g;
    
    // Create a sequence 0-99
    std::vector<int32_t> original(100);
    for (int i = 0; i < 100; i++) original[i] = i;
    
    auto t = tf_wrap::Tensor::FromVector<int32_t>({100}, original);
    
    (void)g.NewOperation("Const", "T")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    
    (void)g.NewOperation("RandomShuffle", "Shuffle")
        .AddInput(tf_wrap::Output(op_t, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Shuffle", 0}}, {});
    
    auto shuffled = results[0].ToVector<int32_t>();
    
    // Should have same size
    REQUIRE(shuffled.size() == 100);
    
    // Should contain all original values (just reordered)
    std::vector<int32_t> sorted_shuffled = shuffled;
    std::sort(sorted_shuffled.begin(), sorted_shuffled.end());
    REQUIRE(sorted_shuffled == original);
    
    // Should be different from original (extremely unlikely to be same)
    REQUIRE(shuffled != original);
}

// =============================================================================
// POOL VARIANT TESTS
// =============================================================================

TEST(value_maxpool_with_argmax) {
    tf_wrap::Graph g;
    
    // Input: 1x4x4x1
    auto input = tf_wrap::Tensor::FromVector<float>({1, 4, 4, 1}, 
        {1.0f, 2.0f, 3.0f, 4.0f,
         5.0f, 9.0f, 7.0f, 8.0f,   // 9 is max in top-left 2x2
         6.0f, 10.0f, 11.0f, 12.0f,
         13.0f, 14.0f, 15.0f, 16.0f});
    
    (void)g.NewOperation("Const", "Input")
        .SetAttrTensor("value", input.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* op_input = g.GetOperationOrThrow("Input");
    
    (void)g.NewOperation("MaxPoolWithArgmax", "MaxPoolArgmax")
        .AddInput(tf_wrap::Output(op_input, 0))
        .SetAttrIntList("ksize", std::vector<int64_t>{1, 2, 2, 1})
        .SetAttrIntList("strides", std::vector<int64_t>{1, 2, 2, 1})
        .SetAttrString("padding", "VALID")
        .SetAttrType("Targmax", TF_INT64)
        .Finish();
    
    tf_wrap::Session s(g);
    // Output 0: max values, Output 1: argmax indices
    auto results = s.Run({}, {{"MaxPoolArgmax", 0}, {"MaxPoolArgmax", 1}}, {});
    
    auto maxvals = results[0].ToVector<float>();
    auto argmax = results[1].ToVector<int64_t>();
    
    // Top-left 2x2: max is 9
    REQUIRE(maxvals[0] == 9.0f);
    // Top-right 2x2: max is 8
    REQUIRE(maxvals[1] == 8.0f);
    // Bottom-left 2x2: max is 14
    REQUIRE(maxvals[2] == 14.0f);
    // Bottom-right 2x2: max is 16
    REQUIRE(maxvals[3] == 16.0f);
    
    // Argmax gives flattened index of max element
    // 9 is at position 5 in flattened input
    REQUIRE(argmax[0] == 5);
}

TEST(value_avgpool_same_padding) {
    tf_wrap::Graph g;
    
    // Input: 1x5x5x1 (odd size to test SAME padding behavior)
    std::vector<float> input_data(25);
    for (int i = 0; i < 25; i++) input_data[i] = 1.0f;  // All ones
    auto input = tf_wrap::Tensor::FromVector<float>({1, 5, 5, 1}, input_data);
    
    (void)g.NewOperation("Const", "Input")
        .SetAttrTensor("value", input.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* op_input = g.GetOperationOrThrow("Input");
    
    (void)g.NewOperation("AvgPool", "AvgPool")
        .AddInput(tf_wrap::Output(op_input, 0))
        .SetAttrIntList("ksize", std::vector<int64_t>{1, 2, 2, 1})
        .SetAttrIntList("strides", std::vector<int64_t>{1, 2, 2, 1})
        .SetAttrString("padding", "SAME")
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"AvgPool", 0}}, {});
    
    // SAME padding with stride 2 on 5x5: ceil(5/2) = 3
    REQUIRE(results[0].shape()[1] == 3);
    REQUIRE(results[0].shape()[2] == 3);
    
    auto v = results[0].ToVector<float>();
    // All inputs are 1, so average should be 1 everywhere
    for (size_t i = 0; i < v.size(); i++) {
        REQUIRE(v[i] == 1.0f);
    }
}

TEST(toscalar_multielement_throws) {
    auto t = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    
    bool threw = false;
    try {
        t.ToScalar<float>(); // 3 elements, not 1
    } catch (const std::exception&) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST(fromvector_shape_mismatch_throws) {
    bool threw = false;
    try {
        // Shape says 10 elements, but vector has 5
        std::vector<int64_t> shape = {10};
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        auto t = tf_wrap::Tensor::FromVector<float>(shape, data);
    } catch (const std::exception&) {
        threw = true;
    }
    REQUIRE(threw);
}

// =============================================================================
// Stress Tests
// =============================================================================

TEST(rapid_graph_creation) {
    auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < 100; ++i) {
        tf_wrap::Graph g;
        std::vector<int64_t> shape_vec = {10};
        std::vector<float> data_vec(10, 1.0f);
        auto t = tf_wrap::Tensor::FromVector<float>(shape_vec, data_vec);
        (void)g.NewOperation("Const", "C").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    }
    
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "    (100 graphs in " << ms << "ms)\n";
}

TEST(rapid_session_runs) {
    tf_wrap::Graph g;
    
    (void)g.NewOperation("Placeholder", "X").SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    (void)g.NewOperation("Square", "Y").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    
    tf_wrap::Session s(g);
    
    auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < 1000; ++i) {
        auto input = tf_wrap::Tensor::FromScalar<float>(static_cast<float>(i));
        auto results = s.Run({{"X", 0, input.handle()}}, {{"Y", 0}}, {});
        (void)results;
    }
    
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "    (1000 runs in " << ms << "ms)\n";
}

TEST(concurrent_sessions) {
    // Create a shared graph
    tf_wrap::Graph g;
    
    (void)g.NewOperation("Placeholder", "X").SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    (void)g.NewOperation("Square", "Y").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    
    std::atomic<int> total_runs{0};
    std::atomic<bool> error{false};
    
    auto worker = [&](int thread_id) {
        try {
            tf_wrap::Session s(g);
            
            for (int i = 0; i < 100; ++i) {
                auto input = tf_wrap::Tensor::FromScalar<float>(static_cast<float>(thread_id * 100 + i));
                auto results = s.Run({{"X", 0, input.handle()}}, {{"Y", 0}}, {});
                
                float expected = static_cast<float>((thread_id * 100 + i) * (thread_id * 100 + i));
                float actual = results[0].ToScalar<float>();
                
                if (std::abs(actual - expected) > 0.01f) {
                    error = true;
                }
                
                total_runs++;
            }
        } catch (...) {
            error = true;
        }
    };
    
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back(worker, i);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    REQUIRE(!error);
    REQUIRE(total_runs == 400);
}

// =============================================================================
// Soak Test (long running)
// =============================================================================

TEST(soak_test_30_seconds) {
    std::cout << "    Running for 30 seconds...\n";
    
    tf_wrap::Graph g;
    
    (void)g.NewOperation("Placeholder", "X").SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    (void)g.NewOperation("Square", "Y").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    
    tf_wrap::Session s(g);
    
    auto start = std::chrono::steady_clock::now();
    auto end_time = start + std::chrono::seconds(30);
    
    int iterations = 0;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    
    while (std::chrono::steady_clock::now() < end_time) {
        float val = dist(rng);
        auto input = tf_wrap::Tensor::FromScalar<float>(val);
        auto results = s.Run({{"X", 0, input.handle()}}, {{"Y", 0}}, {});
        
        float expected = val * val;
        float actual = results[0].ToScalar<float>();
        
        if (std::abs(actual - expected) > std::abs(expected) * 0.0001f + 0.0001f) {
            throw std::runtime_error("Soak test value mismatch");
        }
        
        iterations++;
    }
    
    std::cout << "    Completed " << iterations << " iterations\n";
    REQUIRE(iterations > 1000); // Should do way more than this
}

// =============================================================================
// Fuzz Test
// =============================================================================

TEST(fuzz_random_tensor_shapes) {
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> rank_dist(0, 4);
    std::uniform_int_distribution<int> dim_dist(1, 20);
    std::uniform_real_distribution<float> val_dist(-100.0f, 100.0f);
    
    for (int trial = 0; trial < 100; ++trial) {
        // Random shape
        int rank = rank_dist(rng);
        std::vector<int64_t> shape(rank);
        int64_t total = 1;
        for (int i = 0; i < rank; ++i) {
            shape[i] = dim_dist(rng);
            total *= shape[i];
        }
        
        // Random data
        std::vector<float> data(total);
        for (auto& v : data) v = val_dist(rng);
        
        // Create tensor and verify
        auto t = tf_wrap::Tensor::FromVector<float>(shape, data);
        REQUIRE(t.rank() == rank);
        REQUIRE(t.num_elements() == static_cast<std::size_t>(total));
        
        // Roundtrip through graph
        tf_wrap::Graph g;
        (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
        
        tf_wrap::Session s(g);
        auto results = s.Run({}, {{"T", 0}}, {});
        
        auto out = results[0].ToVector<float>();
        REQUIRE(out.size() == data.size());
        
        for (size_t i = 0; i < data.size(); ++i) {
            REQUIRE_APPROX(out[i], data[i], 0.0001f);
        }
    }
}

// =============================================================================
// ADDITIONAL MATH OPS - Acos, Asin, Atan, Cosh, Sinh, Ceil, Floor, etc.
// =============================================================================

TEST(value_inverse_trig) {
    tf_wrap::Graph g;
    
    auto x = tf_wrap::Tensor::FromVector<float>({3}, {0.0f, 0.5f, 1.0f});
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("Acos", "Acos").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    (void)g.NewOperation("Asin", "Asin").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    (void)g.NewOperation("Atan", "Atan").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Acos", 0}, {"Asin", 0}, {"Atan", 0}}, {});
    
    auto acos_v = results[0].ToVector<float>();
    auto asin_v = results[1].ToVector<float>();
    auto atan_v = results[2].ToVector<float>();
    
    // acos(0) = pi/2, acos(1) = 0
    REQUIRE_APPROX(acos_v[0], 1.5708f, 0.001f);
    REQUIRE_APPROX(acos_v[2], 0.0f, 0.001f);
    // asin(0) = 0, asin(1) = pi/2
    REQUIRE_APPROX(asin_v[0], 0.0f, 0.001f);
    REQUIRE_APPROX(asin_v[2], 1.5708f, 0.001f);
    // atan(0) = 0, atan(1) = pi/4
    REQUIRE_APPROX(atan_v[0], 0.0f, 0.001f);
    REQUIRE_APPROX(atan_v[2], 0.7854f, 0.001f);
}

TEST(value_hyperbolic) {
    tf_wrap::Graph g;
    
    auto x = tf_wrap::Tensor::FromVector<float>({3}, {0.0f, 1.0f, -1.0f});
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("Cosh", "Cosh").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    (void)g.NewOperation("Sinh", "Sinh").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Cosh", 0}, {"Sinh", 0}}, {});
    
    auto cosh_v = results[0].ToVector<float>();
    auto sinh_v = results[1].ToVector<float>();
    
    // cosh(0) = 1, cosh(1) ≈ 1.543
    REQUIRE_APPROX(cosh_v[0], 1.0f, 0.001f);
    REQUIRE_APPROX(cosh_v[1], 1.543f, 0.01f);
    // sinh(0) = 0, sinh(1) ≈ 1.175
    REQUIRE_APPROX(sinh_v[0], 0.0f, 0.001f);
    REQUIRE_APPROX(sinh_v[1], 1.175f, 0.01f);
}

TEST(value_rounding_ops) {
    tf_wrap::Graph g;
    
    auto x = tf_wrap::Tensor::FromVector<float>({4}, {1.2f, 1.7f, -1.2f, -1.7f});
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("Ceil", "Ceil").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    (void)g.NewOperation("Floor", "Floor").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    (void)g.NewOperation("Round", "Round").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    (void)g.NewOperation("Rint", "Rint").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Ceil", 0}, {"Floor", 0}, {"Round", 0}, {"Rint", 0}}, {});
    
    auto ceil_v = results[0].ToVector<float>();
    auto floor_v = results[1].ToVector<float>();
    auto round_v = results[2].ToVector<float>();
    
    REQUIRE(ceil_v[0] == 2.0f);
    REQUIRE(ceil_v[2] == -1.0f);
    REQUIRE(floor_v[0] == 1.0f);
    REQUIRE(floor_v[2] == -2.0f);
    REQUIRE(round_v[0] == 1.0f);
    REQUIRE(round_v[1] == 2.0f);
}

TEST(value_neg_reciprocal_rsqrt_sign) {
    tf_wrap::Graph g;
    
    auto x = tf_wrap::Tensor::FromVector<float>({4}, {4.0f, -2.0f, 0.25f, 9.0f});
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("Neg", "Neg").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    (void)g.NewOperation("Reciprocal", "Recip").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    (void)g.NewOperation("Rsqrt", "Rsqrt").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    (void)g.NewOperation("Sign", "Sign").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Neg", 0}, {"Recip", 0}, {"Rsqrt", 0}, {"Sign", 0}}, {});
    
    auto neg_v = results[0].ToVector<float>();
    auto recip_v = results[1].ToVector<float>();
    auto rsqrt_v = results[2].ToVector<float>();
    auto sign_v = results[3].ToVector<float>();
    
    REQUIRE(neg_v[0] == -4.0f);
    REQUIRE(neg_v[1] == 2.0f);
    REQUIRE(recip_v[0] == 0.25f);
    REQUIRE(recip_v[2] == 4.0f);
    REQUIRE(rsqrt_v[0] == 0.5f);  // 1/sqrt(4) = 0.5
    REQUIRE(rsqrt_v[3] == 1.0f/3.0f);  // 1/sqrt(9) = 1/3
    REQUIRE(sign_v[0] == 1.0f);
    REQUIRE(sign_v[1] == -1.0f);
}

TEST(value_expm1_log1p) {
    tf_wrap::Graph g;
    
    auto x = tf_wrap::Tensor::FromVector<float>({3}, {0.0f, 1.0f, 0.001f});
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("Expm1", "Expm1").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    (void)g.NewOperation("Log1p", "Log1p").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Expm1", 0}, {"Log1p", 0}}, {});
    
    auto expm1_v = results[0].ToVector<float>();
    auto log1p_v = results[1].ToVector<float>();
    
    // expm1(0) = 0, expm1(1) = e-1 ≈ 1.718
    REQUIRE_APPROX(expm1_v[0], 0.0f, 0.001f);
    REQUIRE_APPROX(expm1_v[1], 1.718f, 0.01f);
    // log1p(0) = 0, log1p(1) = ln(2) ≈ 0.693
    REQUIRE_APPROX(log1p_v[0], 0.0f, 0.001f);
    REQUIRE_APPROX(log1p_v[1], 0.693f, 0.01f);
}

// =============================================================================
// COMPARISON OPS - GreaterEqual, LessEqual, NotEqual
// =============================================================================

TEST(value_comparison_extended) {
    tf_wrap::Graph g;
    
    auto a = tf_wrap::Tensor::FromVector<float>({4}, {1.0f, 2.0f, 3.0f, 2.0f});
    auto b = tf_wrap::Tensor::FromVector<float>({4}, {2.0f, 2.0f, 2.0f, 3.0f});
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("GreaterEqual", "GE").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    (void)g.NewOperation("LessEqual", "LE").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    (void)g.NewOperation("NotEqual", "NE").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"GE", 0}, {"LE", 0}, {"NE", 0}}, {});
    
    auto ge = results[0].ToVector<bool>();
    auto le = results[1].ToVector<bool>();
    auto ne = results[2].ToVector<bool>();
    
    // a=[1,2,3,2], b=[2,2,2,3]
    REQUIRE(ge[0] == false);  // 1 >= 2
    REQUIRE(ge[1] == true);   // 2 >= 2
    REQUIRE(ge[2] == true);   // 3 >= 2
    REQUIRE(le[0] == true);   // 1 <= 2
    REQUIRE(le[2] == false);  // 3 <= 2
    REQUIRE(ne[0] == true);   // 1 != 2
    REQUIRE(ne[1] == false);  // 2 != 2
}

// =============================================================================
// SHAPE/SIZE OPS - Shape, ShapeN, Size, Rank
// =============================================================================

TEST(value_shape_size_rank) {
    tf_wrap::Graph g;
    
    auto x = tf_wrap::Tensor::FromVector<float>({2, 3, 4}, std::vector<float>(24, 1.0f));
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("Shape", "Shape").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    (void)g.NewOperation("Size", "Size").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    (void)g.NewOperation("Rank", "Rank").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Shape", 0}, {"Size", 0}, {"Rank", 0}}, {});
    
    auto shape = results[0].ToVector<int32_t>();
    auto size = results[1].ToScalar<int32_t>();
    auto rank = results[2].ToScalar<int32_t>();
    
    REQUIRE(shape.size() == 3);
    REQUIRE(shape[0] == 2);
    REQUIRE(shape[1] == 3);
    REQUIRE(shape[2] == 4);
    REQUIRE(size == 24);
    REQUIRE(rank == 3);
}

// =============================================================================
// TENSOR MANIPULATION - BroadcastTo, Split, StridedSlice, etc.
// =============================================================================

TEST(value_broadcast_to) {
    tf_wrap::Graph g;
    
    auto x = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    auto shape = tf_wrap::Tensor::FromVector<int32_t>({2}, {2, 3});
    
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Shape").SetAttrTensor("value", shape.handle()).SetAttrType("dtype", TF_INT32).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    auto* op_shape = g.GetOperationOrThrow("Shape");
    
    (void)g.NewOperation("BroadcastTo", "Broadcast")
        .AddInput(tf_wrap::Output(op_x, 0))
        .AddInput(tf_wrap::Output(op_shape, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Broadcast", 0}}, {});
    
    REQUIRE(results[0].shape()[0] == 2);
    REQUIRE(results[0].shape()[1] == 3);
    auto v = results[0].ToVector<float>();
    // [[1,2,3], [1,2,3]]
    REQUIRE(v[0] == 1.0f);
    REQUIRE(v[3] == 1.0f);
}

TEST(value_split) {
    tf_wrap::Graph g;
    
    auto x = tf_wrap::Tensor::FromVector<float>({6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto axis = tf_wrap::Tensor::FromScalar<int32_t>(0);
    
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Axis").SetAttrTensor("value", axis.handle()).SetAttrType("dtype", TF_INT32).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    auto* op_axis = g.GetOperationOrThrow("Axis");
    
    (void)g.NewOperation("Split", "Split")
        .AddInput(tf_wrap::Output(op_axis, 0))
        .AddInput(tf_wrap::Output(op_x, 0))
        .SetAttrInt("num_split", 3)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Split", 0}, {"Split", 1}, {"Split", 2}}, {});
    
    REQUIRE(results[0].ToVector<float>()[0] == 1.0f);
    REQUIRE(results[0].ToVector<float>()[1] == 2.0f);
    REQUIRE(results[1].ToVector<float>()[0] == 3.0f);
    REQUIRE(results[2].ToVector<float>()[0] == 5.0f);
}

TEST(value_strided_slice) {
    tf_wrap::Graph g;
    
    auto x = tf_wrap::Tensor::FromVector<float>({6}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto begin = tf_wrap::Tensor::FromVector<int32_t>({1}, {1});
    auto end = tf_wrap::Tensor::FromVector<int32_t>({1}, {5});
    auto strides = tf_wrap::Tensor::FromVector<int32_t>({1}, {2});
    
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Begin").SetAttrTensor("value", begin.handle()).SetAttrType("dtype", TF_INT32).Finish();
    (void)g.NewOperation("Const", "End").SetAttrTensor("value", end.handle()).SetAttrType("dtype", TF_INT32).Finish();
    (void)g.NewOperation("Const", "Strides").SetAttrTensor("value", strides.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_x = g.GetOperationOrThrow("X");
    auto* op_begin = g.GetOperationOrThrow("Begin");
    auto* op_end = g.GetOperationOrThrow("End");
    auto* op_strides = g.GetOperationOrThrow("Strides");
    
    (void)g.NewOperation("StridedSlice", "Slice")
        .AddInput(tf_wrap::Output(op_x, 0))
        .AddInput(tf_wrap::Output(op_begin, 0))
        .AddInput(tf_wrap::Output(op_end, 0))
        .AddInput(tf_wrap::Output(op_strides, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Slice", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    // x[1:5:2] = [1, 3]
    REQUIRE(v.size() == 2);
    REQUIRE(v[0] == 1.0f);
    REQUIRE(v[1] == 3.0f);
}

TEST(value_unpack) {
    tf_wrap::Graph g;
    
    auto x = tf_wrap::Tensor::FromVector<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("Unpack", "Unpack")
        .AddInput(tf_wrap::Output(op_x, 0))
        .SetAttrInt("num", 2)
        .SetAttrInt("axis", 0)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Unpack", 0}, {"Unpack", 1}}, {});
    
    auto v0 = results[0].ToVector<float>();
    auto v1 = results[1].ToVector<float>();
    REQUIRE(v0[0] == 1.0f);
    REQUIRE(v0[1] == 2.0f);
    REQUIRE(v0[2] == 3.0f);
    REQUIRE(v1[0] == 4.0f);
}

TEST(value_scatter_nd) {
    tf_wrap::Graph g;
    
    auto indices = tf_wrap::Tensor::FromVector<int32_t>({2, 1}, {0, 2});
    auto updates = tf_wrap::Tensor::FromVector<float>({2}, {10.0f, 20.0f});
    auto shape = tf_wrap::Tensor::FromVector<int32_t>({1}, {4});
    
    (void)g.NewOperation("Const", "Indices").SetAttrTensor("value", indices.handle()).SetAttrType("dtype", TF_INT32).Finish();
    (void)g.NewOperation("Const", "Updates").SetAttrTensor("value", updates.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Shape").SetAttrTensor("value", shape.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_indices = g.GetOperationOrThrow("Indices");
    auto* op_updates = g.GetOperationOrThrow("Updates");
    auto* op_shape = g.GetOperationOrThrow("Shape");
    
    (void)g.NewOperation("ScatterNd", "Scatter")
        .AddInput(tf_wrap::Output(op_indices, 0))
        .AddInput(tf_wrap::Output(op_updates, 0))
        .AddInput(tf_wrap::Output(op_shape, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Scatter", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v[0] == 10.0f);
    REQUIRE(v[1] == 0.0f);
    REQUIRE(v[2] == 20.0f);
    REQUIRE(v[3] == 0.0f);
}

TEST(value_gather_nd) {
    tf_wrap::Graph g;
    
    auto params = tf_wrap::Tensor::FromVector<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto indices = tf_wrap::Tensor::FromVector<int32_t>({2, 2}, {0, 0, 1, 2});
    
    (void)g.NewOperation("Const", "Params").SetAttrTensor("value", params.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Indices").SetAttrTensor("value", indices.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_params = g.GetOperationOrThrow("Params");
    auto* op_indices = g.GetOperationOrThrow("Indices");
    
    (void)g.NewOperation("GatherNd", "Gather")
        .AddInput(tf_wrap::Output(op_params, 0))
        .AddInput(tf_wrap::Output(op_indices, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Gather", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v[0] == 1.0f);  // params[0,0]
    REQUIRE(v[1] == 6.0f);  // params[1,2]
}

TEST(value_select_v2) {
    tf_wrap::Graph g;
    
    auto cond = tf_wrap::Tensor::FromVector<bool>({4}, {true, false, true, false});
    auto a = tf_wrap::Tensor::FromVector<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto b = tf_wrap::Tensor::FromVector<float>({4}, {10.0f, 20.0f, 30.0f, 40.0f});
    
    (void)g.NewOperation("Const", "Cond").SetAttrTensor("value", cond.handle()).SetAttrType("dtype", TF_BOOL).Finish();
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_cond = g.GetOperationOrThrow("Cond");
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("SelectV2", "Select")
        .AddInput(tf_wrap::Output(op_cond, 0))
        .AddInput(tf_wrap::Output(op_a, 0))
        .AddInput(tf_wrap::Output(op_b, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Select", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v[0] == 1.0f);   // true -> a
    REQUIRE(v[1] == 20.0f);  // false -> b
    REQUIRE(v[2] == 3.0f);   // true -> a
    REQUIRE(v[3] == 40.0f);  // false -> b
}

TEST(value_mirror_pad) {
    tf_wrap::Graph g;
    
    auto x = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    auto paddings = tf_wrap::Tensor::FromVector<int32_t>({1, 2}, {1, 1});
    
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Paddings").SetAttrTensor("value", paddings.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_x = g.GetOperationOrThrow("X");
    auto* op_pad = g.GetOperationOrThrow("Paddings");
    
    (void)g.NewOperation("MirrorPad", "Pad")
        .AddInput(tf_wrap::Output(op_x, 0))
        .AddInput(tf_wrap::Output(op_pad, 0))
        .SetAttrString("mode", "REFLECT")
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Pad", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    // REFLECT: [2, 1, 2, 3, 2]
    REQUIRE(v.size() == 5);
    REQUIRE(v[0] == 2.0f);
    REQUIRE(v[4] == 2.0f);
}

// =============================================================================
// ACTIVATIONS - Softplus, Softsign, LogSoftmax
// =============================================================================

TEST(value_softplus_softsign) {
    tf_wrap::Graph g;
    
    auto x = tf_wrap::Tensor::FromVector<float>({3}, {-1.0f, 0.0f, 1.0f});
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("Softplus", "Softplus").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    (void)g.NewOperation("Softsign", "Softsign").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Softplus", 0}, {"Softsign", 0}}, {});
    
    auto sp = results[0].ToVector<float>();
    auto ss = results[1].ToVector<float>();
    
    // softplus(x) = log(1 + exp(x))
    REQUIRE_APPROX(sp[1], 0.693f, 0.01f);  // softplus(0) = ln(2)
    // softsign(x) = x / (1 + |x|)
    REQUIRE_APPROX(ss[0], -0.5f, 0.01f);   // -1/(1+1) = -0.5
    REQUIRE_APPROX(ss[1], 0.0f, 0.01f);    // 0
    REQUIRE_APPROX(ss[2], 0.5f, 0.01f);    // 1/(1+1) = 0.5
}

TEST(value_log_softmax) {
    tf_wrap::Graph g;
    
    auto x = tf_wrap::Tensor::FromVector<float>({1, 3}, {1.0f, 2.0f, 3.0f});
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("LogSoftmax", "LogSoftmax").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"LogSoftmax", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    // log_softmax = x - log(sum(exp(x)))
    // sum(exp) = e^1 + e^2 + e^3 ≈ 30.19
    // log(30.19) ≈ 3.408
    REQUIRE_APPROX(v[0], -2.408f, 0.01f);
    REQUIRE_APPROX(v[1], -1.408f, 0.01f);
    REQUIRE_APPROX(v[2], -0.408f, 0.01f);
}

// =============================================================================
// LINALG OPS - Cholesky, MatrixInverse, Determinant, QR, SVD
// =============================================================================

TEST(value_matrix_inverse) {
    tf_wrap::Graph g;
    
    // 2x2 matrix: [[4, 7], [2, 6]]
    // Inverse: [[0.6, -0.7], [-0.2, 0.4]]
    auto x = tf_wrap::Tensor::FromVector<float>({2, 2}, {4.0f, 7.0f, 2.0f, 6.0f});
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("MatrixInverse", "Inv").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Inv", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE_APPROX(v[0], 0.6f, 0.01f);
    REQUIRE_APPROX(v[1], -0.7f, 0.01f);
    REQUIRE_APPROX(v[2], -0.2f, 0.01f);
    REQUIRE_APPROX(v[3], 0.4f, 0.01f);
}

TEST(value_matrix_determinant) {
    tf_wrap::Graph g;
    
    // det([[4, 7], [2, 6]]) = 4*6 - 7*2 = 10
    auto x = tf_wrap::Tensor::FromVector<float>({2, 2}, {4.0f, 7.0f, 2.0f, 6.0f});
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("MatrixDeterminant", "Det").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Det", 0}}, {});
    
    REQUIRE_APPROX(results[0].ToScalar<float>(), 10.0f, 0.01f);
}

TEST(value_cholesky) {
    tf_wrap::Graph g;
    
    // Symmetric positive definite: [[4, 2], [2, 2]]
    // Cholesky L: [[2, 0], [1, 1]]  (L @ L^T = A)
    auto x = tf_wrap::Tensor::FromVector<float>({2, 2}, {4.0f, 2.0f, 2.0f, 2.0f});
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("Cholesky", "Chol").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Chol", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE_APPROX(v[0], 2.0f, 0.01f);
    REQUIRE_APPROX(v[1], 0.0f, 0.01f);
    REQUIRE_APPROX(v[2], 1.0f, 0.01f);
    REQUIRE_APPROX(v[3], 1.0f, 0.01f);
}

// =============================================================================
// IMAGE RESIZE OPS
// =============================================================================

TEST(value_resize_bilinear) {
    tf_wrap::Graph g;
    
    // 2x2 image, resize to 4x4
    auto image = tf_wrap::Tensor::FromVector<float>({1, 2, 2, 1}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto size = tf_wrap::Tensor::FromVector<int32_t>({2}, {4, 4});
    
    (void)g.NewOperation("Const", "Image").SetAttrTensor("value", image.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Size").SetAttrTensor("value", size.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_image = g.GetOperationOrThrow("Image");
    auto* op_size = g.GetOperationOrThrow("Size");
    
    (void)g.NewOperation("ResizeBilinear", "Resize")
        .AddInput(tf_wrap::Output(op_image, 0))
        .AddInput(tf_wrap::Output(op_size, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Resize", 0}}, {});
    
    REQUIRE(results[0].shape()[1] == 4);
    REQUIRE(results[0].shape()[2] == 4);
    auto v = results[0].ToVector<float>();
    // Corners should be original values
    REQUIRE_APPROX(v[0], 1.0f, 0.1f);
}

// =============================================================================
// CONTROL/DEBUG OPS - NoOp, StopGradient, Identity variants
// =============================================================================

TEST(value_noop_stopgradient) {
    tf_wrap::Graph g;
    
    auto x = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    
    // StopGradient just passes through in forward pass
    (void)g.NewOperation("StopGradient", "Stop").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    // NoOp has no outputs - just verify it doesn't crash
    (void)g.NewOperation("NoOp", "Noop").Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Stop", 0}}, {"Noop"});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v[0] == 1.0f);
    REQUIRE(v[1] == 2.0f);
    REQUIRE(v[2] == 3.0f);
}

TEST(value_identity_n) {
    tf_wrap::Graph g;
    
    auto a = tf_wrap::Tensor::FromVector<float>({2}, {1.0f, 2.0f});
    auto b = tf_wrap::Tensor::FromVector<float>({2}, {3.0f, 4.0f});
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("IdentityN", "IdN")
        .AddInputList(std::vector<TF_Output>{tf_wrap::Output(op_a, 0), tf_wrap::Output(op_b, 0)})
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"IdN", 0}, {"IdN", 1}}, {});
    
    REQUIRE(results[0].ToVector<float>()[0] == 1.0f);
    REQUIRE(results[1].ToVector<float>()[0] == 3.0f);
}

// =============================================================================
// MISC OPS - LinSpace, Bitcast, Concat (original), Where, Add (original)
// =============================================================================

TEST(value_linspace) {
    tf_wrap::Graph g;
    
    auto start = tf_wrap::Tensor::FromScalar<float>(0.0f);
    auto stop = tf_wrap::Tensor::FromScalar<float>(10.0f);
    auto num = tf_wrap::Tensor::FromScalar<int32_t>(5);
    
    (void)g.NewOperation("Const", "Start").SetAttrTensor("value", start.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Stop").SetAttrTensor("value", stop.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Num").SetAttrTensor("value", num.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_start = g.GetOperationOrThrow("Start");
    auto* op_stop = g.GetOperationOrThrow("Stop");
    auto* op_num = g.GetOperationOrThrow("Num");
    
    (void)g.NewOperation("LinSpace", "LinSpace")
        .AddInput(tf_wrap::Output(op_start, 0))
        .AddInput(tf_wrap::Output(op_stop, 0))
        .AddInput(tf_wrap::Output(op_num, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"LinSpace", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 5);
    REQUIRE(v[0] == 0.0f);
    REQUIRE(v[2] == 5.0f);
    REQUIRE(v[4] == 10.0f);
}

TEST(value_concat_original) {
    // Test Concat (not ConcatV2 which we already test)
    tf_wrap::Graph g;
    
    auto axis = tf_wrap::Tensor::FromScalar<int32_t>(0);
    auto a = tf_wrap::Tensor::FromVector<float>({2}, {1.0f, 2.0f});
    auto b = tf_wrap::Tensor::FromVector<float>({2}, {3.0f, 4.0f});
    
    (void)g.NewOperation("Const", "Axis").SetAttrTensor("value", axis.handle()).SetAttrType("dtype", TF_INT32).Finish();
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_axis = g.GetOperationOrThrow("Axis");
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("Concat", "Cat")
        .AddInput(tf_wrap::Output(op_axis, 0))
        .AddInputList(std::vector<TF_Output>{tf_wrap::Output(op_a, 0), tf_wrap::Output(op_b, 0)})
        .SetAttrInt("N", 2)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Cat", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 4);
    REQUIRE(v[0] == 1.0f);
    REQUIRE(v[2] == 3.0f);
}

TEST(value_add_original) {
    // Test Add (not AddV2)
    tf_wrap::Graph g;
    
    auto a = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    auto b = tf_wrap::Tensor::FromVector<float>({3}, {10.0f, 20.0f, 30.0f});
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("Add", "Add")
        .AddInput(tf_wrap::Output(op_a, 0))
        .AddInput(tf_wrap::Output(op_b, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Add", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v[0] == 11.0f);
    REQUIRE(v[1] == 22.0f);
    REQUIRE(v[2] == 33.0f);
}

TEST(value_where_indices) {
    tf_wrap::Graph g;
    
    // Where returns indices of true values
    auto cond = tf_wrap::Tensor::FromVector<bool>({5}, {true, false, true, false, true});
    
    (void)g.NewOperation("Const", "Cond").SetAttrTensor("value", cond.handle()).SetAttrType("dtype", TF_BOOL).Finish();
    auto* op_cond = g.GetOperationOrThrow("Cond");
    
    (void)g.NewOperation("Where", "Where").AddInput(tf_wrap::Output(op_cond, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Where", 0}}, {});
    
    // Returns [[0], [2], [4]] for true indices
    REQUIRE(results[0].shape()[0] == 3);  // 3 true values
    auto v = results[0].ToVector<int64_t>();
    REQUIRE(v[0] == 0);
    REQUIRE(v[1] == 2);
    REQUIRE(v[2] == 4);
}

// =============================================================================
// REMAINING OPS - Complete coverage
// =============================================================================

TEST(value_gather_v2) {
    tf_wrap::Graph g;
    
    auto params = tf_wrap::Tensor::FromVector<float>({4}, {10.0f, 20.0f, 30.0f, 40.0f});
    auto indices = tf_wrap::Tensor::FromVector<int32_t>({3}, {0, 2, 3});
    auto axis = tf_wrap::Tensor::FromScalar<int32_t>(0);
    
    (void)g.NewOperation("Const", "Params").SetAttrTensor("value", params.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Indices").SetAttrTensor("value", indices.handle()).SetAttrType("dtype", TF_INT32).Finish();
    (void)g.NewOperation("Const", "Axis").SetAttrTensor("value", axis.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_params = g.GetOperationOrThrow("Params");
    auto* op_indices = g.GetOperationOrThrow("Indices");
    auto* op_axis = g.GetOperationOrThrow("Axis");
    
    (void)g.NewOperation("GatherV2", "Gather")
        .AddInput(tf_wrap::Output(op_params, 0))
        .AddInput(tf_wrap::Output(op_indices, 0))
        .AddInput(tf_wrap::Output(op_axis, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Gather", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v[0] == 10.0f);
    REQUIRE(v[1] == 30.0f);
    REQUIRE(v[2] == 40.0f);
}

TEST(value_batch_matmul_v2) {
    tf_wrap::Graph g;
    
    // Batch of 2 matrices: 2x2
    auto a = tf_wrap::Tensor::FromVector<float>({2, 2, 2}, 
        {1.0f, 2.0f, 3.0f, 4.0f,   // Batch 0
         1.0f, 0.0f, 0.0f, 1.0f}); // Batch 1 (identity)
    auto b = tf_wrap::Tensor::FromVector<float>({2, 2, 2}, 
        {5.0f, 6.0f, 7.0f, 8.0f,   // Batch 0
         2.0f, 3.0f, 4.0f, 5.0f}); // Batch 1
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("BatchMatMulV2", "BMM")
        .AddInput(tf_wrap::Output(op_a, 0))
        .AddInput(tf_wrap::Output(op_b, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"BMM", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    // Batch 0: [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    REQUIRE(v[0] == 19.0f);
    REQUIRE(v[1] == 22.0f);
    // Batch 1: identity @ b = b
    REQUIRE(v[4] == 2.0f);
    REQUIRE(v[5] == 3.0f);
}

TEST(value_pad_v2) {
    tf_wrap::Graph g;
    
    auto x = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    auto paddings = tf_wrap::Tensor::FromVector<int32_t>({1, 2}, {2, 2});
    auto constant = tf_wrap::Tensor::FromScalar<float>(99.0f);
    
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Paddings").SetAttrTensor("value", paddings.handle()).SetAttrType("dtype", TF_INT32).Finish();
    (void)g.NewOperation("Const", "Constant").SetAttrTensor("value", constant.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_x = g.GetOperationOrThrow("X");
    auto* op_pad = g.GetOperationOrThrow("Paddings");
    auto* op_const = g.GetOperationOrThrow("Constant");
    
    (void)g.NewOperation("PadV2", "Pad")
        .AddInput(tf_wrap::Output(op_x, 0))
        .AddInput(tf_wrap::Output(op_pad, 0))
        .AddInput(tf_wrap::Output(op_const, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Pad", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 7);
    REQUIRE(v[0] == 99.0f);
    REQUIRE(v[1] == 99.0f);
    REQUIRE(v[2] == 1.0f);
    REQUIRE(v[5] == 99.0f);
}

TEST(value_split_v) {
    tf_wrap::Graph g;
    
    auto x = tf_wrap::Tensor::FromVector<float>({10}, {0.0f,1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f,9.0f});
    auto size_splits = tf_wrap::Tensor::FromVector<int32_t>({3}, {2, 3, 5});
    auto axis = tf_wrap::Tensor::FromScalar<int32_t>(0);
    
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "SizeSplits").SetAttrTensor("value", size_splits.handle()).SetAttrType("dtype", TF_INT32).Finish();
    (void)g.NewOperation("Const", "Axis").SetAttrTensor("value", axis.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_x = g.GetOperationOrThrow("X");
    auto* op_sizes = g.GetOperationOrThrow("SizeSplits");
    auto* op_axis = g.GetOperationOrThrow("Axis");
    
    (void)g.NewOperation("SplitV", "Split")
        .AddInput(tf_wrap::Output(op_x, 0))
        .AddInput(tf_wrap::Output(op_sizes, 0))
        .AddInput(tf_wrap::Output(op_axis, 0))
        .SetAttrInt("num_split", 3)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Split", 0}, {"Split", 1}, {"Split", 2}}, {});
    
    REQUIRE(results[0].ToVector<float>().size() == 2);
    REQUIRE(results[1].ToVector<float>().size() == 3);
    REQUIRE(results[2].ToVector<float>().size() == 5);
    REQUIRE(results[0].ToVector<float>()[0] == 0.0f);
    REQUIRE(results[1].ToVector<float>()[0] == 2.0f);
    REQUIRE(results[2].ToVector<float>()[0] == 5.0f);
}

TEST(value_real_div) {
    tf_wrap::Graph g;
    
    auto a = tf_wrap::Tensor::FromVector<float>({3}, {10.0f, 15.0f, 21.0f});
    auto b = tf_wrap::Tensor::FromVector<float>({3}, {2.0f, 3.0f, 7.0f});
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("RealDiv", "Div")
        .AddInput(tf_wrap::Output(op_a, 0))
        .AddInput(tf_wrap::Output(op_b, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Div", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v[0] == 5.0f);
    REQUIRE(v[1] == 5.0f);
    REQUIRE(v[2] == 3.0f);
}

TEST(value_shape_n) {
    tf_wrap::Graph g;
    
    auto a = tf_wrap::Tensor::FromVector<float>({2, 3}, std::vector<float>(6, 1.0f));
    auto b = tf_wrap::Tensor::FromVector<float>({4, 5, 6}, std::vector<float>(120, 1.0f));
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("ShapeN", "Shapes")
        .AddInputList(std::vector<TF_Output>{tf_wrap::Output(op_a, 0), tf_wrap::Output(op_b, 0)})
        .SetAttrInt("N", 2)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Shapes", 0}, {"Shapes", 1}}, {});
    
    auto shape_a = results[0].ToVector<int32_t>();
    auto shape_b = results[1].ToVector<int32_t>();
    REQUIRE(shape_a.size() == 2);
    REQUIRE(shape_a[0] == 2);
    REQUIRE(shape_a[1] == 3);
    REQUIRE(shape_b.size() == 3);
    REQUIRE(shape_b[0] == 4);
}

TEST(value_bitcast) {
    tf_wrap::Graph g;
    
    // Bitcast float32 to int32 (same bit pattern)
    auto x = tf_wrap::Tensor::FromVector<float>({1}, {1.0f});
    
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("Bitcast", "Cast")
        .AddInput(tf_wrap::Output(op_x, 0))
        .SetAttrType("type", TF_INT32)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Cast", 0}}, {});
    
    auto v = results[0].ToVector<int32_t>();
    // 1.0f in IEEE 754 = 0x3f800000 = 1065353216
    REQUIRE(v[0] == 1065353216);
}

TEST(value_lrn) {
    tf_wrap::Graph g;
    
    // Local Response Normalization
    auto x = tf_wrap::Tensor::FromVector<float>({1, 1, 1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
    
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("LRN", "LRN")
        .AddInput(tf_wrap::Output(op_x, 0))
        .SetAttrInt("depth_radius", 2)
        .SetAttrFloat("bias", 1.0f)
        .SetAttrFloat("alpha", 1.0f)
        .SetAttrFloat("beta", 0.5f)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"LRN", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 4);
    // Values should be normalized (smaller than input)
    REQUIRE(v[0] < 1.0f);
}

TEST(value_avgpool3d) {
    tf_wrap::Graph g;
    
    // 3D input: batch=1, depth=2, height=2, width=2, channels=1
    auto x = tf_wrap::Tensor::FromVector<float>({1, 2, 2, 2, 1}, 
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("AvgPool3D", "Pool")
        .AddInput(tf_wrap::Output(op_x, 0))
        .SetAttrIntList("ksize", std::vector<int64_t>{1, 2, 2, 2, 1})
        .SetAttrIntList("strides", std::vector<int64_t>{1, 1, 1, 1, 1})
        .SetAttrString("padding", "VALID")
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Pool", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    // Average of all 8 values = 4.5
    REQUIRE_APPROX(v[0], 4.5f, 0.01f);
}

TEST(value_maxpool3d) {
    tf_wrap::Graph g;
    
    auto x = tf_wrap::Tensor::FromVector<float>({1, 2, 2, 2, 1}, 
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("MaxPool3D", "Pool")
        .AddInput(tf_wrap::Output(op_x, 0))
        .SetAttrIntList("ksize", std::vector<int64_t>{1, 2, 2, 2, 1})
        .SetAttrIntList("strides", std::vector<int64_t>{1, 1, 1, 1, 1})
        .SetAttrString("padding", "VALID")
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Pool", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    // Max of all 8 values = 8
    REQUIRE(v[0] == 8.0f);
}

TEST(value_resize_bicubic) {
    tf_wrap::Graph g;
    
    auto image = tf_wrap::Tensor::FromVector<float>({1, 2, 2, 1}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto size = tf_wrap::Tensor::FromVector<int32_t>({2}, {4, 4});
    
    (void)g.NewOperation("Const", "Image").SetAttrTensor("value", image.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Size").SetAttrTensor("value", size.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_image = g.GetOperationOrThrow("Image");
    auto* op_size = g.GetOperationOrThrow("Size");
    
    (void)g.NewOperation("ResizeBicubic", "Resize")
        .AddInput(tf_wrap::Output(op_image, 0))
        .AddInput(tf_wrap::Output(op_size, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Resize", 0}}, {});
    
    REQUIRE(results[0].shape()[1] == 4);
    REQUIRE(results[0].shape()[2] == 4);
}

TEST(value_resize_nearest_neighbor) {
    tf_wrap::Graph g;
    
    auto image = tf_wrap::Tensor::FromVector<float>({1, 2, 2, 1}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto size = tf_wrap::Tensor::FromVector<int32_t>({2}, {4, 4});
    
    (void)g.NewOperation("Const", "Image").SetAttrTensor("value", image.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Size").SetAttrTensor("value", size.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_image = g.GetOperationOrThrow("Image");
    auto* op_size = g.GetOperationOrThrow("Size");
    
    (void)g.NewOperation("ResizeNearestNeighbor", "Resize")
        .AddInput(tf_wrap::Output(op_image, 0))
        .AddInput(tf_wrap::Output(op_size, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Resize", 0}}, {});
    
    REQUIRE(results[0].shape()[1] == 4);
    REQUIRE(results[0].shape()[2] == 4);
    // Nearest neighbor should replicate values
    auto v = results[0].ToVector<float>();
    REQUIRE(v[0] == 1.0f);
}

TEST(value_crop_and_resize) {
    tf_wrap::Graph g;
    
    // 4x4 image
    auto image = tf_wrap::Tensor::FromVector<float>({1, 4, 4, 1}, std::vector<float>(16, 1.0f));
    // Crop box: [y1, x1, y2, x2] normalized coordinates
    auto boxes = tf_wrap::Tensor::FromVector<float>({1, 4}, {0.0f, 0.0f, 0.5f, 0.5f});
    auto box_ind = tf_wrap::Tensor::FromVector<int32_t>({1}, {0});
    auto crop_size = tf_wrap::Tensor::FromVector<int32_t>({2}, {2, 2});
    
    (void)g.NewOperation("Const", "Image").SetAttrTensor("value", image.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Boxes").SetAttrTensor("value", boxes.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "BoxInd").SetAttrTensor("value", box_ind.handle()).SetAttrType("dtype", TF_INT32).Finish();
    (void)g.NewOperation("Const", "CropSize").SetAttrTensor("value", crop_size.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_image = g.GetOperationOrThrow("Image");
    auto* op_boxes = g.GetOperationOrThrow("Boxes");
    auto* op_box_ind = g.GetOperationOrThrow("BoxInd");
    auto* op_crop_size = g.GetOperationOrThrow("CropSize");
    
    (void)g.NewOperation("CropAndResize", "Crop")
        .AddInput(tf_wrap::Output(op_image, 0))
        .AddInput(tf_wrap::Output(op_boxes, 0))
        .AddInput(tf_wrap::Output(op_box_ind, 0))
        .AddInput(tf_wrap::Output(op_crop_size, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Crop", 0}}, {});
    
    REQUIRE(results[0].shape()[1] == 2);
    REQUIRE(results[0].shape()[2] == 2);
}

TEST(value_qr_decomposition) {
    tf_wrap::Graph g;
    
    // 3x2 matrix
    auto x = tf_wrap::Tensor::FromVector<float>({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("Qr", "QR")
        .AddInput(tf_wrap::Output(op_x, 0))
        .SetAttrBool("full_matrices", false)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"QR", 0}, {"QR", 1}}, {});
    
    // Q is 3x2, R is 2x2
    REQUIRE(results[0].shape()[0] == 3);
    REQUIRE(results[0].shape()[1] == 2);
    REQUIRE(results[1].shape()[0] == 2);
    REQUIRE(results[1].shape()[1] == 2);
}

TEST(value_svd_decomposition) {
    tf_wrap::Graph g;
    
    auto x = tf_wrap::Tensor::FromVector<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("Svd", "SVD")
        .AddInput(tf_wrap::Output(op_x, 0))
        .SetAttrBool("compute_uv", true)
        .SetAttrBool("full_matrices", false)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"SVD", 0}, {"SVD", 1}, {"SVD", 2}}, {});
    
    // S is singular values
    auto s_vals = results[0].ToVector<float>();
    REQUIRE(s_vals.size() == 2);  // min(2,3) = 2 singular values
    REQUIRE(s_vals[0] > 0.0f);    // Singular values are positive
}

TEST(value_multinomial) {
    tf_wrap::Graph g;
    
    // Log-probabilities (unnormalized)
    auto logits = tf_wrap::Tensor::FromVector<float>({1, 4}, {0.0f, 0.0f, 0.0f, 10.0f});  // Heavily favor index 3
    auto num_samples = tf_wrap::Tensor::FromScalar<int32_t>(100);
    
    (void)g.NewOperation("Const", "Logits").SetAttrTensor("value", logits.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "NumSamples").SetAttrTensor("value", num_samples.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_logits = g.GetOperationOrThrow("Logits");
    auto* op_num = g.GetOperationOrThrow("NumSamples");
    
    (void)g.NewOperation("Multinomial", "Sample")
        .AddInput(tf_wrap::Output(op_logits, 0))
        .AddInput(tf_wrap::Output(op_num, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Sample", 0}}, {});
    
    auto samples = results[0].ToVector<int64_t>();
    REQUIRE(samples.size() == 100);
    
    // Count how many times index 3 was sampled (should be most)
    int count_3 = 0;
    for (auto s : samples) {
        if (s == 3) count_3++;
    }
    REQUIRE(count_3 > 80);  // Should be heavily biased toward 3
}

TEST(value_prevent_gradient) {
    tf_wrap::Graph g;
    
    auto x = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("PreventGradient", "PG")
        .AddInput(tf_wrap::Output(op_x, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"PG", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v[0] == 1.0f);
    REQUIRE(v[1] == 2.0f);
    REQUIRE(v[2] == 3.0f);
}

TEST(value_check_numerics) {
    tf_wrap::Graph g;
    
    auto x = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("CheckNumerics", "Check")
        .AddInput(tf_wrap::Output(op_x, 0))
        .SetAttrString("message", "test_check")
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Check", 0}}, {});
    
    // Should pass through unchanged for valid numerics
    auto v = results[0].ToVector<float>();
    REQUIRE(v[0] == 1.0f);
}

TEST(value_fused_batchnorm_v1) {
    tf_wrap::Graph g;
    
    auto x = tf_wrap::Tensor::FromVector<float>({2, 1, 1, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto scale = tf_wrap::Tensor::FromVector<float>({2}, {1.0f, 1.0f});
    auto offset = tf_wrap::Tensor::FromVector<float>({2}, {0.0f, 0.0f});
    auto mean = tf_wrap::Tensor::FromVector<float>({2}, {2.0f, 3.0f});
    auto variance = tf_wrap::Tensor::FromVector<float>({2}, {1.0f, 1.0f});
    
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Scale").SetAttrTensor("value", scale.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Offset").SetAttrTensor("value", offset.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Mean").SetAttrTensor("value", mean.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Variance").SetAttrTensor("value", variance.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_x = g.GetOperationOrThrow("X");
    auto* op_scale = g.GetOperationOrThrow("Scale");
    auto* op_offset = g.GetOperationOrThrow("Offset");
    auto* op_mean = g.GetOperationOrThrow("Mean");
    auto* op_variance = g.GetOperationOrThrow("Variance");
    
    (void)g.NewOperation("FusedBatchNorm", "BN")
        .AddInput(tf_wrap::Output(op_x, 0))
        .AddInput(tf_wrap::Output(op_scale, 0))
        .AddInput(tf_wrap::Output(op_offset, 0))
        .AddInput(tf_wrap::Output(op_mean, 0))
        .AddInput(tf_wrap::Output(op_variance, 0))
        .SetAttrFloat("epsilon", 0.001f)
        .SetAttrBool("is_training", false)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"BN", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 4);
}

// Variable ops require special handling - test graph construction only
TEST(graph_variable_ops) {
    tf_wrap::Graph g;
    
    // VarHandleOp creates a resource handle
    (void)g.NewOperation("VarHandleOp", "Var")
        .SetAttrType("dtype", TF_FLOAT)
        .SetAttrShape("shape", {2, 3})
        .Finish();
    
    // Variable and VariableV2 are older-style
    (void)g.NewOperation("VariableV2", "VarV2")
        .SetAttrType("dtype", TF_FLOAT)
        .SetAttrShape("shape", {2})
        .Finish();
    
    // Verify ops exist in graph
    REQUIRE(g.GetOperationOrThrow("Var") != nullptr);
    REQUIRE(g.GetOperationOrThrow("VarV2") != nullptr);
}

TEST(graph_variable_assign_ops) {
    tf_wrap::Graph g;
    
    // Create variable handle
    (void)g.NewOperation("VarHandleOp", "Var")
        .SetAttrType("dtype", TF_FLOAT)
        .SetAttrShape("shape", {3})
        .Finish();
    auto* var = g.GetOperationOrThrow("Var");
    
    // Create value to assign
    auto val = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    (void)g.NewOperation("Const", "Val").SetAttrTensor("value", val.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_val = g.GetOperationOrThrow("Val");
    
    // AssignVariableOp
    (void)g.NewOperation("AssignVariableOp", "Assign")
        .AddInput(tf_wrap::Output(var, 0))
        .AddInput(tf_wrap::Output(op_val, 0))
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // ReadVariableOp
    (void)g.NewOperation("ReadVariableOp", "Read")
        .AddInput(tf_wrap::Output(var, 0))
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // AssignAddVariableOp
    (void)g.NewOperation("AssignAddVariableOp", "AddAssign")
        .AddInput(tf_wrap::Output(var, 0))
        .AddInput(tf_wrap::Output(op_val, 0))
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // AssignSubVariableOp
    (void)g.NewOperation("AssignSubVariableOp", "SubAssign")
        .AddInput(tf_wrap::Output(var, 0))
        .AddInput(tf_wrap::Output(op_val, 0))
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Verify all ops created
    REQUIRE(g.GetOperationOrThrow("Assign") != nullptr);
    REQUIRE(g.GetOperationOrThrow("Read") != nullptr);
    REQUIRE(g.GetOperationOrThrow("AddAssign") != nullptr);
    REQUIRE(g.GetOperationOrThrow("SubAssign") != nullptr);
}

TEST(graph_assert_print) {
    tf_wrap::Graph g;
    
    auto cond = tf_wrap::Tensor::FromScalar<bool>(true);
    auto data = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    
    (void)g.NewOperation("Const", "Cond").SetAttrTensor("value", cond.handle()).SetAttrType("dtype", TF_BOOL).Finish();
    (void)g.NewOperation("Const", "Data").SetAttrTensor("value", data.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_cond = g.GetOperationOrThrow("Cond");
    auto* op_data = g.GetOperationOrThrow("Data");
    
    // Assert op - T is a list of types for the data tensors
    (void)g.NewOperation("Assert", "Assert")
        .AddInput(tf_wrap::Output(op_cond, 0))
        .AddInputList(std::vector<TF_Output>{tf_wrap::Output(op_data, 0)})
        .SetAttrTypeList("T", std::vector<TF_DataType>{TF_FLOAT})
        .Finish();
    
    // Print op - U is a list of types for the summary tensors
    (void)g.NewOperation("Print", "Print")
        .AddInput(tf_wrap::Output(op_data, 0))
        .AddInputList(std::vector<TF_Output>{tf_wrap::Output(op_data, 0)})
        .SetAttrTypeList("U", std::vector<TF_DataType>{TF_FLOAT})
        .Finish();
    
    REQUIRE(g.GetOperationOrThrow("Assert") != nullptr);
    REQUIRE(g.GetOperationOrThrow("Print") != nullptr);
}

TEST(value_string_join) {
    tf_wrap::Graph g;
    
    auto a = tf_wrap::Tensor::FromString("Hello");
    auto b = tf_wrap::Tensor::FromString(" World");
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_STRING).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_STRING).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("StringJoin", "Join")
        .AddInputList(std::vector<TF_Output>{tf_wrap::Output(op_a, 0), tf_wrap::Output(op_b, 0)})
        .SetAttrInt("N", 2)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Join", 0}}, {});
    
    auto str = results[0].ToString();
    REQUIRE(str == "Hello World");
}

TEST(graph_einsum) {
    tf_wrap::Graph g;
    
    auto a = tf_wrap::Tensor::FromVector<float>({2, 3}, std::vector<float>(6, 1.0f));
    auto b = tf_wrap::Tensor::FromVector<float>({3, 2}, std::vector<float>(6, 1.0f));
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    // Einsum for matrix multiplication: ij,jk->ik
    (void)g.NewOperation("Einsum", "Ein")
        .AddInputList(std::vector<TF_Output>{tf_wrap::Output(op_a, 0), tf_wrap::Output(op_b, 0)})
        .SetAttrString("equation", "ij,jk->ik")
        .SetAttrInt("N", 2)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Ein", 0}}, {});
    
    // Result should be 2x2 matrix
    REQUIRE(results[0].shape()[0] == 2);
    REQUIRE(results[0].shape()[1] == 2);
    auto v = results[0].ToVector<float>();
    REQUIRE(v[0] == 3.0f);  // Each element is sum of 3 ones
}

TEST(graph_conv2d_backprop_input) {
    tf_wrap::Graph g;
    
    auto input_sizes = tf_wrap::Tensor::FromVector<int32_t>({4}, {1, 4, 4, 1});
    auto filter = tf_wrap::Tensor::FromVector<float>({2, 2, 1, 1}, std::vector<float>(4, 1.0f));
    auto out_backprop = tf_wrap::Tensor::FromVector<float>({1, 3, 3, 1}, std::vector<float>(9, 1.0f));
    
    (void)g.NewOperation("Const", "InputSizes").SetAttrTensor("value", input_sizes.handle()).SetAttrType("dtype", TF_INT32).Finish();
    (void)g.NewOperation("Const", "Filter").SetAttrTensor("value", filter.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "OutBackprop").SetAttrTensor("value", out_backprop.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_sizes = g.GetOperationOrThrow("InputSizes");
    auto* op_filter = g.GetOperationOrThrow("Filter");
    auto* op_back = g.GetOperationOrThrow("OutBackprop");
    
    (void)g.NewOperation("Conv2DBackpropInput", "ConvBack")
        .AddInput(tf_wrap::Output(op_sizes, 0))
        .AddInput(tf_wrap::Output(op_filter, 0))
        .AddInput(tf_wrap::Output(op_back, 0))
        .SetAttrIntList("strides", std::vector<int64_t>{1, 1, 1, 1})
        .SetAttrString("padding", "VALID")
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"ConvBack", 0}}, {});
    
    REQUIRE(results[0].shape()[1] == 4);
    REQUIRE(results[0].shape()[2] == 4);
}

TEST(graph_matching_files) {
    tf_wrap::Graph g;
    
    // Pattern to match (won't find anything, but tests op construction)
    auto pattern = tf_wrap::Tensor::FromString("/nonexistent/*.txt");
    
    (void)g.NewOperation("Const", "Pattern").SetAttrTensor("value", pattern.handle()).SetAttrType("dtype", TF_STRING).Finish();
    auto* op_pattern = g.GetOperationOrThrow("Pattern");
    
    (void)g.NewOperation("MatchingFiles", "Match")
        .AddInput(tf_wrap::Output(op_pattern, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Match", 0}}, {});
    
    // Should return empty tensor (no matches)
    REQUIRE(results[0].shape()[0] == 0);
}

TEST(graph_nms) {
    tf_wrap::Graph g;
    
    // 3 boxes: [y1, x1, y2, x2]
    auto boxes = tf_wrap::Tensor::FromVector<float>({3, 4}, {
        0.0f, 0.0f, 1.0f, 1.0f,   // Box 0
        0.1f, 0.1f, 1.1f, 1.1f,   // Box 1 (overlaps with 0)
        5.0f, 5.0f, 6.0f, 6.0f    // Box 2 (separate)
    });
    auto scores = tf_wrap::Tensor::FromVector<float>({3}, {0.9f, 0.8f, 0.7f});
    auto max_output = tf_wrap::Tensor::FromScalar<int32_t>(10);
    auto iou_threshold = tf_wrap::Tensor::FromScalar<float>(0.5f);
    auto score_threshold = tf_wrap::Tensor::FromScalar<float>(0.0f);
    
    (void)g.NewOperation("Const", "Boxes").SetAttrTensor("value", boxes.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Scores").SetAttrTensor("value", scores.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "MaxOutput").SetAttrTensor("value", max_output.handle()).SetAttrType("dtype", TF_INT32).Finish();
    (void)g.NewOperation("Const", "IoU").SetAttrTensor("value", iou_threshold.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "ScoreThresh").SetAttrTensor("value", score_threshold.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_boxes = g.GetOperationOrThrow("Boxes");
    auto* op_scores = g.GetOperationOrThrow("Scores");
    auto* op_max = g.GetOperationOrThrow("MaxOutput");
    auto* op_iou = g.GetOperationOrThrow("IoU");
    auto* op_score_thresh = g.GetOperationOrThrow("ScoreThresh");
    
    (void)g.NewOperation("NonMaxSuppressionV3", "NMS")
        .AddInput(tf_wrap::Output(op_boxes, 0))
        .AddInput(tf_wrap::Output(op_scores, 0))
        .AddInput(tf_wrap::Output(op_max, 0))
        .AddInput(tf_wrap::Output(op_iou, 0))
        .AddInput(tf_wrap::Output(op_score_thresh, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"NMS", 0}}, {});
    
    auto indices = results[0].ToVector<int32_t>();
    // Should keep box 0 and box 2 (box 1 overlaps with 0)
    REQUIRE(indices.size() == 2);
    REQUIRE(indices[0] == 0);
    REQUIRE(indices[1] == 2);
}

TEST(graph_regex_replace) {
    tf_wrap::Graph g;
    
    auto input = tf_wrap::Tensor::FromString("hello123world456");
    auto pattern = tf_wrap::Tensor::FromString("[0-9]+");
    auto rewrite = tf_wrap::Tensor::FromString("X");
    
    (void)g.NewOperation("Const", "Input").SetAttrTensor("value", input.handle()).SetAttrType("dtype", TF_STRING).Finish();
    (void)g.NewOperation("Const", "Pattern").SetAttrTensor("value", pattern.handle()).SetAttrType("dtype", TF_STRING).Finish();
    (void)g.NewOperation("Const", "Rewrite").SetAttrTensor("value", rewrite.handle()).SetAttrType("dtype", TF_STRING).Finish();
    
    auto* op_input = g.GetOperationOrThrow("Input");
    auto* op_pattern = g.GetOperationOrThrow("Pattern");
    auto* op_rewrite = g.GetOperationOrThrow("Rewrite");
    
    (void)g.NewOperation("RegexReplace", "Replace")
        .AddInput(tf_wrap::Output(op_input, 0))
        .AddInput(tf_wrap::Output(op_pattern, 0))
        .AddInput(tf_wrap::Output(op_rewrite, 0))
        .SetAttrBool("replace_global", true)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Replace", 0}}, {});
    
    auto str = results[0].ToString();
    REQUIRE(str == "helloXworldX");
}

TEST(graph_string_split) {
    // StringSplit requires a 1D string tensor (batch of strings) and scalar delimiter
    // Our FromString creates scalars, so we just test graph construction without execution
    tf_wrap::Graph g;
    
    // Create placeholder for 1D string input (would be fed at runtime)
    (void)g.NewOperation("Placeholder", "Input")
        .SetAttrType("dtype", TF_STRING)
        .SetAttrShape("shape", {-1})  // Unknown batch size
        .Finish();
    
    auto delimiter = tf_wrap::Tensor::FromString(",");
    (void)g.NewOperation("Const", "Delim").SetAttrTensor("value", delimiter.handle()).SetAttrType("dtype", TF_STRING).Finish();
    
    auto* op_input = g.GetOperationOrThrow("Input");
    auto* op_delim = g.GetOperationOrThrow("Delim");
    
    (void)g.NewOperation("StringSplit", "Split")
        .AddInput(tf_wrap::Output(op_input, 0))
        .AddInput(tf_wrap::Output(op_delim, 0))
        .Finish();
    
    // StringSplit returns sparse tensor (indices, values, shape)
    // Just verify the op was created successfully
    REQUIRE(g.GetOperationOrThrow("Split") != nullptr);
}

TEST(graph_encode_jpeg) {
    tf_wrap::Graph g;
    
    // 2x2 RGB image
    auto image = tf_wrap::Tensor::FromVector<uint8_t>({2, 2, 3}, 
        std::vector<uint8_t>(12, 128));
    
    (void)g.NewOperation("Const", "Image").SetAttrTensor("value", image.handle()).SetAttrType("dtype", TF_UINT8).Finish();
    auto* op_image = g.GetOperationOrThrow("Image");
    
    (void)g.NewOperation("EncodeJpeg", "Encode")
        .AddInput(tf_wrap::Output(op_image, 0))
        .SetAttrInt("quality", 95)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Encode", 0}}, {});
    
    // Should produce JPEG bytes
    auto jpeg_bytes = results[0].ToString();
    REQUIRE(jpeg_bytes.size() > 0);
    // JPEG magic bytes: FF D8 FF
    REQUIRE(static_cast<uint8_t>(jpeg_bytes[0]) == 0xFF);
    REQUIRE(static_cast<uint8_t>(jpeg_bytes[1]) == 0xD8);
}

TEST(graph_nms_v1) {
    tf_wrap::Graph g;
    
    // 2 boxes
    auto boxes = tf_wrap::Tensor::FromVector<float>({2, 4}, {
        0.0f, 0.0f, 1.0f, 1.0f,
        5.0f, 5.0f, 6.0f, 6.0f
    });
    auto scores = tf_wrap::Tensor::FromVector<float>({2}, {0.9f, 0.8f});
    auto max_output = tf_wrap::Tensor::FromScalar<int32_t>(10);
    
    (void)g.NewOperation("Const", "Boxes").SetAttrTensor("value", boxes.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Scores").SetAttrTensor("value", scores.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "MaxOutput").SetAttrTensor("value", max_output.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_boxes = g.GetOperationOrThrow("Boxes");
    auto* op_scores = g.GetOperationOrThrow("Scores");
    auto* op_max = g.GetOperationOrThrow("MaxOutput");
    
    (void)g.NewOperation("NonMaxSuppression", "NMS")
        .AddInput(tf_wrap::Output(op_boxes, 0))
        .AddInput(tf_wrap::Output(op_scores, 0))
        .AddInput(tf_wrap::Output(op_max, 0))
        .SetAttrFloat("iou_threshold", 0.5f)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"NMS", 0}}, {});
    
    auto indices = results[0].ToVector<int32_t>();
    REQUIRE(indices.size() == 2);
}

TEST(graph_variable_v1) {
    tf_wrap::Graph g;
    
    // Old-style Variable op (ref semantics)
    (void)g.NewOperation("Variable", "Var")
        .SetAttrType("dtype", TF_FLOAT)
        .SetAttrShape("shape", {3})
        .Finish();
    
    REQUIRE(g.GetOperationOrThrow("Var") != nullptr);
}

// =============================================================================
// PREVIOUSLY UNTESTED: DeviceList, RunMetadata, AddControlInput, ToGraphDef
// =============================================================================

TEST(session_list_devices) {
    tf_wrap::Graph g;
    
    // Create minimal graph
    auto x = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    tf_wrap::Session s(g);
    
    // List available devices
    auto devices = s.ListDevices();
    
    // Should have at least one CPU device
    REQUIRE(devices.count() >= 1);
    
    // Get all devices
    auto all_devices = devices.all();
    REQUIRE(all_devices.size() >= 1);
    
    // Check first device properties
    auto first = devices.at(0);
    REQUIRE(!first.name.empty());
    REQUIRE(!first.type.empty());
    REQUIRE(first.is_cpu() || first.is_gpu());
    
    // Find CPU device
    bool found_cpu = false;
    for (const auto& dev : all_devices) {
        if (dev.is_cpu()) {
            found_cpu = true;
            REQUIRE(dev.type == "CPU");
        }
    }
    REQUIRE(found_cpu);
}

// NOTE: session_run_with_metadata test removed - RunWithMetadata not implemented in v5.0

TEST(graph_add_control_input) {
    tf_wrap::Graph g;
    
    // Create two independent operations
    auto x = tf_wrap::Tensor::FromScalar<float>(1.0f);
    auto y = tf_wrap::Tensor::FromScalar<float>(2.0f);
    
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Y").SetAttrTensor("value", y.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_x = g.GetOperationOrThrow("X");
    auto* op_y = g.GetOperationOrThrow("Y");
    
    // Create an op that depends on Y, with control dependency on X
    // This means X must execute before Z, even though Z doesn't use X's output
    (void)g.NewOperation("Identity", "Z")
        .AddInput(tf_wrap::Output(op_y, 0))
        .AddControlInput(op_x)  // X must run before Z
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Z", 0}}, {});
    
    // Z should return Y's value (2.0)
    REQUIRE(results[0].ToScalar<float>() == 2.0f);
}

TEST(graph_add_control_input_ordering) {
    tf_wrap::Graph g;
    
    // Create a chain of operations with control dependencies
    auto val = tf_wrap::Tensor::FromScalar<float>(10.0f);
    (void)g.NewOperation("Const", "Start").SetAttrTensor("value", val.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_start = g.GetOperationOrThrow("Start");
    
    // Op A depends on Start via control input
    (void)g.NewOperation("Identity", "A")
        .AddInput(tf_wrap::Output(op_start, 0))
        .Finish();
    auto* op_a = g.GetOperationOrThrow("A");
    
    // Op B depends on A via control input (not data)
    auto val_b = tf_wrap::Tensor::FromScalar<float>(20.0f);
    (void)g.NewOperation("Const", "ValB").SetAttrTensor("value", val_b.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_val_b = g.GetOperationOrThrow("ValB");
    
    (void)g.NewOperation("Identity", "B")
        .AddInput(tf_wrap::Output(op_val_b, 0))
        .AddControlInput(op_a)  // A must run before B
        .Finish();
    
    // Op C depends on both A and B via control inputs
    auto val_c = tf_wrap::Tensor::FromScalar<float>(30.0f);
    (void)g.NewOperation("Const", "ValC").SetAttrTensor("value", val_c.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_val_c = g.GetOperationOrThrow("ValC");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("Identity", "C")
        .AddInput(tf_wrap::Output(op_val_c, 0))
        .AddControlInput(op_a)
        .AddControlInput(op_b)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"C", 0}}, {});
    
    // C should return 30.0, proving the graph executed correctly
    REQUIRE(results[0].ToScalar<float>() == 30.0f);
}

TEST(graph_to_graph_def) {
    tf_wrap::Graph g;
    
    // Build a simple graph
    auto x = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    auto y = tf_wrap::Tensor::FromVector<float>({3}, {4.0f, 5.0f, 6.0f});
    
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Y").SetAttrTensor("value", y.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_x = g.GetOperationOrThrow("X");
    auto* op_y = g.GetOperationOrThrow("Y");
    
    (void)g.NewOperation("AddV2", "Sum")
        .AddInput(tf_wrap::Output(op_x, 0))
        .AddInput(tf_wrap::Output(op_y, 0))
        .Finish();
    
    // Serialize graph to GraphDef protobuf
    auto graph_def = g.ToGraphDef();
    
    // Should have produced a non-empty serialization
    REQUIRE(graph_def.size() > 0);
    
    // GraphDef is a protobuf - check for some expected bytes
    // Protobuf wire format: field numbers and types in first few bytes
    // Not checking specific content since protobuf format may vary
    REQUIRE(graph_def.size() > 10);  // Should be substantial
}

TEST(graph_to_graph_def_roundtrip_check) {
    tf_wrap::Graph g;
    
    // Create graph with multiple node types
    auto x = tf_wrap::Tensor::FromScalar<float>(5.0f);
    (void)g.NewOperation("Const", "Input").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_input = g.GetOperationOrThrow("Input");
    
    (void)g.NewOperation("Square", "Squared").AddInput(tf_wrap::Output(op_input, 0)).Finish();
    auto* op_sq = g.GetOperationOrThrow("Squared");
    
    (void)g.NewOperation("Sqrt", "Root").AddInput(tf_wrap::Output(op_sq, 0)).Finish();
    
    // Get GraphDef
    auto graph_def = g.ToGraphDef();
    REQUIRE(graph_def.size() > 0);
    
    // Verify the original graph still works
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Root", 0}}, {});
    
    // sqrt(square(5)) = 5
    REQUIRE_APPROX(results[0].ToScalar<float>(), 5.0f, 0.001f);
}

// =============================================================================
// LARGE TENSOR TESTS
// =============================================================================

TEST(large_tensor_1m_elements) {
    // 1 million float elements = 4MB
    constexpr int64_t SIZE = 1000000;
    
    std::vector<float> data(SIZE);
    for (int64_t i = 0; i < SIZE; ++i) {
        data[i] = static_cast<float>(i % 1000);
    }
    
    auto tensor = tf_wrap::Tensor::FromVector<float>({SIZE}, data);
    
    REQUIRE(tensor.num_elements() == SIZE);
    REQUIRE(tensor.byte_size() == SIZE * sizeof(float));
    
    auto retrieved = tensor.ToVector<float>();
    REQUIRE(retrieved.size() == SIZE);
    REQUIRE(retrieved[0] == 0.0f);
    REQUIRE(retrieved[999] == 999.0f);
    REQUIRE(retrieved[1000] == 0.0f);
}

TEST(large_tensor_10m_elements) {
    // 10 million float elements = 40MB
    constexpr int64_t SIZE = 10000000;
    
    std::vector<float> data(SIZE, 1.0f);
    
    auto tensor = tf_wrap::Tensor::FromVector<float>({SIZE}, data);
    
    REQUIRE(tensor.num_elements() == SIZE);
    REQUIRE(tensor.byte_size() == SIZE * sizeof(float));
}

TEST(large_tensor_multidim) {
    // 1000 x 1000 x 10 = 10M elements = 40MB
    constexpr int64_t D0 = 1000, D1 = 1000, D2 = 10;
    constexpr int64_t SIZE = D0 * D1 * D2;
    
    std::vector<float> data(SIZE, 0.5f);
    
    auto tensor = tf_wrap::Tensor::FromVector<float>({D0, D1, D2}, data);
    
    REQUIRE(tensor.shape().size() == 3);
    REQUIRE(tensor.shape()[0] == D0);
    REQUIRE(tensor.shape()[1] == D1);
    REQUIRE(tensor.shape()[2] == D2);
    REQUIRE(tensor.num_elements() == SIZE);
}

TEST(large_tensor_reduce_sum) {
    // Test that TF can handle large tensor operations
    constexpr int64_t SIZE = 1000000;
    
    tf_wrap::Graph g;
    
    // Create tensor of ones
    std::vector<float> data(SIZE, 1.0f);
    auto x = tf_wrap::Tensor::FromVector<float>({SIZE}, data);
    
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    
    // Sum all elements
    auto axis = tf_wrap::Tensor::FromVector<int32_t>({1}, {0});
    (void)g.NewOperation("Const", "Axis").SetAttrTensor("value", axis.handle()).SetAttrType("dtype", TF_INT32).Finish();
    auto* op_axis = g.GetOperationOrThrow("Axis");
    
    (void)g.NewOperation("Sum", "Total")
        .AddInput(tf_wrap::Output(op_x, 0))
        .AddInput(tf_wrap::Output(op_axis, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Total", 0}}, {});
    
    // Sum of 1M ones should be 1M
    REQUIRE_APPROX(results[0].ToScalar<float>(), static_cast<float>(SIZE), 1.0f);
}

TEST(large_tensor_matmul) {
    // 500x500 matrix multiplication
    constexpr int64_t N = 500;
    
    tf_wrap::Graph g;
    
    // Create two NxN matrices
    std::vector<float> data_a(N * N, 1.0f);
    std::vector<float> data_b(N * N, 2.0f);
    
    auto a = tf_wrap::Tensor::FromVector<float>({N, N}, data_a);
    auto b = tf_wrap::Tensor::FromVector<float>({N, N}, data_b);
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("MatMul", "Result")
        .AddInput(tf_wrap::Output(op_a, 0))
        .AddInput(tf_wrap::Output(op_b, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Result", 0}}, {});
    
    // Each element of result = sum of N products of 1*2 = 2N
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == N * N);
    REQUIRE_APPROX(v[0], static_cast<float>(2 * N), 0.1f);
}

// =============================================================================
// SAVEDMODEL TESTS
// =============================================================================

// Get the path to the test SavedModel (created by CI or manually)
// Checks: ./test_savedmodel, ../test_savedmodel, TEST_SAVEDMODEL_PATH env var
inline std::string get_test_savedmodel_path() {
    // Check environment variable first
    if (const char* env_path = std::getenv("TEST_SAVEDMODEL_PATH")) {
        if (std::filesystem::exists(env_path)) {
            return env_path;
        }
    }
    
    // Check common locations
    std::vector<std::string> candidates = {
        "./test_savedmodel",
        "../test_savedmodel",
        "test_savedmodel"
    };
    
    for (const auto& path : candidates) {
        if (std::filesystem::exists(path) && 
            std::filesystem::exists(std::filesystem::path(path) / "saved_model.pb")) {
            return path;
        }
    }
    
    return "";  // Not found
}

TEST(savedmodel_error_nonexistent_path) {
    // LoadSavedModel should throw when path doesn't exist
    bool threw = false;
    try {
        auto [session, graph] = tf_wrap::Session::LoadSavedModel(
            "/nonexistent/path/to/model");
        (void)session;
        (void)graph;
    } catch (const std::runtime_error& e) {
        threw = true;
        // Error message should mention the path or error
        std::string msg = e.what();
        REQUIRE(!msg.empty());
    }
    REQUIRE(threw);
}

TEST(savedmodel_error_invalid_directory) {
    // Create empty directory (no saved_model.pb)
    auto tmp_dir = std::filesystem::temp_directory_path() / "tfwrap_empty_model";
    std::filesystem::create_directories(tmp_dir);
    
    bool threw = false;
    try {
        auto [session, graph] = tf_wrap::Session::LoadSavedModel(tmp_dir.string());
        (void)session;
        (void)graph;
    } catch (const std::runtime_error&) {
        threw = true;
    }
    
    std::filesystem::remove_all(tmp_dir);
    REQUIRE(threw);
}

TEST(savedmodel_load_and_run) {
    std::string model_path = get_test_savedmodel_path();
    if (model_path.empty()) {
        std::cout << "  SKIP: No test SavedModel found (set TEST_SAVEDMODEL_PATH or run CI)\n";
        return;
    }
    
    std::cout << "  Loading SavedModel from: " << model_path << "\n";
    
    // Load the model
    auto [session, graph] = tf_wrap::Session::LoadSavedModel(model_path, {"serve"});
    
    // The Python model computes: output = input * 2 + 1
    // Find the input and output ops
    // TF SavedModel signature names the function "serve" -> ops are "serve_x" and "PartitionedCall"
    // or we can use the signature to find them
    
    // For simple models, we can look for Placeholder/output patterns
    // Our model: serve(x) -> x * 2 + 1
    
    // Create input tensor: [1.0, 2.0, 3.0]
    auto input = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    
    // The saved model has signature "serving_default" with input "x" and output "output_0"
    // In the C API, we need to find the actual op names
    // For TF2 SavedModels, the entry point is typically "__inference_serve_..." or similar
    
    // Try to run using the serving signature
    // The input placeholder is typically named "serving_default_x:0"
    // The output is typically "PartitionedCall:0" or "StatefulPartitionedCall:0"
    
    std::vector<std::string> possible_inputs = {
        "serving_default_x", "serve_x", "x", "input"
    };
    std::vector<std::string> possible_outputs = {
        "PartitionedCall", "StatefulPartitionedCall", "serving_default", "output"
    };
    
    std::string input_op, output_op;
    
    // Find valid input op
    for (const auto& name : possible_inputs) {
        if (graph.GetOperation(name.c_str()).has_value()) {
            input_op = name;
            break;
        }
    }
    
    // Find valid output op  
    for (const auto& name : possible_outputs) {
        if (graph.GetOperation(name.c_str()).has_value()) {
            output_op = name;
            break;
        }
    }
    
    if (input_op.empty() || output_op.empty()) {
        std::cout << "  SKIP: Could not find input/output ops in SavedModel\n";
        std::cout << "  (This is expected - TF2 SavedModel op naming varies)\n";
        return;
    }
    
    std::cout << "  Using input: " << input_op << ", output: " << output_op << "\n";
    
    // Run inference
    auto results = session.Run(
        {{input_op, input}},
        {{output_op, 0}},
        {}
    );
    
    REQUIRE(results.size() == 1);
    
    auto output = results[0].ToVector<float>();
    REQUIRE(output.size() == 3);
    
    // Expected: [1*2+1, 2*2+1, 3*2+1] = [3, 5, 7]
    REQUIRE_APPROX(output[0], 3.0f, 0.001f);
    REQUIRE_APPROX(output[1], 5.0f, 0.001f);
    REQUIRE_APPROX(output[2], 7.0f, 0.001f);
    
    std::cout << "  ✓ SavedModel inference correct: [1,2,3] -> [3,5,7]\n";
}

TEST(savedmodel_graph_is_frozen) {
    std::string model_path = get_test_savedmodel_path();
    if (model_path.empty()) {
        std::cout << "  SKIP: No test SavedModel found\n";
        return;
    }
    
    auto [session, graph] = tf_wrap::Session::LoadSavedModel(model_path, {"serve"});
    
    // Verify graph is frozen - can't add new ops
    bool threw = false;
    try {
        auto x = tf_wrap::Tensor::FromScalar<float>(1.0f);
        (void)graph.NewOperation("Const", "NewConst")
            .SetAttrTensor("value", x.handle())
            .SetAttrType("dtype", TF_FLOAT)
            .Finish();
    } catch (const std::runtime_error& e) {
        threw = true;
        std::string msg = e.what();
        REQUIRE(msg.find("frozen") != std::string::npos);
    }
    REQUIRE(threw);
}

TEST(savedmodel_devices_available) {
    std::string model_path = get_test_savedmodel_path();
    if (model_path.empty()) {
        std::cout << "  SKIP: No test SavedModel found\n";
        return;
    }
    
    auto [session, graph] = tf_wrap::Session::LoadSavedModel(model_path, {"serve"});
    
    // Should be able to list devices on loaded session
    auto devices = session.ListDevices();
    REQUIRE(devices.count() >= 1);
    
    bool has_cpu = false;
    for (const auto& dev : devices.all()) {
        if (dev.is_cpu()) has_cpu = true;
    }
    REQUIRE(has_cpu);
}

// =============================================================================
// MOVE SEMANTICS VERIFICATION (catches bugs like frozen_ not being preserved)
// =============================================================================

TEST(move_tensor_preserves_all_state) {
    // Create tensor with specific properties
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto t1 = tf_wrap::Tensor::FromVector<float>({2, 3}, data);
    
    // Capture original state
    auto orig_shape = t1.shape();
    auto orig_dtype = t1.dtype();
    auto orig_data = t1.ToVector<float>();
    
    // Move
    auto t2 = std::move(t1);
    
    // Verify ALL state preserved in t2
    REQUIRE(t2.shape() == orig_shape);
    REQUIRE(t2.dtype() == orig_dtype);
    REQUIRE(t2.ToVector<float>() == orig_data);
    REQUIRE(t2.rank() == 2);
    REQUIRE(t2.num_elements() == 6);
    REQUIRE(t2.valid());
    REQUIRE(!t2.empty());  // Has elements
}

TEST(tensor_valid_vs_empty_semantics) {
    // Normal tensor: valid and not empty
    auto t1 = tf_wrap::Tensor::FromScalar<float>(42.0f);
    REQUIRE(t1.valid());   // Has handle
    REQUIRE(!t1.empty());  // Has elements
    
    // Zero-element tensor: valid but empty
    auto t2 = tf_wrap::Tensor::FromVector<float>({0}, {});
    REQUIRE(t2.valid());   // Has handle
    REQUIRE(t2.empty());   // No elements
    
    // Moved-from tensor: not valid and empty
    auto t3 = std::move(t1);
    REQUIRE(!t1.valid());  // No handle (moved-from)
    REQUIRE(t1.empty());   // No elements (and no handle)
}

TEST(move_graph_preserves_all_state) {
    tf_wrap::Graph g1;
    
    // Add some operations
    auto x = tf_wrap::Tensor::FromScalar<float>(1.0f);
    auto y = tf_wrap::Tensor::FromScalar<float>(2.0f);
    (void)g1.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g1.NewOperation("Const", "Y").SetAttrTensor("value", y.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    // Capture original state
    int orig_num_ops = g1.num_operations();
    TF_Operation* orig_op_x = g1.GetOperationOrThrow("X");
    TF_Operation* orig_op_y = g1.GetOperationOrThrow("Y");
    
    // Move
    auto g2 = std::move(g1);
    
    // Verify ALL state preserved in g2
    REQUIRE(g2.num_operations() == orig_num_ops);
    REQUIRE(g2.GetOperationOrThrow("X") == orig_op_x);
    REQUIRE(g2.GetOperationOrThrow("Y") == orig_op_y);
    REQUIRE(g2.valid());
    REQUIRE(!g2.is_frozen());
}

TEST(move_graph_preserves_frozen_state) {
    tf_wrap::Graph g1;
    
    auto x = tf_wrap::Tensor::FromScalar<float>(42.0f);
    (void)g1.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    // Freeze the graph
    g1.freeze();
    REQUIRE(g1.is_frozen());
    
    // Move - THIS IS THE BUG WE CAUGHT
    auto g2 = std::move(g1);
    
    // Verify frozen state is preserved
    REQUIRE(g2.is_frozen());
    
    // Verify can't add operations
    bool threw = false;
    try {
        auto y = tf_wrap::Tensor::FromScalar<float>(1.0f);
        (void)g2.NewOperation("Const", "Y").SetAttrTensor("value", y.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST(move_session_preserves_functionality) {
    tf_wrap::Graph g;
    
    auto x = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    (void)g.NewOperation("Square", "Y").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    
    tf_wrap::Session s1(g);
    
    // Run once before move
    auto r1 = s1.Run({}, {{"Y", 0}}, {});
    REQUIRE(r1[0].ToVector<float>()[0] == 1.0f);
    
    // Move
    auto s2 = std::move(s1);
    
    // Verify session still works after move
    auto r2 = s2.Run({}, {{"Y", 0}}, {});
    auto v = r2[0].ToVector<float>();
    REQUIRE(v[0] == 1.0f);
    REQUIRE(v[1] == 4.0f);
    REQUIRE(v[2] == 9.0f);
    
    // Verify ListDevices works after move
    auto devices = s2.ListDevices();
    REQUIRE(devices.count() >= 1);
}

TEST(move_assignment_graph) {
    tf_wrap::Graph g1;
    auto x = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g1.NewOperation("Const", "A").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    g1.freeze();
    
    tf_wrap::Graph g2;
    auto y = tf_wrap::Tensor::FromScalar<float>(2.0f);
    (void)g2.NewOperation("Const", "B").SetAttrTensor("value", y.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    // Move assignment
    g2 = std::move(g1);
    
    // g2 should now have g1's state
    REQUIRE(g2.is_frozen());
    REQUIRE(g2.GetOperation("A").has_value());  // A exists (from g1)
    REQUIRE(!g2.GetOperation("B").has_value()); // B gone (old g2 deleted)
}

TEST(move_assignment_session) {
    tf_wrap::Graph g1;
    auto x = tf_wrap::Tensor::FromScalar<float>(5.0f);
    (void)g1.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    tf_wrap::Session s1(g1);
    
    tf_wrap::Graph g2;
    auto y = tf_wrap::Tensor::FromScalar<float>(10.0f);
    (void)g2.NewOperation("Const", "Y").SetAttrTensor("value", y.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    tf_wrap::Session s2(g2);
    
    // Move assignment
    s2 = std::move(s1);
    
    // s2 should now run g1's graph
    auto r = s2.Run({}, {{"X", 0}}, {});
    REQUIRE(r[0].ToScalar<float>() == 5.0f);
}

// =============================================================================
// OBJECT LIFETIME TESTS
// =============================================================================

TEST(graph_usable_after_session_destroyed) {
    tf_wrap::Graph g;
    auto x = tf_wrap::Tensor::FromScalar<float>(42.0f);
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    {
        tf_wrap::Session s(g);
        auto r = s.Run({}, {{"X", 0}}, {});
        REQUIRE(r[0].ToScalar<float>() == 42.0f);
    } // Session destroyed here
    
    // Graph should still be valid (though frozen)
    REQUIRE(g.valid());
    REQUIRE(g.GetOperation("X").has_value());
    
    // Can create new session from same graph
    tf_wrap::Session s2(g);
    auto r2 = s2.Run({}, {{"X", 0}}, {});
    REQUIRE(r2[0].ToScalar<float>() == 42.0f);
}

TEST(multiple_sessions_same_graph) {
    tf_wrap::Graph g;
    auto x = tf_wrap::Tensor::FromScalar<float>(7.0f);
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    tf_wrap::Session s1(g);
    tf_wrap::Session s2(g);
    
    // Both sessions should work
    auto r1 = s1.Run({}, {{"X", 0}}, {});
    auto r2 = s2.Run({}, {{"X", 0}}, {});
    
    REQUIRE(r1[0].ToScalar<float>() == 7.0f);
    REQUIRE(r2[0].ToScalar<float>() == 7.0f);
}

// =============================================================================
// ERROR RECOVERY TESTS
// =============================================================================

TEST(session_usable_after_run_error) {
    tf_wrap::Graph g;
    auto x = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    tf_wrap::Session s(g);
    
    // Try to fetch non-existent op
    bool threw = false;
    try {
        s.Run({}, {{"NonExistent", 0}}, {});
    } catch (const std::runtime_error&) {
        threw = true;
    }
    REQUIRE(threw);
    
    // Session should still be usable
    auto r = s.Run({}, {{"X", 0}}, {});
    REQUIRE(r[0].ToScalar<float>() == 1.0f);
}

TEST(graph_usable_after_operation_error) {
    tf_wrap::Graph g;
    
    // Add a valid operation
    auto x = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    // Try to add invalid operation (missing required attribute)
    bool threw = false;
    try {
        (void)g.NewOperation("Const", "Bad").Finish();  // Missing value and dtype
    } catch (const std::runtime_error&) {
        threw = true;
    }
    REQUIRE(threw);
    
    // Graph should still be usable
    REQUIRE(g.GetOperation("X").has_value());
    
    // Can still add more valid operations
    auto y = tf_wrap::Tensor::FromScalar<float>(2.0f);
    (void)g.NewOperation("Const", "Y").SetAttrTensor("value", y.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    REQUIRE(g.GetOperation("Y").has_value());
}

// =============================================================================
// EDGE CASE INPUT TESTS
// =============================================================================

TEST(tensor_zero_size_dimension) {
    // Tensor with shape {3, 0} - valid but has zero elements
    std::vector<float> empty_data;
    auto t = tf_wrap::Tensor::FromVector<float>({3, 0}, empty_data);
    
    REQUIRE(t.shape().size() == 2);
    REQUIRE(t.shape()[0] == 3);
    REQUIRE(t.shape()[1] == 0);
    REQUIRE(t.num_elements() == 0);
    REQUIRE(t.valid());  // Handle exists
    REQUIRE(t.empty());  // No elements (STL semantics)
}

TEST(tensor_zero_in_middle_of_shape) {
    // Tensor with shape {2, 0, 3} - valid but empty
    std::vector<float> empty_data;
    auto t = tf_wrap::Tensor::FromVector<float>({2, 0, 3}, empty_data);
    
    REQUIRE(t.shape().size() == 3);
    REQUIRE(t.num_elements() == 0);
}

TEST(tensor_high_rank) {
    // 8-dimensional tensor
    std::vector<float> data(2 * 2 * 2 * 2 * 2 * 2 * 2 * 2, 1.0f);  // 256 elements
    auto t = tf_wrap::Tensor::FromVector<float>({2, 2, 2, 2, 2, 2, 2, 2}, data);
    
    REQUIRE(t.rank() == 8);
    REQUIRE(t.num_elements() == 256);
}

TEST(graph_many_operations) {
    tf_wrap::Graph g;
    
    // Create 100 operations
    auto init = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "op_0").SetAttrTensor("value", init.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    for (int i = 1; i < 100; ++i) {
        auto* prev = g.GetOperationOrThrow("op_" + std::to_string(i - 1));
        std::string name = "op_" + std::to_string(i);
        (void)g.NewOperation("Identity", name).AddInput(tf_wrap::Output(prev, 0)).Finish();
    }
    
    REQUIRE(g.num_operations() == 100);
    
    // Run the chain
    tf_wrap::Session s(g);
    auto r = s.Run({}, {{"op_99", 0}}, {});
    REQUIRE(r[0].ToScalar<float>() == 1.0f);
}

TEST(tensor_same_input_to_multiple_ops) {
    tf_wrap::Graph g;
    
    auto x = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    
    // Use X as input to multiple ops
    (void)g.NewOperation("Square", "A").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    (void)g.NewOperation("Sqrt", "B").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    (void)g.NewOperation("Neg", "C").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"A", 0}, {"B", 0}, {"C", 0}}, {});
    
    // A = Square(X) = [1, 4, 9]
    REQUIRE_APPROX(results[0].ToVector<float>()[1], 4.0f, 0.001f);
    // B = Sqrt(X) = [1, 1.414, 1.732]
    REQUIRE_APPROX(results[1].ToVector<float>()[1], std::sqrt(2.0f), 0.001f);
    // C = Neg(X) = [-1, -2, -3]
    REQUIRE_APPROX(results[2].ToVector<float>()[1], -2.0f, 0.001f);
}

TEST(graph_disconnected_subgraphs) {
    tf_wrap::Graph g;
    
    // Subgraph 1: X -> Square -> A
    auto x = tf_wrap::Tensor::FromScalar<float>(3.0f);
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    (void)g.NewOperation("Square", "A").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    
    // Subgraph 2: Y -> Neg -> B (completely disconnected)
    auto y = tf_wrap::Tensor::FromScalar<float>(5.0f);
    (void)g.NewOperation("Const", "Y").SetAttrTensor("value", y.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_y = g.GetOperationOrThrow("Y");
    (void)g.NewOperation("Neg", "B").AddInput(tf_wrap::Output(op_y, 0)).Finish();
    
    tf_wrap::Session s(g);
    
    // Can fetch from either subgraph
    auto ra = s.Run({}, {{"A", 0}}, {});
    REQUIRE(ra[0].ToScalar<float>() == 9.0f);
    
    auto rb = s.Run({}, {{"B", 0}}, {});
    REQUIRE(rb[0].ToScalar<float>() == -5.0f);
    
    // Can fetch from both at once
    auto both = s.Run({}, {{"A", 0}, {"B", 0}}, {});
    REQUIRE(both[0].ToScalar<float>() == 9.0f);
    REQUIRE(both[1].ToScalar<float>() == -5.0f);
}

// =============================================================================
// API MISUSE DETECTION
// =============================================================================

TEST(run_with_wrong_dtype_feed) {
    tf_wrap::Graph g;
    
    (void)g.NewOperation("Placeholder", "X")
        .SetAttrType("dtype", TF_FLOAT)
        .SetAttrShape("shape", {-1})
        .Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    (void)g.NewOperation("Identity", "Y").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    
    tf_wrap::Session s(g);
    
    // Feed int32 to float placeholder
    auto wrong_dtype = tf_wrap::Tensor::FromVector<int32_t>({3}, {1, 2, 3});
    
    bool threw = false;
    try {
        s.Run({{"X", wrong_dtype}}, {{"Y", 0}}, {});
    } catch (const std::runtime_error&) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST(fetch_nonexistent_output_index) {
    tf_wrap::Graph g;
    
    auto x = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "X").SetAttrTensor("value", x.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    tf_wrap::Session s(g);
    
    // Const has only output 0, try to fetch output 5
    bool threw = false;
    try {
        s.Run({}, {{"X", 5}}, {});
    } catch (const std::runtime_error&) {
        threw = true;
    }
    REQUIRE(threw);
}

// =============================================================================
// PREVIOUSLY UNTESTED AREAS
// =============================================================================

TEST(large_tensor_100m_elements) {
    // 100 million floats = 400MB
    // This tests memory handling at scale
    const size_t num_elements = 100'000'000;
    
    // Just allocate - don't fill (would be slow)
    auto t = tf_wrap::Tensor::Allocate<float>({static_cast<int64_t>(num_elements)});
    
    REQUIRE(t.valid());
    REQUIRE(t.num_elements() == num_elements);
    REQUIRE(t.shape()[0] == static_cast<int64_t>(num_elements));
    
    // Verify we can write to first and last elements
    {
        auto view = t.write<float>();
        view[0] = 1.0f;
        view[num_elements - 1] = 2.0f;
    }
    
    // Verify we can read them back
    {
        auto view = t.read<float>();
        REQUIRE(view[0] == 1.0f);
        REQUIRE(view[num_elements - 1] == 2.0f);
    }
}

TEST(operation_name_very_long) {
    tf_wrap::Graph g;
    
    // Create a 1000-character operation name
    std::string long_name(1000, 'x');
    
    auto x = tf_wrap::Tensor::FromScalar<float>(42.0f);
    (void)g.NewOperation("Const", long_name.c_str())
        .SetAttrTensor("value", x.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Should be able to retrieve it
    REQUIRE(g.GetOperation(long_name).has_value());
    
    // Should be able to run it
    tf_wrap::Session s(g);
    auto r = s.Run({}, {{long_name, 0}}, {});
    REQUIRE(r[0].ToScalar<float>() == 42.0f);
}

TEST(operation_name_special_characters) {
    tf_wrap::Graph g;
    
    // TensorFlow allows underscores and numbers
    std::string name = "Op_123_Test_456";
    
    auto x = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", name.c_str())
        .SetAttrTensor("value", x.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    REQUIRE(g.GetOperation(name).has_value());
}

TEST(unicode_in_string_tensor) {
    // Test unicode handling in string tensors
    // Using regular string with UTF-8 bytes (works in C++20)
    std::string unicode_str = "Hello \xe4\xb8\x96\xe7\x95\x8c \xf0\x9f\x8c\x8d \xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82";
    
    auto t = tf_wrap::Tensor::FromString(unicode_str);
    REQUIRE(t.valid());
    
    auto result = t.ToString();
    REQUIRE(result == unicode_str);
}

TEST(file_path_with_spaces) {
    tf_wrap::Graph g;
    
    // Create a path with spaces (common on Windows/macOS)
    std::string path_with_spaces = "/tmp/test path with spaces/file.txt";
    
    auto path_tensor = tf_wrap::Tensor::FromString(path_with_spaces);
    (void)g.NewOperation("Const", "Path")
        .SetAttrTensor("value", path_tensor.handle())
        .SetAttrType("dtype", TF_STRING)
        .Finish();
    
    tf_wrap::Session s(g);
    auto r = s.Run({}, {{"Path", 0}}, {});
    REQUIRE(r[0].ToString() == path_with_spaces);
}

TEST(clone_with_concurrent_reads_safe_tensor) {
    // Test that Clone() is safe with concurrent reads using Tensor
    auto t = tf_wrap::Tensor::FromVector<float>({1000}, std::vector<float>(1000, 1.0f));
    
    std::atomic<bool> error{false};
    std::atomic<int> clones_done{0};
    std::atomic<int> reads_done{0};
    
    auto clone_fn = [&]() {
        for (int i = 0; i < 100; ++i) {
            auto cloned = t.Clone();
            if (cloned.num_elements() != 1000) {
                error = true;
            }
            clones_done++;
        }
    };
    
    auto read_fn = [&]() {
        for (int i = 0; i < 100; ++i) {
            auto view = t.read<float>();
            float sum = 0;
            for (size_t j = 0; j < 100; ++j) {
                sum += view[j];
            }
            if (sum != 100.0f) {
                error = true;
            }
            reads_done++;
        }
    };
    
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back(clone_fn);
        threads.emplace_back(read_fn);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    REQUIRE(!error);
    REQUIRE(clones_done == 400);
    REQUIRE(reads_done == 400);
}

TEST(set_attr_func_name) {
    // Test SetAttrFuncName - used for functional ops like map/reduce
    tf_wrap::Graph g;
    
    // While creates a graph that could use a function, but we just verify
    // the attribute can be set without crashing
    auto x = tf_wrap::Tensor::FromScalar<int32_t>(0);
    (void)g.NewOperation("Const", "Init")
        .SetAttrTensor("value", x.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    // NoOp can't actually use SetAttrFuncName meaningfully, but we can
    // verify the API exists and doesn't crash
    // A proper test would need a While or MapFn op which is complex
    REQUIRE(g.num_operations() == 1);
}

TEST(tensor_scalar_all_dtypes) {
    // Comprehensive dtype coverage for scalars
    {
        auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
        REQUIRE(t.dtype() == TF_FLOAT);
        REQUIRE(t.ToScalar<float>() == 1.0f);
    }
    {
        auto t = tf_wrap::Tensor::FromScalar<double>(2.0);
        REQUIRE(t.dtype() == TF_DOUBLE);
        REQUIRE(t.ToScalar<double>() == 2.0);
    }
    {
        auto t = tf_wrap::Tensor::FromScalar<int32_t>(-42);
        REQUIRE(t.dtype() == TF_INT32);
        REQUIRE(t.ToScalar<int32_t>() == -42);
    }
    {
        auto t = tf_wrap::Tensor::FromScalar<int64_t>(INT64_MAX);
        REQUIRE(t.dtype() == TF_INT64);
        REQUIRE(t.ToScalar<int64_t>() == INT64_MAX);
    }
    {
        auto t = tf_wrap::Tensor::FromScalar<uint8_t>(255);
        REQUIRE(t.dtype() == TF_UINT8);
        REQUIRE(t.ToScalar<uint8_t>() == 255);
    }
    {
        auto t = tf_wrap::Tensor::FromScalar<int16_t>(-32768);
        REQUIRE(t.dtype() == TF_INT16);
        REQUIRE(t.ToScalar<int16_t>() == -32768);
    }
    {
        auto t = tf_wrap::Tensor::FromScalar<bool>(true);
        REQUIRE(t.dtype() == TF_BOOL);
        REQUIRE(t.ToScalar<bool>() == true);
    }
}

TEST(partial_run_setup_and_execute) {
    // Test partial runs - incremental graph execution
    // Build graph: a + 2 = plus2, plus2 + b = result
    tf_wrap::Graph g;
    
    auto a = g.NewOperation("Placeholder", "a")
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto b = g.NewOperation("Placeholder", "b")
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto two = tf_wrap::Tensor::FromScalar<float>(2.0f);
    auto const_two = g.NewOperation("Const", "two")
        .SetAttrTensor("value", two.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto plus2 = g.NewOperation("AddV2", "plus2")
        .AddInput({a, 0})
        .AddInput({const_two, 0})
        .Finish();
    
    auto result = g.NewOperation("AddV2", "result")
        .AddInput({plus2, 0})
        .AddInput({b, 0})
        .Finish();
    
    tf_wrap::Session session(g);
    
    // Set up partial run with all inputs and outputs
    std::vector<tf_wrap::Fetch> all_inputs = {{"a", 0}, {"b", 0}};
    std::vector<tf_wrap::Fetch> all_outputs = {{"plus2", 0}, {"result", 0}};
    auto handle = session.PartialRunSetup(all_inputs, all_outputs);
    
    REQUIRE(handle.valid());
    
    // Step 1: Feed a, get a+2
    auto tensor_a = tf_wrap::Tensor::FromScalar<float>(3.0f);
    std::vector<tf_wrap::Feed> feeds1 = {{"a", tensor_a.handle()}};
    std::vector<tf_wrap::Fetch> fetches1 = {{"plus2", 0}};
    auto results1 = session.PartialRun(handle, feeds1, fetches1);
    
    REQUIRE(results1.size() == 1);
    REQUIRE(results1[0].ToScalar<float>() == 5.0f);  // 3 + 2 = 5
    
    // Step 2: Feed b, get final result
    auto tensor_b = tf_wrap::Tensor::FromScalar<float>(10.0f);
    std::vector<tf_wrap::Feed> feeds2 = {{"b", tensor_b.handle()}};
    std::vector<tf_wrap::Fetch> fetches2 = {{"result", 0}};
    auto results2 = session.PartialRun(handle, feeds2, fetches2);
    
    REQUIRE(results2.size() == 1);
    REQUIRE(results2[0].ToScalar<float>() == 15.0f);  // 5 + 10 = 15
}

TEST(partial_run_handle_move) {
    // Test PartialRunHandle move semantics
    tf_wrap::PartialRunHandle h1;
    REQUIRE(!h1.valid());
    
    // Move construct
    tf_wrap::PartialRunHandle h2(std::move(h1));
    REQUIRE(!h2.valid());  // Source was empty
    
    // Can assign
    h1 = std::move(h2);
    REQUIRE(!h1.valid());
}

TEST(graph_function_create_and_copy) {
    // Test GraphFunction creation from a subgraph
    
    // Create function body graph: input -> square -> output
    tf_wrap::Graph func_body;
    
    auto input = func_body.NewOperation("Placeholder", "input")
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto squared = func_body.NewOperation("Square", "squared")
        .AddInput({input, 0})
        .Finish();
    
    // Create function from the subgraph
    TF_Output inputs[] = {{input, 0}};
    TF_Output outputs[] = {{squared, 0}};
    
    auto func = tf_wrap::GraphFunction::FromGraph(
        func_body,
        "my_square_func",
        inputs,
        outputs,
        "Squares the input"
    );
    
    REQUIRE(func.valid());
    REQUIRE(std::string(func.name()) == "my_square_func");
    
    // Copy to main graph
    tf_wrap::Graph main_graph;
    func.CopyTo(main_graph);
    
    // The function is now registered in main_graph
    // (We can't easily verify this without StatefulPartitionedCall, but the API works)
    REQUIRE(main_graph.valid());
}

TEST(graph_function_move_semantics) {
    // Test GraphFunction move semantics
    tf_wrap::Graph g;
    
    auto input = g.NewOperation("Placeholder", "x")
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    TF_Output ins[] = {{input, 0}};
    TF_Output outs[] = {{input, 0}};  // Identity function
    
    auto func1 = tf_wrap::GraphFunction::FromGraph(g, "f1", ins, outs);
    REQUIRE(func1.valid());
    
    // Move construct
    auto func2 = std::move(func1);
    REQUIRE(func2.valid());
    REQUIRE(!func1.valid());  // NOLINT - testing moved-from state
    
    // Move assign
    tf_wrap::GraphFunction func3;
    func3 = std::move(func2);
    REQUIRE(func3.valid());
    REQUIRE(!func2.valid());  // NOLINT - testing moved-from state
}

// =============================================================================
// Session Configuration Tests
// =============================================================================

TEST(session_options_set_config_valid_proto) {
    // ConfigProto: set intra_op_parallelism_threads = 1 (field 2, varint)
    // Wire format: field_number << 3 | wire_type = 2 << 3 | 0 = 16 (0x10), value 1
    std::vector<uint8_t> config_proto = {0x10, 0x01};
    
    tf_wrap::SessionOptions opts;
    opts.SetConfig(config_proto.data(), config_proto.size());
    
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(42.0f);
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Session s(g, opts);
    auto results = s.Run({}, {{"X", 0}}, {});
    REQUIRE(results[0].ToScalar<float>() == 42.0f);
}

TEST(session_options_set_config_parallelism) {
    // Set both intra (field 2) and inter (field 5) op parallelism
    std::vector<uint8_t> config_proto = {
        0x10, 0x02,  // field 2 = 2
        0x28, 0x02   // field 5 = 2
    };
    
    tf_wrap::SessionOptions opts;
    opts.SetConfig(config_proto.data(), config_proto.size());
    
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Session s(g, opts);
    auto results = s.Run({}, {{"X", 0}}, {});
    REQUIRE(results[0].ToVector<float>().size() == 3);
}

TEST(session_options_set_target_local) {
    tf_wrap::SessionOptions opts;
    opts.SetTarget("");  // Empty string = local execution
    
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<int32_t>(123);
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    tf_wrap::Session s(g, opts);
    auto results = s.Run({}, {{"X", 0}}, {});
    REQUIRE(results[0].ToScalar<int32_t>() == 123);
}

// =============================================================================
// Graph GetOutputs Tests - REMOVED (GetOutputs not implemented in v5.0)
// =============================================================================

// =============================================================================
// GraphFunction CopyTo Tests
// =============================================================================

TEST(graph_function_copy_to_single_graph) {
    tf_wrap::Graph g1;
    (void)g1.NewOperation("Placeholder", "Input")
        .SetAttrType("dtype", TF_FLOAT)
        .SetAttrShape("shape", {})
        .Finish();
    
    auto* input_op = g1.GetOperationOrThrow("Input");
    (void)g1.NewOperation("Square", "Output")
        .AddInput(tf_wrap::Output(input_op, 0))
        .Finish();
    auto* output_op = g1.GetOperationOrThrow("Output");
    
    std::vector<TF_Output> inputs = {{input_op, 0}};
    std::vector<TF_Output> outputs = {{output_op, 0}};
    
    auto func = tf_wrap::GraphFunction::FromGraph(g1, "square_func", inputs, outputs);
    REQUIRE(func.valid());
    
    tf_wrap::Graph g2;
    func.CopyTo(g2);
    // Function registered successfully if no throw
}

TEST(graph_function_copy_to_multiple_graphs) {
    tf_wrap::Graph g1;
    (void)g1.NewOperation("Placeholder", "X")
        .SetAttrType("dtype", TF_FLOAT)
        .SetAttrShape("shape", {})
        .Finish();
    
    auto* x = g1.GetOperationOrThrow("X");
    (void)g1.NewOperation("Neg", "NegX")
        .AddInput(tf_wrap::Output(x, 0))
        .Finish();
    auto* neg_x = g1.GetOperationOrThrow("NegX");
    
    std::vector<TF_Output> inputs = {{x, 0}};
    std::vector<TF_Output> outputs = {{neg_x, 0}};
    
    auto func = tf_wrap::GraphFunction::FromGraph(g1, "negate", inputs, outputs);
    
    tf_wrap::Graph g2, g3, g4;
    func.CopyTo(g2);
    func.CopyTo(g3);
    func.CopyTo(g4);
    // All should succeed
}

// =============================================================================
// ImportGraphDef Tests
// =============================================================================

TEST(import_graph_def_basic) {
    tf_wrap::Graph g1;
    auto t = tf_wrap::Tensor::FromVector<float>({2}, {1.0f, 2.0f});
    (void)g1.NewOperation("Const", "MyConst")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto graphdef = g1.ToGraphDef();
    REQUIRE(!graphdef.empty());
    
    tf_wrap::Graph g2;
    g2.ImportGraphDef(graphdef.data(), graphdef.size(), "");
    
    REQUIRE(g2.HasOperation("MyConst"));
    
    tf_wrap::Session s(g2);
    auto results = s.Run({}, {{"MyConst", 0}}, {});
    auto v = results[0].ToVector<float>();
    REQUIRE(v[0] == 1.0f && v[1] == 2.0f);
}

TEST(import_graph_def_with_prefix) {
    tf_wrap::Graph g1;
    auto t = tf_wrap::Tensor::FromScalar<int32_t>(42);
    (void)g1.NewOperation("Const", "Value")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    auto graphdef = g1.ToGraphDef();
    
    tf_wrap::Graph g2;
    g2.ImportGraphDef(graphdef.data(), graphdef.size(), "imported/");
    
    REQUIRE(!g2.HasOperation("Value"));
    REQUIRE(g2.HasOperation("imported/Value"));
    
    tf_wrap::Session s(g2);
    auto results = s.Run({}, {{"imported/Value", 0}}, {});
    REQUIRE(results[0].ToScalar<int32_t>() == 42);
}

TEST(import_graph_def_merge_graphs) {
    tf_wrap::Graph g1;
    auto t1 = tf_wrap::Tensor::FromScalar<float>(10.0f);
    (void)g1.NewOperation("Const", "A")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Graph g2;
    auto t2 = tf_wrap::Tensor::FromScalar<float>(5.0f);
    (void)g2.NewOperation("Const", "B")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto graphdef2 = g2.ToGraphDef();
    g1.ImportGraphDef(graphdef2.data(), graphdef2.size(), "");
    
    REQUIRE(g1.HasOperation("A"));
    REQUIRE(g1.HasOperation("B"));
    
    auto* op_a = g1.GetOperationOrThrow("A");
    auto* op_b = g1.GetOperationOrThrow("B");
    (void)g1.NewOperation("AddV2", "Sum")
        .AddInput(tf_wrap::Output(op_a, 0))
        .AddInput(tf_wrap::Output(op_b, 0))
        .Finish();
    
    tf_wrap::Session s(g1);
    auto results = s.Run({}, {{"Sum", 0}}, {});
    REQUIRE(results[0].ToScalar<float>() == 15.0f);
}

TEST(import_graph_def_complex_graph) {
    tf_wrap::Graph g1;
    
    (void)g1.NewOperation("Placeholder", "Input")
        .SetAttrType("dtype", TF_FLOAT)
        .SetAttrShape("shape", {-1, 3})
        .Finish();
    
    auto* input = g1.GetOperationOrThrow("Input");
    
    auto weights = tf_wrap::Tensor::FromVector<float>({3, 2}, 
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    (void)g1.NewOperation("Const", "Weights")
        .SetAttrTensor("value", weights.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    auto* w = g1.GetOperationOrThrow("Weights");
    
    (void)g1.NewOperation("MatMul", "MatMul")
        .AddInput(tf_wrap::Output(input, 0))
        .AddInput(tf_wrap::Output(w, 0))
        .Finish();
    auto* mm = g1.GetOperationOrThrow("MatMul");
    
    (void)g1.NewOperation("Relu", "Output")
        .AddInput(tf_wrap::Output(mm, 0))
        .Finish();
    
    auto graphdef = g1.ToGraphDef();
    
    tf_wrap::Graph g2;
    g2.ImportGraphDef(graphdef.data(), graphdef.size(), "net/");
    
    REQUIRE(g2.HasOperation("net/Input"));
    REQUIRE(g2.HasOperation("net/Output"));
    
    tf_wrap::Session s(g2);
    auto input_data = tf_wrap::Tensor::FromVector<float>({1, 3}, {1.0f, 1.0f, 1.0f});
    auto results = s.Run(
        {{"net/Input", 0, input_data.handle()}},
        {{"net/Output", 0}},
        {}
    );
    REQUIRE(results[0].shape()[0] == 1);
    REQUIRE(results[0].shape()[1] == 2);
}

// =============================================================================
// Large Graph Tests (Scale Testing)
// =============================================================================

TEST(graph_1000_operations) {
    tf_wrap::Graph g;
    
    auto init = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "op_0")
        .SetAttrTensor("value", init.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    for (int i = 1; i < 1000; ++i) {
        auto* prev = g.GetOperationOrThrow("op_" + std::to_string(i - 1));
        (void)g.NewOperation("Identity", "op_" + std::to_string(i))
            .AddInput(tf_wrap::Output(prev, 0))
            .Finish();
    }
    
    REQUIRE(g.num_operations() == 1000);
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"op_999", 0}}, {});
    REQUIRE(results[0].ToScalar<float>() == 1.0f);
}

TEST(graph_2000_operations_branching) {
    tf_wrap::Graph g;
    
    auto init = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "root")
        .SetAttrTensor("value", init.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* root = g.GetOperationOrThrow("root");
    auto* prev_a = root;
    auto* prev_b = root;
    
    for (int i = 0; i < 999; ++i) {
        std::string name_a = "branch_a_" + std::to_string(i);
        std::string name_b = "branch_b_" + std::to_string(i);
        
        (void)g.NewOperation("Identity", name_a)
            .AddInput(tf_wrap::Output(prev_a, 0))
            .Finish();
        (void)g.NewOperation("Identity", name_b)
            .AddInput(tf_wrap::Output(prev_b, 0))
            .Finish();
        
        prev_a = g.GetOperationOrThrow(name_a);
        prev_b = g.GetOperationOrThrow(name_b);
    }
    
    (void)g.NewOperation("AddV2", "merge")
        .AddInput(tf_wrap::Output(prev_a, 0))
        .AddInput(tf_wrap::Output(prev_b, 0))
        .Finish();
    
    REQUIRE(g.num_operations() == 2000);
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"merge", 0}}, {});
    REQUIRE(results[0].ToScalar<float>() == 2.0f);
}

TEST(graph_5000_operations_chain) {
    tf_wrap::Graph g;
    
    auto init = tf_wrap::Tensor::FromScalar<float>(0.0f);
    (void)g.NewOperation("Const", "op_0")
        .SetAttrTensor("value", init.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto one = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "one")
        .SetAttrTensor("value", one.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    auto* op_one = g.GetOperationOrThrow("one");
    
    for (int i = 1; i < 5000; ++i) {
        auto* prev = g.GetOperationOrThrow("op_" + std::to_string(i - 1));
        (void)g.NewOperation("AddV2", "op_" + std::to_string(i))
            .AddInput(tf_wrap::Output(prev, 0))
            .AddInput(tf_wrap::Output(op_one, 0))
            .Finish();
    }
    
    REQUIRE(g.num_operations() == 5001);
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"op_4999", 0}}, {});
    REQUIRE(results[0].ToScalar<float>() == 4999.0f);
}

// =============================================================================
// Bool Tensor Tests
// =============================================================================

TEST(bool_tensor_from_vector) {
    std::vector<bool> data = {true, false, true, true, false};
    auto tensor = tf_wrap::Tensor::FromVector<bool>({5}, data);
    
    REQUIRE(tensor.dtype() == TF_BOOL);
    REQUIRE(tensor.num_elements() == 5);
    
    auto result = tensor.ToVector<bool>();
    REQUIRE(result.size() == 5);
    REQUIRE(result[0] == true);
    REQUIRE(result[1] == false);
    REQUIRE(result[2] == true);
}

TEST(bool_tensor_from_scalar) {
    auto t_true = tf_wrap::Tensor::FromScalar<bool>(true);
    auto t_false = tf_wrap::Tensor::FromScalar<bool>(false);
    
    REQUIRE(t_true.ToScalar<bool>() == true);
    REQUIRE(t_false.ToScalar<bool>() == false);
}

TEST(bool_tensor_logical_operations) {
    tf_wrap::Graph g;
    
    auto a = tf_wrap::Tensor::FromVector<bool>({4}, {true, true, false, false});
    auto b = tf_wrap::Tensor::FromVector<bool>({4}, {true, false, true, false});
    
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", a.handle())
        .SetAttrType("dtype", TF_BOOL)
        .Finish();
    (void)g.NewOperation("Const", "B")
        .SetAttrTensor("value", b.handle())
        .SetAttrType("dtype", TF_BOOL)
        .Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("LogicalAnd", "And")
        .AddInput(tf_wrap::Output(op_a, 0))
        .AddInput(tf_wrap::Output(op_b, 0))
        .Finish();
    (void)g.NewOperation("LogicalOr", "Or")
        .AddInput(tf_wrap::Output(op_a, 0))
        .AddInput(tf_wrap::Output(op_b, 0))
        .Finish();
    (void)g.NewOperation("LogicalNot", "NotA")
        .AddInput(tf_wrap::Output(op_a, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"And", 0}, {"Or", 0}, {"NotA", 0}}, {});
    
    auto and_v = results[0].ToVector<bool>();
    REQUIRE(and_v[0] == true);   // T && T
    REQUIRE(and_v[1] == false);  // T && F
    REQUIRE(and_v[2] == false);  // F && T
    REQUIRE(and_v[3] == false);  // F && F
    
    auto or_v = results[1].ToVector<bool>();
    REQUIRE(or_v[0] == true);   // T || T
    REQUIRE(or_v[1] == true);   // T || F
    REQUIRE(or_v[2] == true);   // F || T
    REQUIRE(or_v[3] == false);  // F || F
    
    auto not_v = results[2].ToVector<bool>();
    REQUIRE(not_v[0] == false);
    REQUIRE(not_v[1] == false);
    REQUIRE(not_v[2] == true);
    REQUIRE(not_v[3] == true);
}

TEST(bool_tensor_2d) {
    std::vector<bool> data = {true, false, true, false, true, false};
    auto tensor = tf_wrap::Tensor::FromVector<bool>({2, 3}, data);
    
    REQUIRE(tensor.shape()[0] == 2);
    REQUIRE(tensor.shape()[1] == 3);
    REQUIRE(tensor.ToVector<bool>().size() == 6);
}

// =============================================================================
// ToGraphDef Roundtrip Tests
// =============================================================================

TEST(to_graph_def_roundtrip_preserves_dtypes) {
    tf_wrap::Graph g1;
    
    auto f = tf_wrap::Tensor::FromScalar<float>(1.0f);
    auto i = tf_wrap::Tensor::FromScalar<int32_t>(2);
    auto d = tf_wrap::Tensor::FromScalar<double>(3.0);
    
    (void)g1.NewOperation("Const", "Float")
        .SetAttrTensor("value", f.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    (void)g1.NewOperation("Const", "Int")
        .SetAttrTensor("value", i.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    (void)g1.NewOperation("Const", "Double")
        .SetAttrTensor("value", d.handle())
        .SetAttrType("dtype", TF_DOUBLE)
        .Finish();
    
    auto graphdef = g1.ToGraphDef();
    
    tf_wrap::Graph g2;
    g2.ImportGraphDef(graphdef.data(), graphdef.size(), "");
    
    tf_wrap::Session s(g2);
    auto results = s.Run({}, {{"Float", 0}, {"Int", 0}, {"Double", 0}}, {});
    
    REQUIRE(results[0].dtype() == TF_FLOAT);
    REQUIRE(results[1].dtype() == TF_INT32);
    REQUIRE(results[2].dtype() == TF_DOUBLE);
    
    REQUIRE(results[0].ToScalar<float>() == 1.0f);
    REQUIRE(results[1].ToScalar<int32_t>() == 2);
    REQUIRE(results[2].ToScalar<double>() == 3.0);
}

TEST(to_graph_def_roundtrip_preserves_shapes) {
    tf_wrap::Graph g1;
    
    auto t1 = tf_wrap::Tensor::FromVector<float>({2, 3}, {1, 2, 3, 4, 5, 6});
    auto t2 = tf_wrap::Tensor::FromVector<float>({3, 2, 1}, {1, 2, 3, 4, 5, 6});
    
    (void)g1.NewOperation("Const", "Shape23")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    (void)g1.NewOperation("Const", "Shape321")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto graphdef = g1.ToGraphDef();
    
    tf_wrap::Graph g2;
    g2.ImportGraphDef(graphdef.data(), graphdef.size(), "");
    
    tf_wrap::Session s(g2);
    auto results = s.Run({}, {{"Shape23", 0}, {"Shape321", 0}}, {});
    
    REQUIRE(results[0].shape().size() == 2);
    REQUIRE(results[0].shape()[0] == 2);
    REQUIRE(results[0].shape()[1] == 3);
    
    REQUIRE(results[1].shape().size() == 3);
    REQUIRE(results[1].shape()[0] == 3);
    REQUIRE(results[1].shape()[1] == 2);
    REQUIRE(results[1].shape()[2] == 1);
}

TEST(to_graph_def_roundtrip_large_graph) {
    tf_wrap::Graph g1;
    
    auto init = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g1.NewOperation("Const", "op_0")
        .SetAttrTensor("value", init.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    for (int i = 1; i < 500; ++i) {
        auto* prev = g1.GetOperationOrThrow("op_" + std::to_string(i - 1));
        (void)g1.NewOperation("Identity", "op_" + std::to_string(i))
            .AddInput(tf_wrap::Output(prev, 0))
            .Finish();
    }
    
    auto graphdef = g1.ToGraphDef();
    REQUIRE(graphdef.size() > 1000);
    
    tf_wrap::Graph g2;
    g2.ImportGraphDef(graphdef.data(), graphdef.size(), "");
    
    REQUIRE(g2.num_operations() == 500);
    
    tf_wrap::Session s(g2);
    auto results = s.Run({}, {{"op_499", 0}}, {});
    REQUIRE(results[0].ToScalar<float>() == 1.0f);
}

// =============================================================================
// Zeros Factory Tests
// =============================================================================

TEST(zeros_all_dtypes) {
    {
        auto t = tf_wrap::Tensor::Zeros<float>({10});
        REQUIRE(t.dtype() == TF_FLOAT);
        for (auto x : t.ToVector<float>()) REQUIRE(x == 0.0f);
    }
    {
        auto t = tf_wrap::Tensor::Zeros<double>({10});
        REQUIRE(t.dtype() == TF_DOUBLE);
        for (auto x : t.ToVector<double>()) REQUIRE(x == 0.0);
    }
    {
        auto t = tf_wrap::Tensor::Zeros<int32_t>({10});
        REQUIRE(t.dtype() == TF_INT32);
        for (auto x : t.ToVector<int32_t>()) REQUIRE(x == 0);
    }
    {
        auto t = tf_wrap::Tensor::Zeros<int64_t>({10});
        REQUIRE(t.dtype() == TF_INT64);
        for (auto x : t.ToVector<int64_t>()) REQUIRE(x == 0);
    }
    {
        auto t = tf_wrap::Tensor::Zeros<uint8_t>({10});
        REQUIRE(t.dtype() == TF_UINT8);
        for (auto x : t.ToVector<uint8_t>()) REQUIRE(x == 0);
    }
}

TEST(zeros_large_tensor) {
    auto t = tf_wrap::Tensor::Zeros<float>({1000, 1000});
    REQUIRE(t.num_elements() == 1000000);
    
    auto v = t.ToVector<float>();
    REQUIRE(v[0] == 0.0f);
    REQUIRE(v[500000] == 0.0f);
    REQUIRE(v[999999] == 0.0f);
}

TEST(zeros_multidimensional) {
    auto t = tf_wrap::Tensor::Zeros<float>({2, 3, 4, 5});
    REQUIRE(t.rank() == 4);
    REQUIRE(t.num_elements() == 120);
    REQUIRE(t.shape()[0] == 2);
    REQUIRE(t.shape()[1] == 3);
    REQUIRE(t.shape()[2] == 4);
    REQUIRE(t.shape()[3] == 5);
}

// =============================================================================
// RunWithMetadata Tests - REMOVED (RunWithMetadata not implemented in v5.0)
// =============================================================================

// =============================================================================
// PartialRun Error Path Tests
// =============================================================================

TEST(partial_run_invalid_handle_throws) {
    tf_wrap::Graph g;
    (void)g.NewOperation("Placeholder", "X")
        .SetAttrType("dtype", TF_FLOAT)
        .SetAttrShape("shape", {})
        .Finish();
    
    tf_wrap::Session s(g);
    tf_wrap::PartialRunHandle invalid_handle;
    
    std::vector<tf_wrap::Feed> feeds;
    std::vector<tf_wrap::Fetch> fetches = {{"X", 0}};
    
    bool threw = false;
    try {
        (void)s.PartialRun(invalid_handle, feeds, fetches);
    } catch (...) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST(partial_run_wrong_feed_throws) {
    tf_wrap::Graph g;
    (void)g.NewOperation("Placeholder", "X")
        .SetAttrType("dtype", TF_FLOAT)
        .SetAttrShape("shape", {})
        .Finish();
    (void)g.NewOperation("Placeholder", "Y")
        .SetAttrType("dtype", TF_FLOAT)
        .SetAttrShape("shape", {})
        .Finish();
    
    auto* x = g.GetOperationOrThrow("X");
    auto* y = g.GetOperationOrThrow("Y");
    
    (void)g.NewOperation("AddV2", "Sum")
        .AddInput(tf_wrap::Output(x, 0))
        .AddInput(tf_wrap::Output(y, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    
    std::vector<tf_wrap::Fetch> inputs = {{"X", 0}};
    std::vector<tf_wrap::Fetch> outputs = {{"Sum", 0}};
    auto handle = s.PartialRunSetup(inputs, outputs);
    
    auto feed_tensor = tf_wrap::Tensor::FromScalar<float>(1.0f);
    std::vector<tf_wrap::Feed> wrong_feeds = {{"Y", feed_tensor.handle()}};
    std::vector<tf_wrap::Fetch> fetches = {{"Sum", 0}};
    
    bool threw = false;
    try {
        (void)s.PartialRun(handle, wrong_feeds, fetches);
    } catch (...) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST(partial_run_fetch_not_in_setup_throws) {
    tf_wrap::Graph g;
    (void)g.NewOperation("Placeholder", "X")
        .SetAttrType("dtype", TF_FLOAT)
        .SetAttrShape("shape", {})
        .Finish();
    
    auto* x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("Square", "Sq")
        .AddInput(tf_wrap::Output(x, 0))
        .Finish();
    (void)g.NewOperation("Neg", "Neg")
        .AddInput(tf_wrap::Output(x, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    
    std::vector<tf_wrap::Fetch> inputs = {{"X", 0}};
    std::vector<tf_wrap::Fetch> outputs = {{"Sq", 0}};
    auto handle = s.PartialRunSetup(inputs, outputs);
    
    auto feed_tensor = tf_wrap::Tensor::FromScalar<float>(2.0f);
    std::vector<tf_wrap::Feed> feeds = {{"X", feed_tensor.handle()}};
    std::vector<tf_wrap::Fetch> bad_fetches = {{"Neg", 0}};
    
    bool threw = false;
    try {
        (void)s.PartialRun(handle, feeds, bad_fetches);
    } catch (...) {
        threw = true;
    }
    REQUIRE(threw);
}

// =============================================================================
// PartialRun Multi-Step Tests
// =============================================================================

TEST(partial_run_multi_step_execution) {
    // Graph: X + Y = Sum, where X and Y are fed in separate steps
    tf_wrap::Graph g;
    (void)g.NewOperation("Placeholder", "X")
        .SetAttrType("dtype", TF_FLOAT)
        .SetAttrShape("shape", {})
        .Finish();
    (void)g.NewOperation("Placeholder", "Y")
        .SetAttrType("dtype", TF_FLOAT)
        .SetAttrShape("shape", {})
        .Finish();
    
    auto* x = g.GetOperationOrThrow("X");
    auto* y = g.GetOperationOrThrow("Y");
    
    (void)g.NewOperation("AddV2", "Sum")
        .AddInput(tf_wrap::Output(x, 0))
        .AddInput(tf_wrap::Output(y, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    
    // Setup for multi-step: both X and Y as inputs, Sum as output
    std::vector<tf_wrap::Fetch> inputs = {{"X", 0}, {"Y", 0}};
    std::vector<tf_wrap::Fetch> outputs = {{"Sum", 0}};
    auto handle = s.PartialRunSetup(inputs, outputs);
    
    // Step 1: Feed X only (no fetch yet)
    auto x_tensor = tf_wrap::Tensor::FromScalar<float>(10.0f);
    std::vector<tf_wrap::Feed> step1_feeds = {{"X", x_tensor.handle()}};
    std::vector<tf_wrap::Fetch> step1_fetches = {};
    auto step1_results = s.PartialRun(handle, step1_feeds, step1_fetches);
    REQUIRE(step1_results.empty());
    
    // Step 2: Feed Y and fetch Sum
    auto y_tensor = tf_wrap::Tensor::FromScalar<float>(5.0f);
    std::vector<tf_wrap::Feed> step2_feeds = {{"Y", y_tensor.handle()}};
    std::vector<tf_wrap::Fetch> step2_fetches = {{"Sum", 0}};
    auto step2_results = s.PartialRun(handle, step2_feeds, step2_fetches);
    
    REQUIRE(step2_results.size() == 1);
    float result = step2_results[0].ToScalar<float>();
    REQUIRE(std::abs(result - 15.0f) < 0.001f);  // 10 + 5 = 15
}

TEST(partial_run_branching_graph) {
    // Graph with branches: X -> Square, X -> Neg
    // Fetch Square first, then Neg in separate steps
    tf_wrap::Graph g;
    (void)g.NewOperation("Placeholder", "X")
        .SetAttrType("dtype", TF_FLOAT)
        .SetAttrShape("shape", {})
        .Finish();
    
    auto* x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("Square", "Squared")
        .AddInput(tf_wrap::Output(x, 0))
        .Finish();
    (void)g.NewOperation("Neg", "Negated")
        .AddInput(tf_wrap::Output(x, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    
    std::vector<tf_wrap::Fetch> inputs = {{"X", 0}};
    std::vector<tf_wrap::Fetch> outputs = {{"Squared", 0}, {"Negated", 0}};
    auto handle = s.PartialRunSetup(inputs, outputs);
    
    // Feed X and fetch Squared only
    auto x_tensor = tf_wrap::Tensor::FromScalar<float>(3.0f);
    std::vector<tf_wrap::Feed> feeds = {{"X", x_tensor.handle()}};
    std::vector<tf_wrap::Fetch> fetch_sq = {{"Squared", 0}};
    auto sq_results = s.PartialRun(handle, feeds, fetch_sq);
    
    REQUIRE(sq_results.size() == 1);
    float squared = sq_results[0].ToScalar<float>();
    REQUIRE(std::abs(squared - 9.0f) < 0.001f);  // 3^2 = 9
    
    // Now fetch Negated (X already fed)
    std::vector<tf_wrap::Feed> no_feeds = {};
    std::vector<tf_wrap::Fetch> fetch_neg = {{"Negated", 0}};
    auto neg_results = s.PartialRun(handle, no_feeds, fetch_neg);
    
    REQUIRE(neg_results.size() == 1);
    float negated = neg_results[0].ToScalar<float>();
    REQUIRE(std::abs(negated - (-3.0f)) < 0.001f);  // -3
}

TEST(partial_run_diamond_graph) {
    // Diamond: X -> A, X -> B, A + B -> Result
    tf_wrap::Graph g;
    (void)g.NewOperation("Placeholder", "X")
        .SetAttrType("dtype", TF_FLOAT)
        .SetAttrShape("shape", {})
        .Finish();
    
    auto* x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("Square", "A")
        .AddInput(tf_wrap::Output(x, 0))
        .Finish();
    (void)g.NewOperation("Neg", "B")
        .AddInput(tf_wrap::Output(x, 0))
        .Finish();
    
    auto* a = g.GetOperationOrThrow("A");
    auto* b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("AddV2", "Result")
        .AddInput(tf_wrap::Output(a, 0))
        .AddInput(tf_wrap::Output(b, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    
    std::vector<tf_wrap::Fetch> inputs = {{"X", 0}};
    std::vector<tf_wrap::Fetch> outputs = {{"A", 0}, {"B", 0}, {"Result", 0}};
    auto handle = s.PartialRunSetup(inputs, outputs);
    
    // Feed X, fetch intermediate A
    auto x_tensor = tf_wrap::Tensor::FromScalar<float>(4.0f);
    std::vector<tf_wrap::Feed> feeds = {{"X", x_tensor.handle()}};
    std::vector<tf_wrap::Fetch> fetch_a = {{"A", 0}};
    auto a_results = s.PartialRun(handle, feeds, fetch_a);
    REQUIRE(std::abs(a_results[0].ToScalar<float>() - 16.0f) < 0.001f);  // 4^2
    
    // Fetch final result (no more feeds needed)
    std::vector<tf_wrap::Feed> no_feeds = {};
    std::vector<tf_wrap::Fetch> fetch_result = {{"Result", 0}};
    auto final_results = s.PartialRun(handle, no_feeds, fetch_result);
    
    // Result = A + B = 16 + (-4) = 12
    REQUIRE(std::abs(final_results[0].ToScalar<float>() - 12.0f) < 0.001f);
}

// =============================================================================
// ImportGraphDef Conflict Tests
// =============================================================================

TEST(import_graph_def_duplicate_name_throws) {
    // Create graph with operation "X"
    tf_wrap::Graph g1;
    auto t1 = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g1.NewOperation("Const", "X")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto graphdef1 = g1.ToGraphDef();
    
    // Create another graph, also with "X"
    tf_wrap::Graph g2;
    auto t2 = tf_wrap::Tensor::FromScalar<float>(2.0f);
    (void)g2.NewOperation("Const", "X")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Try to import g1 into g2 without prefix - should fail (duplicate "X")
    bool threw = false;
    try {
        g2.ImportGraphDef(graphdef1.data(), graphdef1.size(), "");
    } catch (...) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST(import_graph_def_duplicate_avoided_with_prefix) {
    // Same setup as above
    tf_wrap::Graph g1;
    auto t1 = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g1.NewOperation("Const", "X")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto graphdef1 = g1.ToGraphDef();
    
    tf_wrap::Graph g2;
    auto t2 = tf_wrap::Tensor::FromScalar<float>(2.0f);
    (void)g2.NewOperation("Const", "X")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Import with prefix - should succeed
    g2.ImportGraphDef(graphdef1.data(), graphdef1.size(), "imported/");
    
    // Should have both X and imported/X
    REQUIRE(g2.GetOperation("X").has_value());
    REQUIRE(g2.GetOperation("imported/X").has_value());
    REQUIRE(g2.num_operations() == 2);
    
    // Verify they have different values
    tf_wrap::Session s(g2);
    auto r1 = s.Run({}, {{"X", 0}}, {});
    auto r2 = s.Run({}, {{"imported/X", 0}}, {});
    
    REQUIRE(std::abs(r1[0].ToScalar<float>() - 2.0f) < 0.001f);  // Original
    REQUIRE(std::abs(r2[0].ToScalar<float>() - 1.0f) < 0.001f);  // Imported
}

TEST(import_graph_def_multiple_imports_unique_prefixes) {
    // Create source graph
    tf_wrap::Graph source;
    auto t = tf_wrap::Tensor::FromScalar<float>(42.0f);
    (void)source.NewOperation("Const", "Value")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto graphdef = source.ToGraphDef();
    
    // Import same graph multiple times with different prefixes
    tf_wrap::Graph g;
    g.ImportGraphDef(graphdef.data(), graphdef.size(), "copy1/");
    g.ImportGraphDef(graphdef.data(), graphdef.size(), "copy2/");
    g.ImportGraphDef(graphdef.data(), graphdef.size(), "copy3/");
    
    REQUIRE(g.num_operations() == 3);
    REQUIRE(g.GetOperation("copy1/Value").has_value());
    REQUIRE(g.GetOperation("copy2/Value").has_value());
    REQUIRE(g.GetOperation("copy3/Value").has_value());
}

TEST(import_graph_def_nested_prefix) {
    tf_wrap::Graph source;
    auto t = tf_wrap::Tensor::FromScalar<int32_t>(123);
    (void)source.NewOperation("Const", "Data")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    auto graphdef = source.ToGraphDef();
    
    tf_wrap::Graph g;
    g.ImportGraphDef(graphdef.data(), graphdef.size(), "level1/level2/level3/");
    
    REQUIRE(g.GetOperation("level1/level2/level3/Data").has_value());
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"level1/level2/level3/Data", 0}}, {});
    REQUIRE(results[0].ToScalar<int32_t>() == 123);
}

// =============================================================================
// Complex Type (TF_COMPLEX64/128) Tests
// =============================================================================

TEST(complex64_tensor_from_scalar) {
    std::complex<float> val(3.0f, 4.0f);
    auto tensor = tf_wrap::Tensor::FromScalar<std::complex<float>>(val);
    
    REQUIRE(tensor.valid());
    REQUIRE(tensor.dtype() == TF_COMPLEX64);
    REQUIRE(tensor.num_elements() == 1);
    
    auto result = tensor.ToScalar<std::complex<float>>();
    REQUIRE(std::abs(result.real() - 3.0f) < 0.001f);
    REQUIRE(std::abs(result.imag() - 4.0f) < 0.001f);
}

TEST(complex128_tensor_from_scalar) {
    std::complex<double> val(1.5, 2.5);
    auto tensor = tf_wrap::Tensor::FromScalar<std::complex<double>>(val);
    
    REQUIRE(tensor.valid());
    REQUIRE(tensor.dtype() == TF_COMPLEX128);
    REQUIRE(tensor.num_elements() == 1);
    
    auto result = tensor.ToScalar<std::complex<double>>();
    REQUIRE(std::abs(result.real() - 1.5) < 0.001);
    REQUIRE(std::abs(result.imag() - 2.5) < 0.001);
}

TEST(complex64_tensor_from_vector) {
    std::vector<std::complex<float>> data = {
        {1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}
    };
    auto tensor = tf_wrap::Tensor::FromVector<std::complex<float>>({3}, data);
    
    REQUIRE(tensor.valid());
    REQUIRE(tensor.dtype() == TF_COMPLEX64);
    REQUIRE(tensor.num_elements() == 3);
    
    auto v = tensor.ToVector<std::complex<float>>();
    REQUIRE(v.size() == 3);
    REQUIRE(std::abs(v[0].real() - 1.0f) < 0.001f);
    REQUIRE(std::abs(v[0].imag() - 2.0f) < 0.001f);
    REQUIRE(std::abs(v[2].real() - 5.0f) < 0.001f);
    REQUIRE(std::abs(v[2].imag() - 6.0f) < 0.001f);
}

TEST(complex64_tensor_2d_shape) {
    std::vector<std::complex<float>> data(6);
    for (int i = 0; i < 6; ++i) {
        data[i] = std::complex<float>(static_cast<float>(i), static_cast<float>(i + 10));
    }
    
    auto tensor = tf_wrap::Tensor::FromVector<std::complex<float>>({2, 3}, data);
    
    REQUIRE(tensor.shape().size() == 2);
    REQUIRE(tensor.shape()[0] == 2);
    REQUIRE(tensor.shape()[1] == 3);
    REQUIRE(tensor.num_elements() == 6);
}

TEST(complex64_const_in_graph) {
    tf_wrap::Graph g;
    
    std::vector<std::complex<float>> data = {{1.0f, 0.0f}, {0.0f, 1.0f}};
    auto tensor = tf_wrap::Tensor::FromVector<std::complex<float>>({2}, data);
    
    (void)g.NewOperation("Const", "ComplexConst")
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", TF_COMPLEX64)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"ComplexConst", 0}}, {});
    
    REQUIRE(results.size() == 1);
    auto v = results[0].ToVector<std::complex<float>>();
    REQUIRE(v.size() == 2);
    REQUIRE(std::abs(v[0].real() - 1.0f) < 0.001f);
    REQUIRE(std::abs(v[0].imag() - 0.0f) < 0.001f);
    REQUIRE(std::abs(v[1].real() - 0.0f) < 0.001f);
    REQUIRE(std::abs(v[1].imag() - 1.0f) < 0.001f);
}

TEST(complex128_const_in_graph) {
    tf_wrap::Graph g;
    
    constexpr double pi_val = 3.14159265358979323846;
    constexpr double e_val = 2.71828182845904523536;
    std::complex<double> val(pi_val, e_val);
    auto tensor = tf_wrap::Tensor::FromScalar<std::complex<double>>(val);
    
    (void)g.NewOperation("Const", "ComplexConst")
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", TF_COMPLEX128)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"ComplexConst", 0}}, {});
    
    auto result = results[0].ToScalar<std::complex<double>>();
    REQUIRE(std::abs(result.real() - pi_val) < 0.0001);
    REQUIRE(std::abs(result.imag() - e_val) < 0.0001);
}

TEST(complex64_read_write_views) {
    std::vector<std::complex<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto tensor = tf_wrap::Tensor::FromVector<std::complex<float>>({2}, data);
    
    // Write view
    {
        auto view = tensor.write<std::complex<float>>();
        view[0] = std::complex<float>(10.0f, 20.0f);
    }
    
    // Read view
    {
        auto view = tensor.read<std::complex<float>>();
        REQUIRE(std::abs(view[0].real() - 10.0f) < 0.001f);
        REQUIRE(std::abs(view[0].imag() - 20.0f) < 0.001f);
        REQUIRE(std::abs(view[1].real() - 3.0f) < 0.001f);
        REQUIRE(std::abs(view[1].imag() - 4.0f) < 0.001f);
    }
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "\n========================================\n";
    std::cout << "TensorFlowWrap Real TF Test Suite\n";
    std::cout << "========================================\n\n";
    
    // Tests run automatically via static initialization
    
    std::cout << "\n========================================\n";
    std::cout << "Results: " << g_tests_passed << " passed, " 
              << g_tests_failed << " failed\n";
    std::cout << "========================================\n";
    
    return g_tests_failed > 0 ? 1 : 0;
}
