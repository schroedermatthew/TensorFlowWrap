// tests/test_real_tf.cpp
// Comprehensive test suite for real TensorFlow C API (not stub)
// Tests operations, error handling, edge cases, and stress scenarios

#include "tf_wrap/all.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
    tf_wrap::FastGraph g;
    
    auto t = tf_wrap::FastTensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* op_x = g.GetOperationOrThrow("X");
    (void)g.NewOperation("Identity", "Y")
        .AddInput(tf_wrap::Output(op_x, 0))
        .Finish();
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"Y", 0}}, {});
    
    REQUIRE(results.size() == 1);
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 3);
    REQUIRE(v[0] == 1.0f && v[1] == 2.0f && v[2] == 3.0f);
}

TEST(add_subtract_multiply) {
    tf_wrap::FastGraph g;
    
    auto a = tf_wrap::FastTensor::FromVector<float>({2}, {10.0f, 20.0f});
    auto b = tf_wrap::FastTensor::FromVector<float>({2}, {3.0f, 4.0f});
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("AddV2", "Add").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    (void)g.NewOperation("Sub", "Sub").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    (void)g.NewOperation("Mul", "Mul").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    
    tf_wrap::FastSession s(g);
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
    tf_wrap::FastGraph g;
    
    // A = [[1, 2], [3, 4]]
    auto a = tf_wrap::FastTensor::FromVector<float>({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    // B = [[5, 6], [7, 8]]
    auto b = tf_wrap::FastTensor::FromVector<float>({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("MatMul", "C")
        .AddInput(tf_wrap::Output(op_a, 0))
        .AddInput(tf_wrap::Output(op_b, 0))
        .Finish();
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"C", 0}}, {});
    
    // C = A @ B = [[19, 22], [43, 50]]
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 4);
    REQUIRE(v[0] == 19.0f && v[1] == 22.0f && v[2] == 43.0f && v[3] == 50.0f);
}

TEST(reshape) {
    tf_wrap::FastGraph g;
    
    auto t = tf_wrap::FastTensor::FromVector<float>({6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto shape_t = tf_wrap::FastTensor::FromVector<int32_t>({2}, {2, 3});
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Shape").SetAttrTensor("value", shape_t.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    auto* op_shape = g.GetOperationOrThrow("Shape");
    
    (void)g.NewOperation("Reshape", "R")
        .AddInput(tf_wrap::Output(op_t, 0))
        .AddInput(tf_wrap::Output(op_shape, 0))
        .Finish();
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"R", 0}}, {});
    
    REQUIRE(results[0].rank() == 2);
    REQUIRE(results[0].shape()[0] == 2);
    REQUIRE(results[0].shape()[1] == 3);
}

TEST(reduce_sum) {
    tf_wrap::FastGraph g;
    
    // 2x3 matrix
    auto t = tf_wrap::FastTensor::FromVector<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto axis = tf_wrap::FastTensor::FromVector<int32_t>({1}, {1}); // reduce along axis 1
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Axis").SetAttrTensor("value", axis.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    auto* op_axis = g.GetOperationOrThrow("Axis");
    
    (void)g.NewOperation("Sum", "S")
        .AddInput(tf_wrap::Output(op_t, 0))
        .AddInput(tf_wrap::Output(op_axis, 0))
        .Finish();
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"S", 0}}, {});
    
    // Sum along axis 1: [1+2+3, 4+5+6] = [6, 15]
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 2);
    REQUIRE(v[0] == 6.0f && v[1] == 15.0f);
}

TEST(relu) {
    tf_wrap::FastGraph g;
    
    auto t = tf_wrap::FastTensor::FromVector<float>({4}, {-2.0f, -1.0f, 1.0f, 2.0f});
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    
    (void)g.NewOperation("Relu", "R")
        .AddInput(tf_wrap::Output(op_t, 0))
        .Finish();
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"R", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v[0] == 0.0f && v[1] == 0.0f && v[2] == 1.0f && v[3] == 2.0f);
}

TEST(softmax) {
    tf_wrap::FastGraph g;
    
    auto t = tf_wrap::FastTensor::FromVector<float>({1, 3}, {1.0f, 2.0f, 3.0f});
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    
    (void)g.NewOperation("Softmax", "S")
        .AddInput(tf_wrap::Output(op_t, 0))
        .Finish();
    
    tf_wrap::FastSession s(g);
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
    tf_wrap::FastGraph g;
    
    (void)g.NewOperation("Placeholder", "X")
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* op_x = g.GetOperationOrThrow("X");
    
    (void)g.NewOperation("Square", "Y")
        .AddInput(tf_wrap::Output(op_x, 0))
        .Finish();
    
    tf_wrap::FastSession s(g);
    
    // Feed different values
    auto input1 = tf_wrap::FastTensor::FromVector<float>({3}, {2.0f, 3.0f, 4.0f});
    auto results1 = s.Run({{"X", 0, input1.handle()}}, {{"Y", 0}}, {});
    auto v1 = results1[0].ToVector<float>();
    REQUIRE(v1[0] == 4.0f && v1[1] == 9.0f && v1[2] == 16.0f);
    
    auto input2 = tf_wrap::FastTensor::FromVector<float>({2}, {5.0f, 6.0f});
    auto results2 = s.Run({{"X", 0, input2.handle()}}, {{"Y", 0}}, {});
    auto v2 = results2[0].ToVector<float>();
    REQUIRE(v2[0] == 25.0f && v2[1] == 36.0f);
}

// =============================================================================
// Data Types
// =============================================================================

TEST(int32_operations) {
    tf_wrap::FastGraph g;
    
    auto a = tf_wrap::FastTensor::FromVector<int32_t>({3}, {10, 20, 30});
    auto b = tf_wrap::FastTensor::FromVector<int32_t>({3}, {1, 2, 3});
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_INT32).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("AddV2", "Sum").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"Sum", 0}}, {});
    
    auto v = results[0].ToVector<int32_t>();
    REQUIRE(v[0] == 11 && v[1] == 22 && v[2] == 33);
}

TEST(int64_operations) {
    tf_wrap::FastGraph g;
    
    auto a = tf_wrap::FastTensor::FromVector<int64_t>({2}, {1000000000000LL, 2000000000000LL});
    auto b = tf_wrap::FastTensor::FromVector<int64_t>({2}, {1LL, 2LL});
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_INT64).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_INT64).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("AddV2", "Sum").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"Sum", 0}}, {});
    
    auto v = results[0].ToVector<int64_t>();
    REQUIRE(v[0] == 1000000000001LL && v[1] == 2000000000002LL);
}

TEST(double_precision) {
    tf_wrap::FastGraph g;
    
    auto a = tf_wrap::FastTensor::FromVector<double>({2}, {1.0000000001, 2.0000000002});
    auto b = tf_wrap::FastTensor::FromVector<double>({2}, {0.0000000001, 0.0000000002});
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_DOUBLE).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_DOUBLE).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("AddV2", "Sum").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"Sum", 0}}, {});
    
    auto v = results[0].ToVector<double>();
    REQUIRE_APPROX(v[0], 1.0000000002, 1e-15);
    REQUIRE_APPROX(v[1], 2.0000000004, 1e-15);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(scalar_tensor) {
    tf_wrap::FastGraph g;
    
    auto scalar = tf_wrap::FastTensor::FromScalar<float>(42.0f);
    
    (void)g.NewOperation("Const", "S").SetAttrTensor("value", scalar.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"S", 0}}, {});
    
    REQUIRE(results[0].rank() == 0);
    REQUIRE(results[0].num_elements() == 1);
    REQUIRE(results[0].ToScalar<float>() == 42.0f);
}

TEST(empty_tensor) {
    tf_wrap::FastGraph g;
    
    auto empty = tf_wrap::FastTensor::FromVector<float>({0}, {});
    
    (void)g.NewOperation("Const", "E").SetAttrTensor("value", empty.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_e = g.GetOperationOrThrow("E");
    (void)g.NewOperation("Identity", "I").AddInput(tf_wrap::Output(op_e, 0)).Finish();
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"I", 0}}, {});
    
    REQUIRE(results[0].num_elements() == 0);
}

TEST(large_tensor) {
    tf_wrap::FastGraph g;
    
    // 1M elements
    std::vector<float> data(1000000);
    std::iota(data.begin(), data.end(), 0.0f);
    std::vector<int64_t> shape_vec = {1000, 1000};
    
    auto large = tf_wrap::FastTensor::FromVector<float>(shape_vec, data);
    
    (void)g.NewOperation("Const", "L").SetAttrTensor("value", large.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_l = g.GetOperationOrThrow("L");
    
    // Reduce to single value
    auto axis = tf_wrap::FastTensor::FromVector<int32_t>({2}, {0, 1});
    (void)g.NewOperation("Const", "Axis").SetAttrTensor("value", axis.handle()).SetAttrType("dtype", TF_INT32).Finish();
    auto* op_axis = g.GetOperationOrThrow("Axis");
    
    (void)g.NewOperation("Sum", "S")
        .AddInput(tf_wrap::Output(op_l, 0))
        .AddInput(tf_wrap::Output(op_axis, 0))
        .Finish();
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"S", 0}}, {});
    
    // Sum of 0..999999 = n*(n-1)/2 = 499999500000
    float expected = 499999500000.0f;
    float actual = results[0].ToScalar<float>();
    REQUIRE_APPROX(actual, expected, expected * 0.0001f); // 0.01% tolerance for float
}

TEST(high_rank_tensor) {
    tf_wrap::FastGraph g;
    
    // 5D tensor: 2x3x4x5x6 = 720 elements
    std::vector<float> data(720);
    std::iota(data.begin(), data.end(), 0.0f);
    std::vector<int64_t> shape_vec = {2, 3, 4, 5, 6};
    
    auto t = tf_wrap::FastTensor::FromVector<float>(shape_vec, data);
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    tf_wrap::FastSession s(g);
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
    tf_wrap::FastGraph g;
    
    bool threw = false;
    try {
        g.GetOperationOrThrow("nonexistent");
    } catch (const std::exception&) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST(dtype_mismatch_throws) {
    auto t = tf_wrap::FastTensor::FromVector<float>({2}, {1.0f, 2.0f});
    
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
    tf_wrap::FastGraph g;
    
    (void)g.NewOperation("Placeholder", "X")
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* op_x = g.GetOperationOrThrow("X");
    (void)g.NewOperation("Identity", "Y")
        .AddInput(tf_wrap::Output(op_x, 0))
        .Finish();
    
    tf_wrap::FastSession s(g);
    
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
    tf_wrap::FastGraph g;
    
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
    
    tf_wrap::FastSession s(g);
    
    // Feed 1D tensors - MatMul requires 2D
    auto a_tensor = tf_wrap::FastTensor::FromVector<float>({6}, 
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto b_tensor = tf_wrap::FastTensor::FromVector<float>({6}, 
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
    tf_wrap::FastGraph g;
    
    // Placeholder expects float
    (void)g.NewOperation("Placeholder", "X")
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* op_x = g.GetOperationOrThrow("X");
    (void)g.NewOperation("Identity", "Y")
        .AddInput(tf_wrap::Output(op_x, 0))
        .Finish();
    
    tf_wrap::FastSession s(g);
    
    // Feed int32 instead of float
    auto wrong_dtype = tf_wrap::FastTensor::FromVector<int32_t>({3}, {1, 2, 3});
    
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
    tf_wrap::FastGraph g;
    
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::FastSession s(g);
    
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
    tf_wrap::FastGraph g;
    
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::FastSession s(g);
    
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
    tf_wrap::FastGraph g;
    
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
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
    tf_wrap::FastGraph g;
    
    // Create float and int tensors
    auto f = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    auto i = tf_wrap::FastTensor::FromScalar<int32_t>(2);
    
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
    tf_wrap::FastGraph g;
    
    // A is 2x3, B is 2x3 - can't multiply (need 2x3 @ 3xN)
    auto a = tf_wrap::FastTensor::FromVector<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto b = tf_wrap::FastTensor::FromVector<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    
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
    tf_wrap::FastGraph g;
    
    // 6 elements can't reshape to 2x4=8
    auto t = tf_wrap::FastTensor::FromVector<float>({6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto bad_shape = tf_wrap::FastTensor::FromVector<int32_t>({2}, {2, 4});
    
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
    tf_wrap::FastGraph g;
    
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    
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
    tf_wrap::FastGraph g;
    
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
    tf_wrap::FastGraph g;
    
    auto a = tf_wrap::FastTensor::FromVector<float>({3}, {7.0f, 8.0f, 9.0f});
    auto b = tf_wrap::FastTensor::FromVector<float>({3}, {2.0f, 3.0f, 4.0f});
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("Div", "Div").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    (void)g.NewOperation("FloorDiv", "FloorDiv").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    
    tf_wrap::FastSession s(g);
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
    tf_wrap::FastGraph g;
    
    auto a = tf_wrap::FastTensor::FromVector<float>({3}, {7.0f, 10.0f, 2.0f});
    auto b = tf_wrap::FastTensor::FromVector<float>({3}, {3.0f, 4.0f, 8.0f});
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("Mod", "Mod").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    (void)g.NewOperation("Pow", "Pow").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    
    tf_wrap::FastSession s(g);
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
    tf_wrap::FastGraph g;
    
    const float pi = 3.14159265358979f;
    auto t = tf_wrap::FastTensor::FromVector<float>({4}, {0.0f, pi/6.0f, pi/4.0f, pi/2.0f});
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_t = g.GetOperationOrThrow("T");
    
    (void)g.NewOperation("Sin", "Sin").AddInput(tf_wrap::Output(op_t, 0)).Finish();
    (void)g.NewOperation("Cos", "Cos").AddInput(tf_wrap::Output(op_t, 0)).Finish();
    (void)g.NewOperation("Tan", "Tan").AddInput(tf_wrap::Output(op_t, 0)).Finish();
    
    tf_wrap::FastSession s(g);
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
    tf_wrap::FastGraph g;
    
    auto t = tf_wrap::FastTensor::FromVector<float>({4}, {0.0f, 1.0f, 2.0f, -1.0f});
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_t = g.GetOperationOrThrow("T");
    
    (void)g.NewOperation("Exp", "Exp").AddInput(tf_wrap::Output(op_t, 0)).Finish();
    
    // Create tensor for log (need positive values)
    auto t_log = tf_wrap::FastTensor::FromVector<float>({4}, {1.0f, 2.718281828f, 10.0f, 0.1f});
    (void)g.NewOperation("Const", "TLog").SetAttrTensor("value", t_log.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_tlog = g.GetOperationOrThrow("TLog");
    
    (void)g.NewOperation("Log", "Log").AddInput(tf_wrap::Output(op_tlog, 0)).Finish();
    
    tf_wrap::FastSession s(g);
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
    tf_wrap::FastGraph g;
    
    auto t = tf_wrap::FastTensor::FromVector<float>({4}, {4.0f, 9.0f, 16.0f, 25.0f});
    auto t_neg = tf_wrap::FastTensor::FromVector<float>({4}, {-3.0f, 5.0f, -7.0f, 0.0f});
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "TNeg").SetAttrTensor("value", t_neg.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    auto* op_tneg = g.GetOperationOrThrow("TNeg");
    
    (void)g.NewOperation("Sqrt", "Sqrt").AddInput(tf_wrap::Output(op_t, 0)).Finish();
    (void)g.NewOperation("Square", "Square").AddInput(tf_wrap::Output(op_tneg, 0)).Finish();
    (void)g.NewOperation("Abs", "Abs").AddInput(tf_wrap::Output(op_tneg, 0)).Finish();
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"Sqrt", 0}, {"Square", 0}, {"Abs", 0}}, {});
    
    auto sqrt_v = results[0].ToVector<float>();
    REQUIRE(sqrt_v[0] == 2.0f && sqrt_v[1] == 3.0f && sqrt_v[2] == 4.0f && sqrt_v[3] == 5.0f);
    
    auto square_v = results[1].ToVector<float>();
    REQUIRE(square_v[0] == 9.0f && square_v[1] == 25.0f && square_v[2] == 49.0f && square_v[3] == 0.0f);
    
    auto abs_v = results[2].ToVector<float>();
    REQUIRE(abs_v[0] == 3.0f && abs_v[1] == 5.0f && abs_v[2] == 7.0f && abs_v[3] == 0.0f);
}

TEST(value_comparison_ops) {
    tf_wrap::FastGraph g;
    
    auto a = tf_wrap::FastTensor::FromVector<float>({4}, {1.0f, 2.0f, 3.0f, 2.0f});
    auto b = tf_wrap::FastTensor::FromVector<float>({4}, {2.0f, 2.0f, 1.0f, 2.0f});
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("Less", "Less").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    (void)g.NewOperation("Greater", "Greater").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    (void)g.NewOperation("Equal", "Equal").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    
    tf_wrap::FastSession s(g);
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
    tf_wrap::FastGraph g;
    
    auto t = tf_wrap::FastTensor::FromVector<float>({6}, {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 2.0f});
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_t = g.GetOperationOrThrow("T");
    
    (void)g.NewOperation("Relu", "Relu").AddInput(tf_wrap::Output(op_t, 0)).Finish();
    (void)g.NewOperation("Relu6", "Relu6").AddInput(tf_wrap::Output(op_t, 0)).Finish();
    (void)g.NewOperation("Sigmoid", "Sigmoid").AddInput(tf_wrap::Output(op_t, 0)).Finish();
    (void)g.NewOperation("Tanh", "Tanh").AddInput(tf_wrap::Output(op_t, 0)).Finish();
    
    tf_wrap::FastSession s(g);
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
    tf_wrap::FastGraph g;
    
    // 2x3 matrix
    auto t = tf_wrap::FastTensor::FromVector<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto axis = tf_wrap::FastTensor::FromVector<int32_t>({1}, {1}); // reduce along axis 1
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Axis").SetAttrTensor("value", axis.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    auto* op_axis = g.GetOperationOrThrow("Axis");
    
    (void)g.NewOperation("Mean", "Mean").AddInput(tf_wrap::Output(op_t, 0)).AddInput(tf_wrap::Output(op_axis, 0)).Finish();
    (void)g.NewOperation("Max", "Max").AddInput(tf_wrap::Output(op_t, 0)).AddInput(tf_wrap::Output(op_axis, 0)).Finish();
    (void)g.NewOperation("Min", "Min").AddInput(tf_wrap::Output(op_t, 0)).AddInput(tf_wrap::Output(op_axis, 0)).Finish();
    
    tf_wrap::FastSession s(g);
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
    tf_wrap::FastGraph g;
    
    // 2x4 matrix
    auto t = tf_wrap::FastTensor::FromVector<float>({2, 4}, {3.0f, 1.0f, 4.0f, 1.0f, 2.0f, 7.0f, 1.0f, 8.0f});
    auto axis = tf_wrap::FastTensor::FromScalar<int32_t>(1); // reduce along axis 1
    
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
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"ArgMax", 0}, {"ArgMin", 0}}, {});
    
    auto argmax_v = results[0].ToVector<int64_t>();
    REQUIRE(argmax_v[0] == 2);  // index of 4.0 in row 0
    REQUIRE(argmax_v[1] == 3);  // index of 8.0 in row 1
    
    auto argmin_v = results[1].ToVector<int64_t>();
    REQUIRE(argmin_v[0] == 1 || argmin_v[0] == 3);  // index of 1.0 in row 0 (first occurrence)
    REQUIRE(argmin_v[1] == 2);  // index of 1.0 in row 1
}

TEST(value_transpose) {
    tf_wrap::FastGraph g;
    
    // 2x3 matrix
    auto t = tf_wrap::FastTensor::FromVector<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto perm = tf_wrap::FastTensor::FromVector<int32_t>({2}, {1, 0}); // swap dimensions
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Perm").SetAttrTensor("value", perm.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    auto* op_perm = g.GetOperationOrThrow("Perm");
    
    (void)g.NewOperation("Transpose", "Trans")
        .AddInput(tf_wrap::Output(op_t, 0))
        .AddInput(tf_wrap::Output(op_perm, 0))
        .Finish();
    
    tf_wrap::FastSession s(g);
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
    tf_wrap::FastGraph g;
    
    auto params = tf_wrap::FastTensor::FromVector<float>({4}, {10.0f, 20.0f, 30.0f, 40.0f});
    auto indices = tf_wrap::FastTensor::FromVector<int32_t>({3}, {3, 0, 2});
    
    (void)g.NewOperation("Const", "Params").SetAttrTensor("value", params.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Indices").SetAttrTensor("value", indices.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_params = g.GetOperationOrThrow("Params");
    auto* op_indices = g.GetOperationOrThrow("Indices");
    
    // Use basic Gather which doesn't need axis
    (void)g.NewOperation("Gather", "Gather")
        .AddInput(tf_wrap::Output(op_params, 0))
        .AddInput(tf_wrap::Output(op_indices, 0))
        .Finish();
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"Gather", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 3);
    REQUIRE(v[0] == 40.0f);  // params[3]
    REQUIRE(v[1] == 10.0f);  // params[0]
    REQUIRE(v[2] == 30.0f);  // params[2]
}

TEST(value_where_select) {
    tf_wrap::FastGraph g;
    
    auto cond = tf_wrap::FastTensor::FromVector<bool>({4}, {true, false, true, false});
    auto x = tf_wrap::FastTensor::FromVector<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto y = tf_wrap::FastTensor::FromVector<float>({4}, {10.0f, 20.0f, 30.0f, 40.0f});
    
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
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"Select", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v[0] == 1.0f);   // cond[0]=true, select x
    REQUIRE(v[1] == 20.0f);  // cond[1]=false, select y
    REQUIRE(v[2] == 3.0f);   // cond[2]=true, select x
    REQUIRE(v[3] == 40.0f);  // cond[3]=false, select y
}

TEST(value_concat) {
    tf_wrap::FastGraph g;
    
    auto t1 = tf_wrap::FastTensor::FromVector<float>({2}, {1.0f, 2.0f});
    auto t2 = tf_wrap::FastTensor::FromVector<float>({3}, {3.0f, 4.0f, 5.0f});
    auto axis = tf_wrap::FastTensor::FromScalar<int32_t>(0);
    
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
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"Concat", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 5);
    REQUIRE(v[0] == 1.0f && v[1] == 2.0f && v[2] == 3.0f && v[3] == 4.0f && v[4] == 5.0f);
}

TEST(value_slice) {
    tf_wrap::FastGraph g;
    
    auto t = tf_wrap::FastTensor::FromVector<float>({6}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto begin = tf_wrap::FastTensor::FromVector<int32_t>({1}, {2});
    auto size = tf_wrap::FastTensor::FromVector<int32_t>({1}, {3});
    
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
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"Slice", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 3);
    REQUIRE(v[0] == 2.0f && v[1] == 3.0f && v[2] == 4.0f);
}

TEST(value_cast_dtypes) {
    tf_wrap::FastGraph g;
    
    auto t_float = tf_wrap::FastTensor::FromVector<float>({3}, {1.5f, 2.7f, -3.9f});
    
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
    
    tf_wrap::FastSession s(g);
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
    tf_wrap::FastGraph g;
    
    auto a = tf_wrap::FastTensor::FromVector<bool>({4}, {true, true, false, false});
    auto b = tf_wrap::FastTensor::FromVector<bool>({4}, {true, false, true, false});
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_BOOL).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_BOOL).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("LogicalAnd", "And").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    (void)g.NewOperation("LogicalOr", "Or").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    (void)g.NewOperation("LogicalNot", "Not").AddInput(tf_wrap::Output(op_a, 0)).Finish();
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"And", 0}, {"Or", 0}, {"Not", 0}}, {});
    
    auto and_v = results[0].ToVector<bool>();
    REQUIRE(and_v[0] == true && and_v[1] == false && and_v[2] == false && and_v[3] == false);
    
    auto or_v = results[1].ToVector<bool>();
    REQUIRE(or_v[0] == true && or_v[1] == true && or_v[2] == true && or_v[3] == false);
    
    auto not_v = results[2].ToVector<bool>();
    REQUIRE(not_v[0] == false && not_v[1] == false && not_v[2] == true && not_v[3] == true);
}

TEST(value_maximum_minimum) {
    tf_wrap::FastGraph g;
    
    auto a = tf_wrap::FastTensor::FromVector<float>({4}, {1.0f, 5.0f, 3.0f, 8.0f});
    auto b = tf_wrap::FastTensor::FromVector<float>({4}, {2.0f, 4.0f, 6.0f, 7.0f});
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("Maximum", "Max").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    (void)g.NewOperation("Minimum", "Min").AddInput(tf_wrap::Output(op_a, 0)).AddInput(tf_wrap::Output(op_b, 0)).Finish();
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"Max", 0}, {"Min", 0}}, {});
    
    auto max_v = results[0].ToVector<float>();
    REQUIRE(max_v[0] == 2.0f && max_v[1] == 5.0f && max_v[2] == 6.0f && max_v[3] == 8.0f);
    
    auto min_v = results[1].ToVector<float>();
    REQUIRE(min_v[0] == 1.0f && min_v[1] == 4.0f && min_v[2] == 3.0f && min_v[3] == 7.0f);
}

TEST(value_fill_ones_zeros_like) {
    tf_wrap::FastGraph g;
    
    auto dims = tf_wrap::FastTensor::FromVector<int32_t>({2}, {2, 3});
    auto val = tf_wrap::FastTensor::FromScalar<float>(5.0f);
    
    (void)g.NewOperation("Const", "Dims").SetAttrTensor("value", dims.handle()).SetAttrType("dtype", TF_INT32).Finish();
    (void)g.NewOperation("Const", "Val").SetAttrTensor("value", val.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_dims = g.GetOperationOrThrow("Dims");
    auto* op_val = g.GetOperationOrThrow("Val");
    
    (void)g.NewOperation("Fill", "Fill")
        .AddInput(tf_wrap::Output(op_dims, 0))
        .AddInput(tf_wrap::Output(op_val, 0))
        .Finish();
    
    auto template_t = tf_wrap::FastTensor::FromVector<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    (void)g.NewOperation("Const", "Template").SetAttrTensor("value", template_t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_template = g.GetOperationOrThrow("Template");
    
    (void)g.NewOperation("OnesLike", "Ones").AddInput(tf_wrap::Output(op_template, 0)).Finish();
    (void)g.NewOperation("ZerosLike", "Zeros").AddInput(tf_wrap::Output(op_template, 0)).Finish();
    
    tf_wrap::FastSession s(g);
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
    tf_wrap::FastGraph g;
    
    auto t = tf_wrap::FastTensor::FromVector<float>({2}, {1.0f, 2.0f});
    auto multiples = tf_wrap::FastTensor::FromVector<int32_t>({1}, {3});
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Multiples").SetAttrTensor("value", multiples.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    auto* op_multiples = g.GetOperationOrThrow("Multiples");
    
    (void)g.NewOperation("Tile", "Tile")
        .AddInput(tf_wrap::Output(op_t, 0))
        .AddInput(tf_wrap::Output(op_multiples, 0))
        .Finish();
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"Tile", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 6);
    REQUIRE(v[0] == 1.0f && v[1] == 2.0f);
    REQUIRE(v[2] == 1.0f && v[3] == 2.0f);
    REQUIRE(v[4] == 1.0f && v[5] == 2.0f);
}

TEST(value_range) {
    tf_wrap::FastGraph g;
    
    auto start = tf_wrap::FastTensor::FromScalar<float>(0.0f);
    auto limit = tf_wrap::FastTensor::FromScalar<float>(5.0f);
    auto delta = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    
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
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"Range", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 5);
    REQUIRE(v[0] == 0.0f && v[1] == 1.0f && v[2] == 2.0f && v[3] == 3.0f && v[4] == 4.0f);
}

TEST(value_reduction_prod) {
    tf_wrap::FastGraph g;
    
    auto t = tf_wrap::FastTensor::FromVector<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto axis = tf_wrap::FastTensor::FromVector<int32_t>({1}, {1}); // reduce along axis 1
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Axis").SetAttrTensor("value", axis.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    auto* op_axis = g.GetOperationOrThrow("Axis");
    
    (void)g.NewOperation("Prod", "Prod")
        .AddInput(tf_wrap::Output(op_t, 0))
        .AddInput(tf_wrap::Output(op_axis, 0))
        .Finish();
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"Prod", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 2);
    REQUIRE(v[0] == 6.0f);    // 1*2*3
    REQUIRE(v[1] == 120.0f);  // 4*5*6
}

TEST(value_reduction_any_all) {
    tf_wrap::FastGraph g;
    
    auto t = tf_wrap::FastTensor::FromVector<bool>({2, 3}, {true, false, true, true, true, true});
    auto axis = tf_wrap::FastTensor::FromVector<int32_t>({1}, {1}); // reduce along axis 1
    
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
    
    tf_wrap::FastSession s(g);
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
    tf_wrap::FastGraph g;
    
    // Batch of 2 matrices: each is 2x2
    // A[0] = [[1,2],[3,4]], A[1] = [[5,6],[7,8]]
    auto a = tf_wrap::FastTensor::FromVector<float>({2, 2, 2}, 
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    // B[0] = [[1,0],[0,1]] (identity), B[1] = [[2,0],[0,2]] (scale by 2)
    auto b = tf_wrap::FastTensor::FromVector<float>({2, 2, 2}, 
        {1.0f, 0.0f, 0.0f, 1.0f, 2.0f, 0.0f, 0.0f, 2.0f});
    
    (void)g.NewOperation("Const", "A").SetAttrTensor("value", a.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "B").SetAttrTensor("value", b.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_a = g.GetOperationOrThrow("A");
    auto* op_b = g.GetOperationOrThrow("B");
    
    (void)g.NewOperation("BatchMatMul", "BMM")
        .AddInput(tf_wrap::Output(op_a, 0))
        .AddInput(tf_wrap::Output(op_b, 0))
        .Finish();
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"BMM", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 8);
    // C[0] = A[0] @ B[0] = A[0] @ I = A[0] = [[1,2],[3,4]]
    REQUIRE(v[0] == 1.0f && v[1] == 2.0f && v[2] == 3.0f && v[3] == 4.0f);
    // C[1] = A[1] @ B[1] = [[5,6],[7,8]] @ [[2,0],[0,2]] = [[10,12],[14,16]]
    REQUIRE(v[4] == 10.0f && v[5] == 12.0f && v[6] == 14.0f && v[7] == 16.0f);
}

TEST(value_conv2d_simple) {
    tf_wrap::FastGraph g;
    
    // Input: NHWC format - batch=1, height=3, width=3, channels=1
    // Simple 3x3 input with values 1-9
    auto input = tf_wrap::FastTensor::FromVector<float>({1, 3, 3, 1}, 
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    
    // Filter: HWIO format - height=2, width=2, in_channels=1, out_channels=1
    // Simple 2x2 filter of all 1s (sum filter)
    auto filter = tf_wrap::FastTensor::FromVector<float>({2, 2, 1, 1}, 
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
    
    tf_wrap::FastSession s(g);
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
    tf_wrap::FastGraph g;
    
    // Input: NHWC format - batch=1, height=4, width=4, channels=1
    auto input = tf_wrap::FastTensor::FromVector<float>({1, 4, 4, 1}, 
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
    
    tf_wrap::FastSession s(g);
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
    tf_wrap::FastGraph g;
    
    // Input: NHWC format - batch=1, height=4, width=4, channels=1
    auto input = tf_wrap::FastTensor::FromVector<float>({1, 4, 4, 1}, 
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
    
    tf_wrap::FastSession s(g);
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
    tf_wrap::FastGraph g;
    
    // Logits: 2 samples, 3 classes
    auto logits = tf_wrap::FastTensor::FromVector<float>({2, 3}, 
        {1.0f, 2.0f, 3.0f,   // sample 0: class 2 has highest logit
         3.0f, 2.0f, 1.0f}); // sample 1: class 0 has highest logit
    
    // Labels: one-hot encoded
    auto labels = tf_wrap::FastTensor::FromVector<float>({2, 3}, 
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
    
    tf_wrap::FastSession s(g);
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
    tf_wrap::FastGraph g;
    
    // Logits: 2 samples, 3 classes
    auto logits = tf_wrap::FastTensor::FromVector<float>({2, 3}, 
        {1.0f, 2.0f, 3.0f,   // sample 0
         3.0f, 2.0f, 1.0f}); // sample 1
    
    // Labels: sparse (class indices)
    auto labels = tf_wrap::FastTensor::FromVector<int32_t>({2}, {2, 0}); // class 2, class 0
    
    (void)g.NewOperation("Const", "Logits").SetAttrTensor("value", logits.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Labels").SetAttrTensor("value", labels.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_logits = g.GetOperationOrThrow("Logits");
    auto* op_labels = g.GetOperationOrThrow("Labels");
    
    (void)g.NewOperation("SparseSoftmaxCrossEntropyWithLogits", "SparseXEnt")
        .AddInput(tf_wrap::Output(op_logits, 0))
        .AddInput(tf_wrap::Output(op_labels, 0))
        .Finish();
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"SparseXEnt", 0}}, {});  // output 0 is loss
    
    auto loss = results[0].ToVector<float>();
    REQUIRE(loss.size() == 2);
    
    // Same expected values as dense version
    REQUIRE(loss[0] > 0.3f && loss[0] < 0.5f);
    REQUIRE(loss[1] > 0.3f && loss[1] < 0.5f);
}

TEST(value_biasadd) {
    tf_wrap::FastGraph g;
    
    // Value: batch=2, features=3
    auto value = tf_wrap::FastTensor::FromVector<float>({2, 3}, 
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto bias = tf_wrap::FastTensor::FromVector<float>({3}, {10.0f, 20.0f, 30.0f});
    
    (void)g.NewOperation("Const", "Value").SetAttrTensor("value", value.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Bias").SetAttrTensor("value", bias.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    
    auto* op_value = g.GetOperationOrThrow("Value");
    auto* op_bias = g.GetOperationOrThrow("Bias");
    
    (void)g.NewOperation("BiasAdd", "BiasAdd")
        .AddInput(tf_wrap::Output(op_value, 0))
        .AddInput(tf_wrap::Output(op_bias, 0))
        .Finish();
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"BiasAdd", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    // Row 0: [1+10, 2+20, 3+30] = [11, 22, 33]
    // Row 1: [4+10, 5+20, 6+30] = [14, 25, 36]
    REQUIRE(v[0] == 11.0f && v[1] == 22.0f && v[2] == 33.0f);
    REQUIRE(v[3] == 14.0f && v[4] == 25.0f && v[5] == 36.0f);
}

TEST(value_leaky_relu) {
    tf_wrap::FastGraph g;
    
    auto t = tf_wrap::FastTensor::FromVector<float>({4}, {-2.0f, -1.0f, 1.0f, 2.0f});
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_t = g.GetOperationOrThrow("T");
    
    (void)g.NewOperation("LeakyRelu", "LeakyRelu")
        .AddInput(tf_wrap::Output(op_t, 0))
        .SetAttrFloat("alpha", 0.1f)
        .Finish();
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"LeakyRelu", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    // LeakyReLU: x if x > 0, alpha*x otherwise
    REQUIRE_APPROX(v[0], -0.2f, 0.0001f);  // -2 * 0.1
    REQUIRE_APPROX(v[1], -0.1f, 0.0001f);  // -1 * 0.1
    REQUIRE(v[2] == 1.0f);
    REQUIRE(v[3] == 2.0f);
}

TEST(value_elu_selu) {
    tf_wrap::FastGraph g;
    
    auto t = tf_wrap::FastTensor::FromVector<float>({4}, {-1.0f, 0.0f, 1.0f, 2.0f});
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_t = g.GetOperationOrThrow("T");
    
    (void)g.NewOperation("Elu", "Elu").AddInput(tf_wrap::Output(op_t, 0)).Finish();
    (void)g.NewOperation("Selu", "Selu").AddInput(tf_wrap::Output(op_t, 0)).Finish();
    
    tf_wrap::FastSession s(g);
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
    tf_wrap::FastGraph g;
    
    auto t = tf_wrap::FastTensor::FromVector<float>({5}, {-5.0f, 0.0f, 5.0f, 10.0f, 15.0f});
    auto clip_min = tf_wrap::FastTensor::FromScalar<float>(0.0f);
    auto clip_max = tf_wrap::FastTensor::FromScalar<float>(10.0f);
    
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
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"Clip", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v[0] == 0.0f);   // -5 clipped to 0
    REQUIRE(v[1] == 0.0f);   // 0 unchanged
    REQUIRE(v[2] == 5.0f);   // 5 unchanged
    REQUIRE(v[3] == 10.0f);  // 10 unchanged
    REQUIRE(v[4] == 10.0f);  // 15 clipped to 10
}

TEST(value_pad) {
    tf_wrap::FastGraph g;
    
    auto t = tf_wrap::FastTensor::FromVector<float>({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto paddings = tf_wrap::FastTensor::FromVector<int32_t>({2, 2}, {1, 1, 1, 1}); // pad 1 on each side
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Paddings").SetAttrTensor("value", paddings.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    auto* op_paddings = g.GetOperationOrThrow("Paddings");
    
    (void)g.NewOperation("Pad", "Pad")
        .AddInput(tf_wrap::Output(op_t, 0))
        .AddInput(tf_wrap::Output(op_paddings, 0))
        .Finish();
    
    tf_wrap::FastSession s(g);
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
    tf_wrap::FastGraph g;
    
    auto t = tf_wrap::FastTensor::FromVector<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto axis = tf_wrap::FastTensor::FromVector<int32_t>({1}, {1}); // reverse along axis 1
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Axis").SetAttrTensor("value", axis.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    auto* op_axis = g.GetOperationOrThrow("Axis");
    
    (void)g.NewOperation("ReverseV2", "Reverse")
        .AddInput(tf_wrap::Output(op_t, 0))
        .AddInput(tf_wrap::Output(op_axis, 0))
        .Finish();
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"Reverse", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    // Reverse each row: [[1,2,3],[4,5,6]] -> [[3,2,1],[6,5,4]]
    REQUIRE(v[0] == 3.0f && v[1] == 2.0f && v[2] == 1.0f);
    REQUIRE(v[3] == 6.0f && v[4] == 5.0f && v[5] == 4.0f);
}

TEST(value_stack_unstack) {
    tf_wrap::FastGraph g;
    
    auto t1 = tf_wrap::FastTensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    auto t2 = tf_wrap::FastTensor::FromVector<float>({3}, {4.0f, 5.0f, 6.0f});
    
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
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"Stack", 0}}, {});
    
    // Stacked along axis 0: shape [2, 3]
    REQUIRE(results[0].shape()[0] == 2);
    REQUIRE(results[0].shape()[1] == 3);
    
    auto v = results[0].ToVector<float>();
    REQUIRE(v[0] == 1.0f && v[1] == 2.0f && v[2] == 3.0f);
    REQUIRE(v[3] == 4.0f && v[4] == 5.0f && v[5] == 6.0f);
}

TEST(value_squeeze_expanddims) {
    tf_wrap::FastGraph g;
    
    auto t = tf_wrap::FastTensor::FromVector<float>({1, 3, 1}, {1.0f, 2.0f, 3.0f});
    auto axis = tf_wrap::FastTensor::FromScalar<int32_t>(1);
    
    (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    (void)g.NewOperation("Const", "Axis").SetAttrTensor("value", axis.handle()).SetAttrType("dtype", TF_INT32).Finish();
    
    auto* op_t = g.GetOperationOrThrow("T");
    auto* op_axis = g.GetOperationOrThrow("Axis");
    
    // Squeeze removes dims of size 1
    (void)g.NewOperation("Squeeze", "Squeeze")
        .AddInput(tf_wrap::Output(op_t, 0))
        .Finish();
    
    // ExpandDims adds a dim at axis
    auto t2 = tf_wrap::FastTensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    (void)g.NewOperation("Const", "T2").SetAttrTensor("value", t2.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_t2 = g.GetOperationOrThrow("T2");
    
    (void)g.NewOperation("ExpandDims", "Expand")
        .AddInput(tf_wrap::Output(op_t2, 0))
        .AddInput(tf_wrap::Output(op_axis, 0))
        .Finish();
    
    tf_wrap::FastSession s(g);
    auto results = s.Run({}, {{"Squeeze", 0}, {"Expand", 0}}, {});
    
    // Squeeze [1,3,1] -> [3]
    REQUIRE(results[0].rank() == 1);
    REQUIRE(results[0].shape()[0] == 3);
    
    // ExpandDims [3] with axis=1 -> [3,1]
    REQUIRE(results[1].rank() == 2);
    REQUIRE(results[1].shape()[0] == 3);
    REQUIRE(results[1].shape()[1] == 1);
}

TEST(toscalar_multielement_throws) {
    auto t = tf_wrap::FastTensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    
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
        auto t = tf_wrap::FastTensor::FromVector<float>(shape, data);
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
        tf_wrap::FastGraph g;
        std::vector<int64_t> shape_vec = {10};
        std::vector<float> data_vec(10, 1.0f);
        auto t = tf_wrap::FastTensor::FromVector<float>(shape_vec, data_vec);
        (void)g.NewOperation("Const", "C").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
    }
    
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "    (100 graphs in " << ms << "ms)\n";
}

TEST(rapid_session_runs) {
    tf_wrap::FastGraph g;
    
    (void)g.NewOperation("Placeholder", "X").SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    (void)g.NewOperation("Square", "Y").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    
    tf_wrap::FastSession s(g);
    
    auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < 1000; ++i) {
        auto input = tf_wrap::FastTensor::FromScalar<float>(static_cast<float>(i));
        auto results = s.Run({{"X", 0, input.handle()}}, {{"Y", 0}}, {});
        (void)results;
    }
    
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "    (1000 runs in " << ms << "ms)\n";
}

TEST(concurrent_sessions) {
    // Create a shared graph
    tf_wrap::FastGraph g;
    
    (void)g.NewOperation("Placeholder", "X").SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    (void)g.NewOperation("Square", "Y").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    
    std::atomic<int> total_runs{0};
    std::atomic<bool> error{false};
    
    auto worker = [&](int thread_id) {
        try {
            tf_wrap::FastSession s(g);
            
            for (int i = 0; i < 100; ++i) {
                auto input = tf_wrap::FastTensor::FromScalar<float>(static_cast<float>(thread_id * 100 + i));
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
    
    tf_wrap::FastGraph g;
    
    (void)g.NewOperation("Placeholder", "X").SetAttrType("dtype", TF_FLOAT).Finish();
    auto* op_x = g.GetOperationOrThrow("X");
    (void)g.NewOperation("Square", "Y").AddInput(tf_wrap::Output(op_x, 0)).Finish();
    
    tf_wrap::FastSession s(g);
    
    auto start = std::chrono::steady_clock::now();
    auto end_time = start + std::chrono::seconds(30);
    
    int iterations = 0;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
    
    while (std::chrono::steady_clock::now() < end_time) {
        float val = dist(rng);
        auto input = tf_wrap::FastTensor::FromScalar<float>(val);
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
        auto t = tf_wrap::FastTensor::FromVector<float>(shape, data);
        REQUIRE(t.rank() == rank);
        REQUIRE(t.num_elements() == static_cast<std::size_t>(total));
        
        // Roundtrip through graph
        tf_wrap::FastGraph g;
        (void)g.NewOperation("Const", "T").SetAttrTensor("value", t.handle()).SetAttrType("dtype", TF_FLOAT).Finish();
        
        tf_wrap::FastSession s(g);
        auto results = s.Run({}, {{"T", 0}}, {});
        
        auto out = results[0].ToVector<float>();
        REQUIRE(out.size() == data.size());
        
        for (size_t i = 0; i < data.size(); ++i) {
            REQUIRE_APPROX(out[i], data[i], 0.0001f);
        }
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
