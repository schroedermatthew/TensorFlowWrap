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
