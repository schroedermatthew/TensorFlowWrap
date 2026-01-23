// tests/test_ops.cpp
// Tests for the auto-generated TensorFlow ops wrapper
//
// Run with: g++ -std=c++20 -I include -I third_party/tf_stub tests/test_ops.cpp third_party/tf_stub/tf_c_stub.cpp -o test_ops && ./test_ops

#include <tf_wrap/ops.hpp>
#include <iostream>
#include <vector>
#include <string>

// Simple test framework
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "[TEST] " << #name << "\n"; \
    try { \
        test_##name(); \
        std::cout << "  PASS\n"; \
        ++tests_passed; \
    } catch (const std::exception& e) { \
        std::cout << "  FAIL: " << e.what() << "\n"; \
        ++tests_failed; \
    } \
} while(0)

#define REQUIRE(cond) do { \
    if (!(cond)) throw std::runtime_error("Assertion failed: " #cond); \
} while(0)

// ============================================================================
// Tests
// ============================================================================

TEST(math_ops_compile) {
    tf_wrap::FastGraph graph;
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    
    auto c1 = graph.NewOperation("Const", "c1")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto c2 = graph.NewOperation("Const", "c2")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output o1{c1, 0}, o2{c2, 0};
    
    // Binary math ops
    (void)Add(graph, "add", o1, o2, TF_FLOAT);
    (void)Sub(graph, "sub", o1, o2, TF_FLOAT);
    (void)Mul(graph, "mul", o1, o2, TF_FLOAT);
    (void)Div(graph, "div", o1, o2, TF_FLOAT);
    (void)Maximum(graph, "max", o1, o2, TF_FLOAT);
    (void)Minimum(graph, "min", o1, o2, TF_FLOAT);
    (void)Pow(graph, "pow", o1, o2, TF_FLOAT);
    
    // Unary math ops
    (void)Neg(graph, "neg", o1, TF_FLOAT);
    (void)Abs(graph, "abs", o1, TF_FLOAT);
    (void)Square(graph, "square", o1, TF_FLOAT);
    (void)Sqrt(graph, "sqrt", o1, TF_FLOAT);
    (void)Exp(graph, "exp", o1, TF_FLOAT);
    (void)Log(graph, "log", o1, TF_FLOAT);
    (void)Sin(graph, "sin", o1, TF_FLOAT);
    (void)Cos(graph, "cos", o1, TF_FLOAT);
    (void)Tanh(graph, "tanh", o1, TF_FLOAT);
    (void)Sigmoid(graph, "sigmoid", o1, TF_FLOAT);
    
    REQUIRE(graph.num_operations() > 10);
}

TEST(nn_ops_compile) {
    tf_wrap::FastGraph graph;
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    
    auto c = graph.NewOperation("Const", "c")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output input{c, 0};
    
    // Activation functions
    (void)Relu(graph, "relu", input, TF_FLOAT);
    (void)Relu6(graph, "relu6", input, TF_FLOAT);
    (void)Elu(graph, "elu", input, TF_FLOAT);
    (void)Selu(graph, "selu", input, TF_FLOAT);
    (void)Softmax(graph, "softmax", input, TF_FLOAT);
    (void)LogSoftmax(graph, "logsoftmax", input, TF_FLOAT);
    (void)Softplus(graph, "softplus", input, TF_FLOAT);
    (void)Softsign(graph, "softsign", input, TF_FLOAT);
    
    REQUIRE(graph.num_operations() > 5);
}

TEST(matrix_ops_compile) {
    tf_wrap::FastGraph graph;
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    
    auto c1 = graph.NewOperation("Const", "c1")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto c2 = graph.NewOperation("Const", "c2")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output o1{c1, 0}, o2{c2, 0};
    
    (void)MatMul(graph, "matmul", o1, o2, TF_FLOAT);
    (void)BatchMatMul(graph, "batchmatmul", o1, o2, TF_FLOAT);
    
    REQUIRE(graph.num_operations() >= 4);
}

TEST(comparison_ops_compile) {
    tf_wrap::FastGraph graph;
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    
    auto c1 = graph.NewOperation("Const", "c1")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto c2 = graph.NewOperation("Const", "c2")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output o1{c1, 0}, o2{c2, 0};
    
    auto eq = Equal(graph, "eq", o1, o2, TF_FLOAT);
    auto ne = NotEqual(graph, "ne", o1, o2, TF_FLOAT);
    auto lt = Less(graph, "lt", o1, o2, TF_FLOAT);
    auto le = LessEqual(graph, "le", o1, o2, TF_FLOAT);
    auto gt = Greater(graph, "gt", o1, o2, TF_FLOAT);
    auto ge = GreaterEqual(graph, "ge", o1, o2, TF_FLOAT);
    
    // Logical ops
    (void)LogicalAnd(graph, "and", eq, ne);
    (void)LogicalOr(graph, "or", lt, gt);
    (void)LogicalNot(graph, "not", le);
    
    (void)ge;  // suppress unused warning
    
    REQUIRE(graph.num_operations() >= 10);
}

TEST(array_ops_compile) {
    tf_wrap::FastGraph graph;
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    
    auto c = graph.NewOperation("Const", "c")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output input{c, 0};
    
    (void)Identity(graph, "identity", input, TF_FLOAT);
    (void)ZerosLike(graph, "zeros", input, TF_FLOAT);
    (void)OnesLike(graph, "ones", input, TF_FLOAT);
    (void)Shape(graph, "shape", input, TF_FLOAT, TF_INT32);
    (void)Rank(graph, "rank", input, TF_FLOAT);
    (void)Size(graph, "size", input, TF_FLOAT, TF_INT32);
    
    REQUIRE(graph.num_operations() >= 6);
}

TEST(reduction_ops_compile) {
    tf_wrap::FastGraph graph;
    auto t_data = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    auto t_axis = tf_wrap::FastTensor::FromScalar<int32_t>(0);
    
    auto data = graph.NewOperation("Const", "data")
        .SetAttrTensor("value", t_data.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto axis = graph.NewOperation("Const", "axis")
        .SetAttrTensor("value", t_axis.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output data_out{data, 0}, axis_out{axis, 0};
    
    (void)Sum(graph, "sum", data_out, axis_out, TF_FLOAT, TF_INT32);
    (void)Prod(graph, "prod", data_out, axis_out, TF_FLOAT, TF_INT32);
    (void)Mean(graph, "mean", data_out, axis_out, TF_FLOAT, TF_INT32);
    (void)Max(graph, "max", data_out, axis_out, TF_FLOAT, TF_INT32);
    (void)Min(graph, "min", data_out, axis_out, TF_FLOAT, TF_INT32);
    (void)ArgMax(graph, "argmax", data_out, axis_out, TF_FLOAT, TF_INT32, TF_INT64);
    (void)ArgMin(graph, "argmin", data_out, axis_out, TF_FLOAT, TF_INT32, TF_INT64);
    
    REQUIRE(graph.num_operations() >= 8);
}

TEST(opresult_properties) {
    tf_wrap::FastGraph graph;
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    
    auto c = graph.NewOperation("Const", "my_const")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    auto result = Identity(graph, "my_identity", TF_Output{c, 0}, TF_FLOAT);
    
    REQUIRE(result.name() == "my_identity");
    REQUIRE(result.num_outputs() >= 1);
    REQUIRE(result.op() != nullptr);
    
    // Test implicit conversion to TF_Output
    TF_Output out = result;
    REQUIRE(out.oper == result.op());
    REQUIRE(out.index == 0);
}

TEST(cast_op_compile) {
    tf_wrap::FastGraph graph;
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    
    auto c = graph.NewOperation("Const", "c")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output input{c, 0};
    
    (void)Cast(graph, "cast", input, TF_FLOAT, TF_DOUBLE);
    
    REQUIRE(graph.num_operations() >= 2);
}

TEST(control_flow_ops_compile) {
    tf_wrap::FastGraph graph;
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    
    auto c = graph.NewOperation("Const", "c")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output input{c, 0};
    
    (void)StopGradient(graph, "stopgrad", input, TF_FLOAT);
    (void)NoOp(graph, "noop");
    
    REQUIRE(graph.num_operations() >= 2);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "Running ops wrapper tests...\n\n";
    
    RUN_TEST(math_ops_compile);
    RUN_TEST(nn_ops_compile);
    RUN_TEST(matrix_ops_compile);
    RUN_TEST(comparison_ops_compile);
    RUN_TEST(array_ops_compile);
    RUN_TEST(reduction_ops_compile);
    RUN_TEST(opresult_properties);
    RUN_TEST(cast_op_compile);
    RUN_TEST(control_flow_ops_compile);
    
    std::cout << "\n========================================\n";
    std::cout << "Passed: " << tests_passed << ", Failed: " << tests_failed << "\n";
    
    return tests_failed > 0 ? 1 : 0;
}
