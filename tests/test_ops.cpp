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
    
    // Scalar for element-wise ops
    auto t_scalar = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    auto c_scalar = graph.NewOperation("Const", "c_scalar")
        .SetAttrTensor("value", t_scalar.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Rank-1 tensor for softmax (requires at least rank 1)
    auto t_vec = tf_wrap::FastTensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    auto c_vec = graph.NewOperation("Const", "c_vec")
        .SetAttrTensor("value", t_vec.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output scalar_input{c_scalar, 0};
    TF_Output vec_input{c_vec, 0};
    
    // Element-wise activation functions (work with scalars)
    (void)Relu(graph, "relu", scalar_input, TF_FLOAT);
    (void)Relu6(graph, "relu6", scalar_input, TF_FLOAT);
    (void)Elu(graph, "elu", scalar_input, TF_FLOAT);
    (void)Selu(graph, "selu", scalar_input, TF_FLOAT);
    (void)Softplus(graph, "softplus", scalar_input, TF_FLOAT);
    (void)Softsign(graph, "softsign", scalar_input, TF_FLOAT);
    
    // Softmax ops require at least rank 1
    (void)Softmax(graph, "softmax", vec_input, TF_FLOAT);
    (void)LogSoftmax(graph, "logsoftmax", vec_input, TF_FLOAT);
    
    REQUIRE(graph.num_operations() > 5);
}

TEST(matrix_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // MatMul requires rank-2 tensors (matrices)
    // Create 2x2 matrices
    auto t1 = tf_wrap::FastTensor::FromVector<float>({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto t2 = tf_wrap::FastTensor::FromVector<float>({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
    
    auto c1 = graph.NewOperation("Const", "c1")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto c2 = graph.NewOperation("Const", "c2")
        .SetAttrTensor("value", t2.handle())
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
    
    // Reduction ops need tensors with valid axes
    // Use a rank-1 tensor so axis 0 is valid
    auto t_data = tf_wrap::FastTensor::FromVector<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
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
    (void)PreventGradient(graph, "preventgrad", input, TF_FLOAT);
    
    REQUIRE(graph.num_operations() >= 3);
}

TEST(trig_ops_compile) {
    tf_wrap::FastGraph graph;
    auto t = tf_wrap::FastTensor::FromScalar<float>(0.5f);
    
    auto c = graph.NewOperation("Const", "c")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output input{c, 0};
    
    (void)Acos(graph, "acos", input, TF_FLOAT);
    (void)Asin(graph, "asin", input, TF_FLOAT);
    (void)Atan(graph, "atan", input, TF_FLOAT);
    (void)Cosh(graph, "cosh", input, TF_FLOAT);
    (void)Sinh(graph, "sinh", input, TF_FLOAT);
    (void)Tan(graph, "tan", input, TF_FLOAT);
    
    REQUIRE(graph.num_operations() >= 6);
}

TEST(more_math_ops_compile) {
    tf_wrap::FastGraph graph;
    auto t = tf_wrap::FastTensor::FromScalar<float>(2.5f);
    
    auto c = graph.NewOperation("Const", "c")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output input{c, 0};
    
    (void)Ceil(graph, "ceil", input, TF_FLOAT);
    (void)Floor(graph, "floor", input, TF_FLOAT);
    (void)Round(graph, "round", input, TF_FLOAT);
    (void)Rint(graph, "rint", input, TF_FLOAT);
    (void)Sign(graph, "sign", input, TF_FLOAT);
    (void)Reciprocal(graph, "reciprocal", input, TF_FLOAT);
    (void)Rsqrt(graph, "rsqrt", input, TF_FLOAT);
    (void)Expm1(graph, "expm1", input, TF_FLOAT);
    (void)Log1p(graph, "log1p", input, TF_FLOAT);
    
    REQUIRE(graph.num_operations() >= 9);
}

TEST(more_binary_ops_compile) {
    tf_wrap::FastGraph graph;
    auto t1 = tf_wrap::FastTensor::FromScalar<float>(10.0f);
    auto t2 = tf_wrap::FastTensor::FromScalar<float>(3.0f);
    
    auto c1 = graph.NewOperation("Const", "c1")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto c2 = graph.NewOperation("Const", "c2")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output o1{c1, 0}, o2{c2, 0};
    
    (void)AddV2(graph, "addv2", o1, o2, TF_FLOAT);
    (void)RealDiv(graph, "realdiv", o1, o2, TF_FLOAT);
    (void)FloorDiv(graph, "floordiv", o1, o2, TF_FLOAT);
    (void)Mod(graph, "mod", o1, o2, TF_FLOAT);
    
    REQUIRE(graph.num_operations() >= 6);
}

TEST(array_manipulation_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // Create a 2x3 tensor
    auto t = tf_wrap::FastTensor::FromVector<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto c = graph.NewOperation("Const", "c")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Shape for reshape: [3, 2]
    auto shape_t = tf_wrap::FastTensor::FromVector<int32_t>({2}, {3, 2});
    auto shape = graph.NewOperation("Const", "shape")
        .SetAttrTensor("value", shape_t.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    // Axis for expand_dims
    auto axis_t = tf_wrap::FastTensor::FromScalar<int32_t>(0);
    auto axis = graph.NewOperation("Const", "axis")
        .SetAttrTensor("value", axis_t.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output data{c, 0}, shape_out{shape, 0}, axis_out{axis, 0};
    
    (void)Reshape(graph, "reshape", data, shape_out, TF_FLOAT, TF_INT32);
    (void)ExpandDims(graph, "expand", data, axis_out, TF_FLOAT, TF_INT32);
    (void)Squeeze(graph, "squeeze", data, TF_FLOAT);
    
    REQUIRE(graph.num_operations() >= 6);
}

TEST(transpose_tile_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // Create a 2x3 tensor
    auto t = tf_wrap::FastTensor::FromVector<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto c = graph.NewOperation("Const", "c")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Perm for transpose: [1, 0]
    auto perm_t = tf_wrap::FastTensor::FromVector<int32_t>({2}, {1, 0});
    auto perm = graph.NewOperation("Const", "perm")
        .SetAttrTensor("value", perm_t.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    // Multiples for tile: [2, 1]
    auto mult_t = tf_wrap::FastTensor::FromVector<int32_t>({2}, {2, 1});
    auto mult = graph.NewOperation("Const", "mult")
        .SetAttrTensor("value", mult_t.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output data{c, 0}, perm_out{perm, 0}, mult_out{mult, 0};
    
    (void)Transpose(graph, "transpose", data, perm_out, TF_FLOAT, TF_INT32);
    (void)Tile(graph, "tile", data, mult_out, TF_FLOAT, TF_INT32);
    
    REQUIRE(graph.num_operations() >= 5);
}

TEST(slice_gather_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // Create a 1D tensor [0, 1, 2, 3, 4]
    auto t = tf_wrap::FastTensor::FromVector<float>({5}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f});
    auto c = graph.NewOperation("Const", "c")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Begin for slice: [1]
    auto begin_t = tf_wrap::FastTensor::FromVector<int32_t>({1}, {1});
    auto begin = graph.NewOperation("Const", "begin")
        .SetAttrTensor("value", begin_t.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    // Size for slice: [3]
    auto size_t_tensor = tf_wrap::FastTensor::FromVector<int32_t>({1}, {3});
    auto size_op = graph.NewOperation("Const", "size")
        .SetAttrTensor("value", size_t_tensor.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    // Indices for gather: [0, 2, 4]
    auto indices_t = tf_wrap::FastTensor::FromVector<int32_t>({3}, {0, 2, 4});
    auto indices = graph.NewOperation("Const", "indices")
        .SetAttrTensor("value", indices_t.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    // Axis for gather
    auto axis_t = tf_wrap::FastTensor::FromScalar<int32_t>(0);
    auto axis = graph.NewOperation("Const", "axis")
        .SetAttrTensor("value", axis_t.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output data{c, 0}, begin_out{begin, 0}, size_out{size_op, 0};
    TF_Output indices_out{indices, 0}, axis_out{axis, 0};
    
    (void)Slice(graph, "slice", data, begin_out, size_out, TF_FLOAT, TF_INT32);
    (void)Gather(graph, "gather", data, indices_out, TF_FLOAT, TF_INT32);
    (void)GatherV2(graph, "gatherv2", data, indices_out, axis_out, TF_FLOAT, TF_INT32, TF_INT32);
    
    REQUIRE(graph.num_operations() >= 8);
}

TEST(concat_split_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // Create two 1D tensors
    auto t1 = tf_wrap::FastTensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    auto t2 = tf_wrap::FastTensor::FromVector<float>({3}, {4.0f, 5.0f, 6.0f});
    
    auto c1 = graph.NewOperation("Const", "c1")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto c2 = graph.NewOperation("Const", "c2")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Axis for concat
    auto axis_t = tf_wrap::FastTensor::FromScalar<int32_t>(0);
    auto axis = graph.NewOperation("Const", "axis")
        .SetAttrTensor("value", axis_t.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output o1{c1, 0}, o2{c2, 0}, axis_out{axis, 0};
    
    std::vector<TF_Output> concat_values = {o1, o2};
    (void)ConcatV2(graph, "concat", concat_values, axis_out, TF_FLOAT, 2, TF_INT32);
    
    // For split, we need a tensor that can be split
    auto t_split = tf_wrap::FastTensor::FromVector<float>({6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto c_split = graph.NewOperation("Const", "c_split")
        .SetAttrTensor("value", t_split.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    TF_Output split_data{c_split, 0};
    (void)Split(graph, "split", axis_out, split_data, TF_FLOAT, 2);
    
    REQUIRE(graph.num_operations() >= 5);
}

TEST(pack_unpack_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // Create scalar tensors to pack
    auto t1 = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    auto t2 = tf_wrap::FastTensor::FromScalar<float>(2.0f);
    auto t3 = tf_wrap::FastTensor::FromScalar<float>(3.0f);
    
    auto c1 = graph.NewOperation("Const", "c1")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto c2 = graph.NewOperation("Const", "c2")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto c3 = graph.NewOperation("Const", "c3")
        .SetAttrTensor("value", t3.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output o1{c1, 0}, o2{c2, 0}, o3{c3, 0};
    
    std::vector<TF_Output> pack_values = {o1, o2, o3};
    auto packed = Pack(graph, "pack", pack_values, TF_FLOAT, 3);
    (void)Unpack(graph, "unpack", packed, TF_FLOAT, 3);
    
    REQUIRE(graph.num_operations() >= 5);
}

TEST(fill_range_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // Dims for Fill: [2, 3]
    auto dims_t = tf_wrap::FastTensor::FromVector<int32_t>({2}, {2, 3});
    auto dims = graph.NewOperation("Const", "dims")
        .SetAttrTensor("value", dims_t.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    // Value to fill
    auto val_t = tf_wrap::FastTensor::FromScalar<float>(5.0f);
    auto val = graph.NewOperation("Const", "val")
        .SetAttrTensor("value", val_t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Range parameters
    auto start_t = tf_wrap::FastTensor::FromScalar<float>(0.0f);
    auto limit_t = tf_wrap::FastTensor::FromScalar<float>(10.0f);
    auto delta_t = tf_wrap::FastTensor::FromScalar<float>(2.0f);
    
    auto start = graph.NewOperation("Const", "start")
        .SetAttrTensor("value", start_t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto limit = graph.NewOperation("Const", "limit")
        .SetAttrTensor("value", limit_t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto delta = graph.NewOperation("Const", "delta")
        .SetAttrTensor("value", delta_t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output dims_out{dims, 0}, val_out{val, 0};
    TF_Output start_out{start, 0}, limit_out{limit, 0}, delta_out{delta, 0};
    
    // Fill: T is the value type (FLOAT), index_type is the dims type (INT32)
    (void)Fill(graph, "fill", dims_out, val_out, TF_FLOAT, TF_INT32);
    (void)Range(graph, "range", start_out, limit_out, delta_out, TF_FLOAT);
    
    REQUIRE(graph.num_operations() >= 7);
}

TEST(pad_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // Create a 2x2 tensor
    auto t = tf_wrap::FastTensor::FromVector<float>({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto c = graph.NewOperation("Const", "c")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Paddings: [[1, 1], [1, 1]] - pad 1 on each side
    auto pad_t = tf_wrap::FastTensor::FromVector<int32_t>({2, 2}, {1, 1, 1, 1});
    auto paddings = graph.NewOperation("Const", "paddings")
        .SetAttrTensor("value", pad_t.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    // Constant value for PadV2
    auto const_val_t = tf_wrap::FastTensor::FromScalar<float>(0.0f);
    auto const_val = graph.NewOperation("Const", "const_val")
        .SetAttrTensor("value", const_val_t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output data{c, 0}, pad_out{paddings, 0}, const_out{const_val, 0};
    
    (void)Pad(graph, "pad", data, pad_out, TF_FLOAT, TF_INT32);
    (void)PadV2(graph, "padv2", data, pad_out, const_out, TF_FLOAT, TF_INT32);
    (void)MirrorPad(graph, "mirrorpad", data, pad_out, TF_FLOAT, TF_INT32, "REFLECT");
    
    REQUIRE(graph.num_operations() >= 6);
}

TEST(broadcast_select_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // Scalar tensor
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    auto c = graph.NewOperation("Const", "c")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Target shape: [2, 3]
    auto shape_t = tf_wrap::FastTensor::FromVector<int32_t>({2}, {2, 3});
    auto shape = graph.NewOperation("Const", "shape")
        .SetAttrTensor("value", shape_t.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    // Condition for SelectV2
    auto cond_t = tf_wrap::FastTensor::FromScalar<bool>(true);
    auto cond = graph.NewOperation("Const", "cond")
        .SetAttrTensor("value", cond_t.handle())
        .SetAttrType("dtype", TF_BOOL)
        .Finish();
    
    // Second value for SelectV2
    auto t2 = tf_wrap::FastTensor::FromScalar<float>(2.0f);
    auto c2 = graph.NewOperation("Const", "c2")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output data{c, 0}, shape_out{shape, 0}, cond_out{cond, 0}, data2{c2, 0};
    
    (void)BroadcastTo(graph, "broadcast", data, shape_out, TF_FLOAT, TF_INT32);
    (void)SelectV2(graph, "select", cond_out, data, data2, TF_FLOAT);
    
    REQUIRE(graph.num_operations() >= 6);
}

TEST(reverse_where_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // Create a 1D tensor
    auto t = tf_wrap::FastTensor::FromVector<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto c = graph.NewOperation("Const", "c")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Axis for reverse
    auto axis_t = tf_wrap::FastTensor::FromVector<int32_t>({1}, {0});
    auto axis = graph.NewOperation("Const", "axis")
        .SetAttrTensor("value", axis_t.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    // Boolean tensor for Where - use scalar bool
    auto bool_t = tf_wrap::FastTensor::FromScalar<bool>(true);
    auto bool_c = graph.NewOperation("Const", "bool_c")
        .SetAttrTensor("value", bool_t.handle())
        .SetAttrType("dtype", TF_BOOL)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output data{c, 0}, axis_out{axis, 0}, bool_out{bool_c, 0};
    
    (void)ReverseV2(graph, "reverse", data, axis_out, TF_FLOAT, TF_INT32);
    (void)Where(graph, "where", bool_out, TF_BOOL);
    
    REQUIRE(graph.num_operations() >= 5);
}

TEST(logical_reduce_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // Create rank-1 int tensor and cast to bool (FromVector<bool> is broken due to std::vector<bool>)
    auto t = tf_wrap::FastTensor::FromVector<int32_t>({4}, {1, 1, 0, 1});
    auto c_int = graph.NewOperation("Const", "c_int")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    // Cast to bool
    auto c = graph.NewOperation("Cast", "c")
        .AddInput(TF_Output{c_int, 0})
        .SetAttrType("SrcT", TF_INT32)
        .SetAttrType("DstT", TF_BOOL)
        .Finish();
    
    // Axis
    auto axis_t = tf_wrap::FastTensor::FromScalar<int32_t>(0);
    auto axis = graph.NewOperation("Const", "axis")
        .SetAttrTensor("value", axis_t.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output data{c, 0}, axis_out{axis, 0};
    
    (void)All(graph, "all", data, axis_out, TF_INT32);
    (void)Any(graph, "any", data, axis_out, TF_INT32);
    
    REQUIRE(graph.num_operations() >= 5);
}

TEST(leaky_relu_biasadd_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // Input tensor [1, 2, 2, 1] for BiasAdd (NHWC format)
    auto t = tf_wrap::FastTensor::FromVector<float>({1, 2, 2, 1}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto c = graph.NewOperation("Const", "c")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Bias tensor [1] (matches last dim of input)
    auto bias_t = tf_wrap::FastTensor::FromVector<float>({1}, {0.5f});
    auto bias = graph.NewOperation("Const", "bias")
        .SetAttrTensor("value", bias_t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output data{c, 0}, bias_out{bias, 0};
    
    (void)LeakyRelu(graph, "leakyrelu", data, TF_FLOAT);
    (void)BiasAdd(graph, "biasadd", data, bias_out, TF_FLOAT);
    
    REQUIRE(graph.num_operations() >= 4);
}

TEST(conv2d_pool_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // Input: [batch=1, height=4, width=4, channels=1]
    auto t_input = tf_wrap::FastTensor::FromVector<float>({1, 4, 4, 1}, 
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    auto input = graph.NewOperation("Const", "input")
        .SetAttrTensor("value", t_input.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Filter: [height=2, width=2, in_channels=1, out_channels=1]
    auto t_filter = tf_wrap::FastTensor::FromVector<float>({2, 2, 1, 1}, {0.25f, 0.25f, 0.25f, 0.25f});
    auto filter = graph.NewOperation("Const", "filter")
        .SetAttrTensor("value", t_filter.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output input_out{input, 0}, filter_out{filter, 0};
    
    std::vector<int64_t> strides = {1, 1, 1, 1};
    std::vector<int64_t> ksize = {1, 2, 2, 1};
    std::vector<int64_t> pool_strides = {1, 2, 2, 1};
    
    // Conv2D with stride 1, SAME padding
    (void)Conv2D(graph, "conv2d", input_out, filter_out, TF_FLOAT, strides, "SAME");
    
    // MaxPool and AvgPool
    (void)MaxPool(graph, "maxpool", input_out, TF_FLOAT, ksize, pool_strides, "SAME");
    (void)AvgPool(graph, "avgpool", input_out, TF_FLOAT, ksize, pool_strides, "SAME");
    
    REQUIRE(graph.num_operations() >= 5);
}

TEST(lrn_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // Input: [batch=1, height=2, width=2, channels=4]
    auto t = tf_wrap::FastTensor::FromVector<float>({1, 2, 2, 4}, 
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    auto c = graph.NewOperation("Const", "c")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output input{c, 0};
    
    // Local Response Normalization
    (void)LRN(graph, "lrn", input, TF_FLOAT);
    
    REQUIRE(graph.num_operations() >= 2);
}

TEST(random_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // Shape for random tensors
    auto shape_t = tf_wrap::FastTensor::FromVector<int32_t>({2}, {3, 3});
    auto shape = graph.NewOperation("Const", "shape")
        .SetAttrTensor("value", shape_t.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output shape_out{shape, 0};
    
    // RandomUniform: dtype is output type (FLOAT), T is shape type (INT32)
    (void)RandomUniform(graph, "randuniform", shape_out, TF_FLOAT, TF_INT32);
    (void)RandomStandardNormal(graph, "randnormal", shape_out, TF_FLOAT, TF_INT32);
    (void)TruncatedNormal(graph, "truncnormal", shape_out, TF_FLOAT, TF_INT32);
    
    REQUIRE(graph.num_operations() >= 4);
}

TEST(linspace_zeros_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // LinSpace parameters
    auto start_t = tf_wrap::FastTensor::FromScalar<float>(0.0f);
    auto stop_t = tf_wrap::FastTensor::FromScalar<float>(10.0f);
    auto num_t = tf_wrap::FastTensor::FromScalar<int32_t>(5);
    
    auto start = graph.NewOperation("Const", "start")
        .SetAttrTensor("value", start_t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto stop = graph.NewOperation("Const", "stop")
        .SetAttrTensor("value", stop_t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto num = graph.NewOperation("Const", "num")
        .SetAttrTensor("value", num_t.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output start_out{start, 0}, stop_out{stop, 0}, num_out{num, 0};
    
    (void)LinSpace(graph, "linspace", start_out, stop_out, num_out, TF_FLOAT, TF_INT32);
    // Note: "Zeros" op doesn't exist in TensorFlow C API. Use Fill or ZerosLike instead.
    
    REQUIRE(graph.num_operations() >= 4);
}

TEST(placeholder_const_ops_compile) {
    tf_wrap::FastGraph graph;
    
    auto t = tf_wrap::FastTensor::FromScalar<float>(42.0f);
    
    using namespace tf_wrap::ops;
    
    (void)Placeholder(graph, "placeholder", TF_FLOAT, {});
    (void)Const(graph, "const", t.handle(), TF_FLOAT);
    
    REQUIRE(graph.num_operations() >= 2);
}

TEST(identityn_shapen_ops_compile) {
    tf_wrap::FastGraph graph;
    
    auto t1 = tf_wrap::FastTensor::FromVector<float>({2}, {1.0f, 2.0f});
    auto t2 = tf_wrap::FastTensor::FromVector<float>({3}, {3.0f, 4.0f, 5.0f});
    
    auto c1 = graph.NewOperation("Const", "c1")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto c2 = graph.NewOperation("Const", "c2")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output o1{c1, 0}, o2{c2, 0};
    
    std::vector<TF_Output> inputs = {o1, o2};
    (void)IdentityN(graph, "identityn", inputs);
    (void)ShapeN(graph, "shapen", inputs, TF_FLOAT, 2, TF_INT32);
    
    REQUIRE(graph.num_operations() >= 4);
}

TEST(bitcast_checknumerics_ops_compile) {
    tf_wrap::FastGraph graph;
    
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    auto c = graph.NewOperation("Const", "c")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    TF_Output input{c, 0};
    
    // Bitcast float to int32 (same size)
    (void)Bitcast(graph, "bitcast", input, TF_FLOAT, TF_INT32);
    (void)CheckNumerics(graph, "checknumerics", input, TF_FLOAT, "checking");
    
    REQUIRE(graph.num_operations() >= 3);
}

TEST(linalg_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // Square matrix for linear algebra ops (must be positive definite for Cholesky)
    // Using a symmetric positive definite matrix: [[4,2],[2,5]]
    auto t = tf_wrap::FastTensor::FromVector<float>({2, 2}, {4.0f, 2.0f, 2.0f, 5.0f});
    auto c = graph.NewOperation("Const", "c")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    TF_Output mat{c, 0};
    
    (void)MatrixDeterminant(graph, "det", mat, TF_FLOAT);
    (void)MatrixInverse(graph, "inv", mat, TF_FLOAT);
    (void)Cholesky(graph, "chol", mat, TF_FLOAT);
    (void)Qr(graph, "qr", mat, TF_FLOAT);
    (void)Svd(graph, "svd", mat, TF_FLOAT);
    
    REQUIRE(graph.num_operations() >= 6);
}

TEST(batchmatmulv2_ops_compile) {
    tf_wrap::FastGraph graph;
    
    auto t1 = tf_wrap::FastTensor::FromVector<float>({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto t2 = tf_wrap::FastTensor::FromVector<float>({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
    
    auto c1 = graph.NewOperation("Const", "c1")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto c2 = graph.NewOperation("Const", "c2")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    TF_Output o1{c1, 0}, o2{c2, 0};
    
    (void)BatchMatMulV2(graph, "batchmatmulv2", o1, o2, TF_FLOAT);
    
    REQUIRE(graph.num_operations() >= 3);
}

TEST(einsum_ops_compile) {
    tf_wrap::FastGraph graph;
    
    auto t1 = tf_wrap::FastTensor::FromVector<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto t2 = tf_wrap::FastTensor::FromVector<float>({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    
    auto c1 = graph.NewOperation("Const", "c1")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto c2 = graph.NewOperation("Const", "c2")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    std::vector<TF_Output> inputs = {{c1, 0}, {c2, 0}};
    (void)Einsum(graph, "einsum", inputs, "ij,jk->ik", TF_FLOAT, 2);
    
    REQUIRE(graph.num_operations() >= 3);
}

TEST(strided_slice_ops_compile) {
    tf_wrap::FastGraph graph;
    
    auto t_data = tf_wrap::FastTensor::FromVector<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto t_begin = tf_wrap::FastTensor::FromVector<int32_t>({1}, {0});
    auto t_end = tf_wrap::FastTensor::FromVector<int32_t>({1}, {3});
    auto t_strides = tf_wrap::FastTensor::FromVector<int32_t>({1}, {2});
    
    auto data = graph.NewOperation("Const", "data")
        .SetAttrTensor("value", t_data.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto begin = graph.NewOperation("Const", "begin")
        .SetAttrTensor("value", t_begin.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    auto end = graph.NewOperation("Const", "end")
        .SetAttrTensor("value", t_end.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    auto strides = graph.NewOperation("Const", "strides")
        .SetAttrTensor("value", t_strides.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    using namespace tf_wrap::ops;
    TF_Output data_out{data, 0}, begin_out{begin, 0}, end_out{end, 0}, strides_out{strides, 0};
    
    (void)StridedSlice(graph, "stridedslice", data_out, begin_out, end_out, strides_out, TF_FLOAT, TF_INT32);
    
    REQUIRE(graph.num_operations() >= 5);
}

TEST(splitv_ops_compile) {
    tf_wrap::FastGraph graph;
    
    auto t_data = tf_wrap::FastTensor::FromVector<float>({6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto t_splits = tf_wrap::FastTensor::FromVector<int32_t>({2}, {2, 4});
    auto t_axis = tf_wrap::FastTensor::FromScalar<int32_t>(0);
    
    auto data = graph.NewOperation("Const", "data")
        .SetAttrTensor("value", t_data.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto splits = graph.NewOperation("Const", "splits")
        .SetAttrTensor("value", t_splits.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    auto axis = graph.NewOperation("Const", "axis")
        .SetAttrTensor("value", t_axis.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    using namespace tf_wrap::ops;
    TF_Output data_out{data, 0}, splits_out{splits, 0}, axis_out{axis, 0};
    
    (void)SplitV(graph, "splitv", data_out, splits_out, axis_out, TF_FLOAT, 2, TF_INT32);
    
    REQUIRE(graph.num_operations() >= 4);
}

TEST(gathernd_scatternd_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // GatherNd: params [2,3], indices [[0,1],[1,0]]
    auto t_params = tf_wrap::FastTensor::FromVector<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto t_indices = tf_wrap::FastTensor::FromVector<int32_t>({2, 2}, {0, 1, 1, 0});
    
    auto params = graph.NewOperation("Const", "params")
        .SetAttrTensor("value", t_params.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto indices = graph.NewOperation("Const", "indices")
        .SetAttrTensor("value", t_indices.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    using namespace tf_wrap::ops;
    TF_Output params_out{params, 0}, indices_out{indices, 0};
    
    (void)GatherNd(graph, "gathernd", params_out, indices_out, TF_FLOAT, TF_INT32);
    
    // ScatterNd: indices [2,1], updates [2], shape [4]
    auto t_sc_indices = tf_wrap::FastTensor::FromVector<int32_t>({2, 1}, {0, 2});
    auto t_updates = tf_wrap::FastTensor::FromVector<float>({2}, {9.0f, 10.0f});
    auto t_shape = tf_wrap::FastTensor::FromVector<int32_t>({1}, {4});
    
    auto sc_indices = graph.NewOperation("Const", "sc_indices")
        .SetAttrTensor("value", t_sc_indices.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    auto updates = graph.NewOperation("Const", "updates")
        .SetAttrTensor("value", t_updates.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto shape = graph.NewOperation("Const", "shape")
        .SetAttrTensor("value", t_shape.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    TF_Output sc_indices_out{sc_indices, 0}, updates_out{updates, 0}, shape_out{shape, 0};
    
    (void)ScatterNd(graph, "scatternd", sc_indices_out, updates_out, shape_out, TF_FLOAT, TF_INT32);
    
    REQUIRE(graph.num_operations() >= 7);
}

TEST(concat_legacy_ops_compile) {
    tf_wrap::FastGraph graph;
    
    auto t1 = tf_wrap::FastTensor::FromVector<float>({2}, {1.0f, 2.0f});
    auto t2 = tf_wrap::FastTensor::FromVector<float>({2}, {3.0f, 4.0f});
    auto t_axis = tf_wrap::FastTensor::FromScalar<int32_t>(0);
    
    auto c1 = graph.NewOperation("Const", "c1")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto c2 = graph.NewOperation("Const", "c2")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto axis = graph.NewOperation("Const", "axis")
        .SetAttrTensor("value", t_axis.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    using namespace tf_wrap::ops;
    TF_Output axis_out{axis, 0};
    std::vector<TF_Output> values = {{c1, 0}, {c2, 0}};
    
    // Legacy Concat (axis first)
    (void)Concat(graph, "concat_legacy", axis_out, values, TF_FLOAT, 2);
    
    REQUIRE(graph.num_operations() >= 4);
}

TEST(fused_batchnorm_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // Input: [batch=1, height=2, width=2, channels=2]
    auto t_x = tf_wrap::FastTensor::FromVector<float>({1, 2, 2, 2}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    auto t_scale = tf_wrap::FastTensor::FromVector<float>({2}, {1.0f, 1.0f});
    auto t_offset = tf_wrap::FastTensor::FromVector<float>({2}, {0.0f, 0.0f});
    auto t_mean = tf_wrap::FastTensor::FromVector<float>({2}, {0.0f, 0.0f});
    auto t_variance = tf_wrap::FastTensor::FromVector<float>({2}, {1.0f, 1.0f});
    
    auto x = graph.NewOperation("Const", "x")
        .SetAttrTensor("value", t_x.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto scale = graph.NewOperation("Const", "scale")
        .SetAttrTensor("value", t_scale.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto offset = graph.NewOperation("Const", "offset")
        .SetAttrTensor("value", t_offset.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto mean = graph.NewOperation("Const", "mean")
        .SetAttrTensor("value", t_mean.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto variance = graph.NewOperation("Const", "variance")
        .SetAttrTensor("value", t_variance.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    TF_Output x_out{x, 0}, scale_out{scale, 0}, offset_out{offset, 0}, mean_out{mean, 0}, var_out{variance, 0};
    
    (void)FusedBatchNorm(graph, "fusedbatchnorm", x_out, scale_out, offset_out, mean_out, var_out, TF_FLOAT);
    (void)FusedBatchNormV3(graph, "fusedbatchnormv3", x_out, scale_out, offset_out, mean_out, var_out, TF_FLOAT, TF_FLOAT);
    
    REQUIRE(graph.num_operations() >= 7);
}

TEST(crossentropy_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // Features: [batch=2, classes=3], Labels: [batch=2, classes=3] (one-hot)
    auto t_features = tf_wrap::FastTensor::FromVector<float>({2, 3}, {0.5f, 0.3f, 0.2f, 0.1f, 0.8f, 0.1f});
    auto t_labels = tf_wrap::FastTensor::FromVector<float>({2, 3}, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f});
    auto t_sparse_labels = tf_wrap::FastTensor::FromVector<int32_t>({2}, {0, 1});
    
    auto features = graph.NewOperation("Const", "features")
        .SetAttrTensor("value", t_features.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto labels = graph.NewOperation("Const", "labels")
        .SetAttrTensor("value", t_labels.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto sparse_labels = graph.NewOperation("Const", "sparse_labels")
        .SetAttrTensor("value", t_sparse_labels.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    using namespace tf_wrap::ops;
    TF_Output features_out{features, 0}, labels_out{labels, 0}, sparse_labels_out{sparse_labels, 0};
    
    (void)SoftmaxCrossEntropyWithLogits(graph, "softmaxce", features_out, labels_out, TF_FLOAT);
    (void)SparseSoftmaxCrossEntropyWithLogits(graph, "sparsesoftmaxce", features_out, sparse_labels_out, TF_FLOAT, TF_INT32);
    
    REQUIRE(graph.num_operations() >= 5);
}

TEST(pool3d_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // Input: [batch=1, depth=4, height=4, width=4, channels=1] - 64 elements
    std::vector<int64_t> input_dims = {1, 4, 4, 4, 1};
    std::vector<float> input_data(64, 1.0f);
    auto t_input = tf_wrap::FastTensor::FromVector<float>(input_dims, input_data);
    
    auto input = graph.NewOperation("Const", "input")
        .SetAttrTensor("value", t_input.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    TF_Output input_out{input, 0};
    
    std::vector<int64_t> ksize = {1, 2, 2, 2, 1};
    std::vector<int64_t> strides = {1, 2, 2, 2, 1};
    
    (void)MaxPool3D(graph, "maxpool3d", input_out, TF_FLOAT, ksize, strides, "VALID");
    (void)AvgPool3D(graph, "avgpool3d", input_out, TF_FLOAT, ksize, strides, "VALID");
    
    REQUIRE(graph.num_operations() >= 3);
}

TEST(depthwise_conv_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // Input: [batch=1, height=4, width=4, in_channels=1]
    std::vector<int64_t> input_dims = {1, 4, 4, 1};
    std::vector<float> input_data(16, 1.0f);
    auto t_input = tf_wrap::FastTensor::FromVector<float>(input_dims, input_data);
    // Filter: [height=2, width=2, in_channels=1, channel_multiplier=1]
    auto t_filter = tf_wrap::FastTensor::FromVector<float>({2, 2, 1, 1}, {1.0f, 1.0f, 1.0f, 1.0f});
    
    auto input = graph.NewOperation("Const", "input")
        .SetAttrTensor("value", t_input.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto filter = graph.NewOperation("Const", "filter")
        .SetAttrTensor("value", t_filter.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    TF_Output input_out{input, 0}, filter_out{filter, 0};
    
    std::vector<int64_t> strides = {1, 1, 1, 1};
    (void)DepthwiseConv2dNative(graph, "depthwiseconv", input_out, filter_out, TF_FLOAT, strides, "VALID");
    
    REQUIRE(graph.num_operations() >= 3);
}

TEST(conv2d_backprop_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // Input sizes: [batch=1, height=4, width=4, channels=1]
    auto t_sizes = tf_wrap::FastTensor::FromVector<int32_t>({4}, {1, 4, 4, 1});
    // Filter: [height=2, width=2, out_channels=1, in_channels=1]
    auto t_filter = tf_wrap::FastTensor::FromVector<float>({2, 2, 1, 1}, {1.0f, 1.0f, 1.0f, 1.0f});
    // Output gradient: [batch=1, height=3, width=3, channels=1]
    auto t_out_backprop = tf_wrap::FastTensor::FromVector<float>({1, 3, 3, 1}, 
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    
    auto sizes = graph.NewOperation("Const", "sizes")
        .SetAttrTensor("value", t_sizes.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    auto filter = graph.NewOperation("Const", "filter")
        .SetAttrTensor("value", t_filter.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto out_backprop = graph.NewOperation("Const", "out_backprop")
        .SetAttrTensor("value", t_out_backprop.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    TF_Output sizes_out{sizes, 0}, filter_out{filter, 0}, backprop_out{out_backprop, 0};
    
    std::vector<int64_t> strides = {1, 1, 1, 1};
    (void)Conv2DBackpropInput(graph, "conv2dbackprop", sizes_out, filter_out, backprop_out, TF_FLOAT, strides, "VALID");
    
    REQUIRE(graph.num_operations() >= 4);
}

TEST(dropout_ops_compile) {
    // Note: "Dropout" op doesn't exist in TensorFlow C API.
    // Dropout is typically implemented as a combination of RandomUniform, Floor, Mul, etc.
    // This test verifies Select which can be used in dropout-like implementations.
    tf_wrap::FastGraph graph;
    
    auto t_x = tf_wrap::FastTensor::FromVector<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    // Use int and cast to bool (FromVector<bool> is broken due to std::vector<bool>)
    auto t_cond_int = tf_wrap::FastTensor::FromVector<int32_t>({4}, {1, 0, 1, 0});
    auto t_zeros = tf_wrap::FastTensor::FromVector<float>({4}, {0.0f, 0.0f, 0.0f, 0.0f});
    
    auto x = graph.NewOperation("Const", "x")
        .SetAttrTensor("value", t_x.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto cond_int = graph.NewOperation("Const", "cond_int")
        .SetAttrTensor("value", t_cond_int.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    // Cast to bool
    auto cond = graph.NewOperation("Cast", "cond")
        .AddInput(TF_Output{cond_int, 0})
        .SetAttrType("SrcT", TF_INT32)
        .SetAttrType("DstT", TF_BOOL)
        .Finish();
    
    auto zeros = graph.NewOperation("Const", "zeros")
        .SetAttrTensor("value", t_zeros.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    TF_Output x_out{x, 0}, cond_out{cond, 0}, zeros_out{zeros, 0};
    
    // Select can be used to implement dropout masking
    (void)SelectV2(graph, "dropout_select", cond_out, x_out, zeros_out, TF_FLOAT);
    
    REQUIRE(graph.num_operations() >= 5);
}

TEST(multinomial_randomshuffle_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // Logits: [batch=1, classes=4]
    auto t_logits = tf_wrap::FastTensor::FromVector<float>({1, 4}, {0.1f, 0.2f, 0.3f, 0.4f});
    auto t_num_samples = tf_wrap::FastTensor::FromScalar<int32_t>(5);
    auto t_value = tf_wrap::FastTensor::FromVector<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    
    auto logits = graph.NewOperation("Const", "logits")
        .SetAttrTensor("value", t_logits.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto num_samples = graph.NewOperation("Const", "num_samples")
        .SetAttrTensor("value", t_num_samples.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    auto value = graph.NewOperation("Const", "value")
        .SetAttrTensor("value", t_value.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    TF_Output logits_out{logits, 0}, num_out{num_samples, 0}, value_out{value, 0};
    
    (void)Multinomial(graph, "multinomial", logits_out, num_out, TF_FLOAT, TF_INT64);
    (void)RandomShuffle(graph, "randomshuffle", value_out, TF_FLOAT);
    
    REQUIRE(graph.num_operations() >= 5);
}

TEST(variable_ops_compile) {
    tf_wrap::FastGraph graph;
    
    using namespace tf_wrap::ops;
    
    std::vector<int64_t> shape = {2, 3};
    
    (void)Variable(graph, "variable", shape, TF_FLOAT);
    (void)VariableV2(graph, "variablev2", shape, TF_FLOAT);
    (void)VarHandleOp(graph, "varhandle", TF_FLOAT, shape);
    
    REQUIRE(graph.num_operations() >= 3);
}

TEST(variable_read_assign_ops_compile) {
    tf_wrap::FastGraph graph;
    
    using namespace tf_wrap::ops;
    
    std::vector<int64_t> shape = {2, 2};
    auto var_handle = VarHandleOp(graph, "varhandle", TF_FLOAT, shape);
    
    auto t_value = tf_wrap::FastTensor::FromVector<float>({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto value = graph.NewOperation("Const", "value")
        .SetAttrTensor("value", t_value.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    TF_Output value_out{value, 0};
    
    (void)ReadVariableOp(graph, "readvariable", var_handle, TF_FLOAT);
    (void)AssignVariableOp(graph, "assignvariable", var_handle, value_out, TF_FLOAT);
    (void)AssignAddVariableOp(graph, "assignaddvariable", var_handle, value_out, TF_FLOAT);
    (void)AssignSubVariableOp(graph, "assignsubvariable", var_handle, value_out, TF_FLOAT);
    
    REQUIRE(graph.num_operations() >= 6);
}

TEST(image_resize_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // Image: [batch=1, height=4, width=4, channels=1]
    std::vector<int64_t> image_dims = {1, 4, 4, 1};
    std::vector<float> image_data(16, 0.5f);
    auto t_image = tf_wrap::FastTensor::FromVector<float>(image_dims, image_data);
    auto t_size = tf_wrap::FastTensor::FromVector<int32_t>({2}, {8, 8});
    
    auto image = graph.NewOperation("Const", "image")
        .SetAttrTensor("value", t_image.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto size = graph.NewOperation("Const", "size")
        .SetAttrTensor("value", t_size.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    using namespace tf_wrap::ops;
    TF_Output image_out{image, 0}, size_out{size, 0};
    
    (void)ResizeBilinear(graph, "resizebilinear", image_out, size_out, TF_FLOAT);
    (void)ResizeBicubic(graph, "resizebicubic", image_out, size_out, TF_FLOAT);
    (void)ResizeNearestNeighbor(graph, "resizenearestneighbor", image_out, size_out, TF_FLOAT);
    
    REQUIRE(graph.num_operations() >= 5);
}

TEST(crop_and_resize_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // Image: [batch=1, height=4, width=4, channels=1]
    std::vector<int64_t> image_dims = {1, 4, 4, 1};
    std::vector<float> image_data(16, 0.5f);
    auto t_image = tf_wrap::FastTensor::FromVector<float>(image_dims, image_data);
    // Boxes: [num_boxes=1, 4] normalized coords [y1, x1, y2, x2]
    auto t_boxes = tf_wrap::FastTensor::FromVector<float>({1, 4}, {0.0f, 0.0f, 1.0f, 1.0f});
    // Box indices: which image each box belongs to
    auto t_box_ind = tf_wrap::FastTensor::FromVector<int32_t>({1}, {0});
    // Crop size
    auto t_crop_size = tf_wrap::FastTensor::FromVector<int32_t>({2}, {2, 2});
    
    auto image = graph.NewOperation("Const", "image")
        .SetAttrTensor("value", t_image.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto boxes = graph.NewOperation("Const", "boxes")
        .SetAttrTensor("value", t_boxes.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto box_ind = graph.NewOperation("Const", "box_ind")
        .SetAttrTensor("value", t_box_ind.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    auto crop_size = graph.NewOperation("Const", "crop_size")
        .SetAttrTensor("value", t_crop_size.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    using namespace tf_wrap::ops;
    TF_Output image_out{image, 0}, boxes_out{boxes, 0}, box_ind_out{box_ind, 0}, crop_size_out{crop_size, 0};
    
    (void)CropAndResize(graph, "cropandresize", image_out, boxes_out, box_ind_out, crop_size_out, TF_FLOAT);
    
    REQUIRE(graph.num_operations() >= 5);
}

TEST(nms_ops_compile) {
    tf_wrap::FastGraph graph;
    
    // Boxes: [num_boxes=3, 4]
    auto t_boxes = tf_wrap::FastTensor::FromVector<float>({3, 4}, {
        0.0f, 0.0f, 1.0f, 1.0f,
        0.1f, 0.1f, 1.1f, 1.1f,
        2.0f, 2.0f, 3.0f, 3.0f
    });
    // Scores: [num_boxes=3]
    auto t_scores = tf_wrap::FastTensor::FromVector<float>({3}, {0.9f, 0.8f, 0.7f});
    auto t_max_output = tf_wrap::FastTensor::FromScalar<int32_t>(2);
    auto t_iou_threshold = tf_wrap::FastTensor::FromScalar<float>(0.5f);
    auto t_score_threshold = tf_wrap::FastTensor::FromScalar<float>(0.0f);
    
    auto boxes = graph.NewOperation("Const", "boxes")
        .SetAttrTensor("value", t_boxes.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto scores = graph.NewOperation("Const", "scores")
        .SetAttrTensor("value", t_scores.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto max_output = graph.NewOperation("Const", "max_output")
        .SetAttrTensor("value", t_max_output.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    auto iou_threshold = graph.NewOperation("Const", "iou_threshold")
        .SetAttrTensor("value", t_iou_threshold.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto score_threshold = graph.NewOperation("Const", "score_threshold")
        .SetAttrTensor("value", t_score_threshold.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    TF_Output boxes_out{boxes, 0}, scores_out{scores, 0}, max_out{max_output, 0};
    TF_Output iou_out{iou_threshold, 0}, score_out{score_threshold, 0};
    
    (void)NonMaxSuppression(graph, "nms", boxes_out, scores_out, max_out);
    (void)NonMaxSuppressionV3(graph, "nmsv3", boxes_out, scores_out, max_out, iou_out, score_out);
    
    REQUIRE(graph.num_operations() >= 7);
}

// Note: File I/O ops (ReadFile, WriteFile, DecodeJpeg, etc.) and string ops
// require runtime file access or specific formats - tested separately or skipped

TEST(pack_ops_compile) {
    tf_wrap::FastGraph graph;
    
    auto t1 = tf_wrap::FastTensor::FromVector<float>({2}, {1.0f, 2.0f});
    auto t2 = tf_wrap::FastTensor::FromVector<float>({2}, {3.0f, 4.0f});
    
    auto c1 = graph.NewOperation("Const", "c1")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto c2 = graph.NewOperation("Const", "c2")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    using namespace tf_wrap::ops;
    
    std::vector<TF_Output> values = {{c1, 0}, {c2, 0}};
    // Pack: last parameter is N (number of tensors), not axis
    (void)Pack(graph, "pack", values, TF_FLOAT, 2);
    
    REQUIRE(graph.num_operations() >= 3);
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
    RUN_TEST(trig_ops_compile);
    RUN_TEST(more_math_ops_compile);
    RUN_TEST(more_binary_ops_compile);
    RUN_TEST(array_manipulation_ops_compile);
    RUN_TEST(transpose_tile_ops_compile);
    RUN_TEST(slice_gather_ops_compile);
    RUN_TEST(concat_split_ops_compile);
    RUN_TEST(pack_unpack_ops_compile);
    RUN_TEST(fill_range_ops_compile);
    RUN_TEST(pad_ops_compile);
    RUN_TEST(broadcast_select_ops_compile);
    RUN_TEST(reverse_where_ops_compile);
    RUN_TEST(logical_reduce_ops_compile);
    RUN_TEST(leaky_relu_biasadd_ops_compile);
    RUN_TEST(conv2d_pool_ops_compile);
    RUN_TEST(lrn_ops_compile);
    RUN_TEST(random_ops_compile);
    RUN_TEST(linspace_zeros_ops_compile);
    RUN_TEST(placeholder_const_ops_compile);
    RUN_TEST(identityn_shapen_ops_compile);
    RUN_TEST(bitcast_checknumerics_ops_compile);
    RUN_TEST(linalg_ops_compile);
    RUN_TEST(batchmatmulv2_ops_compile);
    RUN_TEST(einsum_ops_compile);
    RUN_TEST(strided_slice_ops_compile);
    RUN_TEST(splitv_ops_compile);
    RUN_TEST(gathernd_scatternd_ops_compile);
    RUN_TEST(concat_legacy_ops_compile);
    RUN_TEST(fused_batchnorm_ops_compile);
    RUN_TEST(crossentropy_ops_compile);
    RUN_TEST(pool3d_ops_compile);
    RUN_TEST(depthwise_conv_ops_compile);
    RUN_TEST(conv2d_backprop_ops_compile);
    RUN_TEST(dropout_ops_compile);
    RUN_TEST(multinomial_randomshuffle_ops_compile);
    RUN_TEST(variable_ops_compile);
    RUN_TEST(variable_read_assign_ops_compile);
    RUN_TEST(image_resize_ops_compile);
    RUN_TEST(crop_and_resize_ops_compile);
    RUN_TEST(nms_ops_compile);
    RUN_TEST(pack_ops_compile);
    
    std::cout << "\n========================================\n";
    std::cout << "Passed: " << tests_passed << ", Failed: " << tests_failed << "\n";
    
    return tests_failed > 0 ? 1 : 0;
}
