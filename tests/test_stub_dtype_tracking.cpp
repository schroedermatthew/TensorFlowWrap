// tests/test_stub_dtype_tracking.cpp
// Tests that verify the TF stub correctly tracks dtype attributes
//
// Purpose: Ensure dtype-related bugs are visible in stub tests
// Problem: Original stub's TF_SetAttrType was empty

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "tf_wrap/core.hpp"

// ============================================================================
// Basic Dtype Tracking
// ============================================================================

TEST_CASE("Const with TF_INT32 dtype") {
    tf_wrap::Graph g;
    
    auto tensor = tf_wrap::Tensor::FromVector<int32_t>({2}, {1, 2});
    auto* op = g.NewOperation("Const", "IntConst")
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    // Must return TF_INT32, not TF_FLOAT
    CHECK(TF_OperationOutputType({op, 0}) == TF_INT32);
}

TEST_CASE("Const with TF_DOUBLE dtype") {
    tf_wrap::Graph g;
    
    auto tensor = tf_wrap::Tensor::FromVector<double>({2}, {1.0, 2.0});
    auto* op = g.NewOperation("Const", "DoubleConst")
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", TF_DOUBLE)
        .Finish();
    
    CHECK(TF_OperationOutputType({op, 0}) == TF_DOUBLE);
}

TEST_CASE("Const with TF_BOOL dtype") {
    tf_wrap::Graph g;
    
    auto tensor = tf_wrap::Tensor::FromVector<bool>({2}, {true, false});
    auto* op = g.NewOperation("Const", "BoolConst")
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", TF_BOOL)
        .Finish();
    
    CHECK(TF_OperationOutputType({op, 0}) == TF_BOOL);
}

TEST_CASE("Identity with T attribute") {
    tf_wrap::Graph g;
    
    auto tensor = tf_wrap::Tensor::FromVector<int64_t>({2}, {1, 2});
    auto* const_op = g.NewOperation("Const", "Int64Const")
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", TF_INT64)
        .Finish();
    
    auto* id_op = g.NewOperation("Identity", "Id")
        .AddInput({const_op, 0})
        .SetAttrType("T", TF_INT64)
        .Finish();
    
    CHECK(TF_OperationOutputType({id_op, 0}) == TF_INT64);
}

TEST_CASE("Placeholder dtype tracking") {
    tf_wrap::Graph g;
    
    auto* ph = g.NewOperation("Placeholder", "input")
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    CHECK(TF_OperationOutputType({ph, 0}) == TF_FLOAT);
}

// ============================================================================
// Multiple Ops with Different Dtypes
// ============================================================================

TEST_CASE("Graph with mixed dtypes") {
    tf_wrap::Graph g;
    
    auto f = tf_wrap::Tensor::FromScalar<float>(1.0f);
    auto i = tf_wrap::Tensor::FromScalar<int32_t>(1);
    auto d = tf_wrap::Tensor::FromScalar<double>(1.0);
    auto b = tf_wrap::Tensor::FromScalar<bool>(true);
    
    auto* float_op = g.NewOperation("Const", "Float")
        .SetAttrTensor("value", f.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* int_op = g.NewOperation("Const", "Int")
        .SetAttrTensor("value", i.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    auto* double_op = g.NewOperation("Const", "Double")
        .SetAttrTensor("value", d.handle())
        .SetAttrType("dtype", TF_DOUBLE)
        .Finish();
    
    auto* bool_op = g.NewOperation("Const", "Bool")
        .SetAttrTensor("value", b.handle())
        .SetAttrType("dtype", TF_BOOL)
        .Finish();
    
    CHECK(TF_OperationOutputType({float_op, 0}) == TF_FLOAT);
    CHECK(TF_OperationOutputType({int_op, 0}) == TF_INT32);
    CHECK(TF_OperationOutputType({double_op, 0}) == TF_DOUBLE);
    CHECK(TF_OperationOutputType({bool_op, 0}) == TF_BOOL);
}

// ============================================================================
// Binary Ops
// ============================================================================

TEST_CASE("AddV2 with T attribute") {
    tf_wrap::Graph g;
    
    auto a = tf_wrap::Tensor::FromScalar<float>(1.0f);
    auto b = tf_wrap::Tensor::FromScalar<float>(2.0f);
    
    auto* op_a = g.NewOperation("Const", "A")
        .SetAttrTensor("value", a.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* op_b = g.NewOperation("Const", "B")
        .SetAttrTensor("value", b.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* add_op = g.NewOperation("AddV2", "Sum")
        .AddInput({op_a, 0})
        .AddInput({op_b, 0})
        .SetAttrType("T", TF_FLOAT)
        .Finish();
    
    CHECK(TF_OperationOutputType({add_op, 0}) == TF_FLOAT);
}

// ============================================================================
// Cast Operation
// ============================================================================

TEST_CASE("Cast uses DstT for output dtype") {
    tf_wrap::Graph g;
    
    auto input = tf_wrap::Tensor::FromScalar<float>(1.0f);
    auto* input_op = g.NewOperation("Const", "Input")
        .SetAttrTensor("value", input.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* cast_op = g.NewOperation("Cast", "CastToInt")
        .AddInput({input_op, 0})
        .SetAttrType("SrcT", TF_FLOAT)
        .SetAttrType("DstT", TF_INT32)
        .Finish();
    
    // Output should be DstT, not SrcT
    CHECK(TF_OperationOutputType({cast_op, 0}) == TF_INT32);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_CASE("Null operation returns default") {
    TF_Output out{nullptr, 0};
    CHECK(TF_OperationOutputType(out) == TF_FLOAT);
}

TEST_CASE("Op without dtype attr returns default") {
    tf_wrap::Graph g;
    auto* op = g.NewOperation("NoOp", "noop").Finish();
    
    TF_DataType dt = TF_OperationOutputType({op, 0});
    CHECK(dt == TF_FLOAT);
}

// ============================================================================
// All Supported Dtypes
// ============================================================================

TEST_CASE("All common dtypes tracked correctly") {
    struct TestCase {
        TF_DataType dtype;
        const char* name;
    };
    
    std::vector<TestCase> cases = {
        {TF_FLOAT, "TF_FLOAT"},
        {TF_DOUBLE, "TF_DOUBLE"},
        {TF_INT32, "TF_INT32"},
        {TF_INT64, "TF_INT64"},
        {TF_INT16, "TF_INT16"},
        {TF_INT8, "TF_INT8"},
        {TF_UINT8, "TF_UINT8"},
        {TF_UINT16, "TF_UINT16"},
        {TF_UINT32, "TF_UINT32"},
        {TF_UINT64, "TF_UINT64"},
        {TF_BOOL, "TF_BOOL"},
        {TF_COMPLEX64, "TF_COMPLEX64"},
        {TF_COMPLEX128, "TF_COMPLEX128"},
    };
    
    for (const auto& tc : cases) {
        SUBCASE(tc.name) {
            tf_wrap::Graph g;
            
            std::string name = std::string("ph_") + tc.name;
            auto* op = g.NewOperation("Placeholder", name)
                .SetAttrType("dtype", tc.dtype)
                .Finish();
            
            CHECK(TF_OperationOutputType({op, 0}) == tc.dtype);
        }
    }
}
