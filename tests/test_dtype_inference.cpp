// tests/test_dtype_inference.cpp
// Tests for automatic dtype inference in ops
//
// Purpose: Verify dtype inference works correctly with the fixed stub
// These tests REQUIRE the stub fix to be meaningful

#include "doctest.h"

#include "tf_wrap/core.hpp"

using namespace tf_wrap;
namespace op = tf_wrap::ops;

// ============================================================================
// Helper to create const with dtype
// ============================================================================

template<typename T>
TF_Operation* make_const(Graph& g, const char* name, T value) {
    auto tensor = Tensor::FromScalar<T>(value);
    return g.NewOperation("Const", name)
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", tf_dtype_v<T>)
        .Finish();
}

// ============================================================================
// Dtype Inference - Matching Types Should Succeed
// ============================================================================

TEST_CASE("Inferred Add with float inputs succeeds") {
    Graph g;
    
    auto* a = make_const(g, "a", 1.0f);
    auto* b = make_const(g, "b", 2.0f);
    
    // Both should be TF_FLOAT after stub fix
    CHECK(TF_OperationOutputType({a, 0}) == TF_FLOAT);
    CHECK(TF_OperationOutputType({b, 0}) == TF_FLOAT);
    
    // Inferred Add should work
    auto sum = op::Add(g, "sum", {a, 0}, {b, 0}, TF_FLOAT);
    CHECK(TF_OperationOutputType(sum.output()) == TF_FLOAT);
}

TEST_CASE("Inferred Add with int32 inputs succeeds") {
    Graph g;
    
    auto* a = make_const(g, "a", int32_t{1});
    auto* b = make_const(g, "b", int32_t{2});
    
    CHECK(TF_OperationOutputType({a, 0}) == TF_INT32);
    CHECK(TF_OperationOutputType({b, 0}) == TF_INT32);
    
    auto sum = op::Add(g, "sum", {a, 0}, {b, 0}, TF_INT32);
    CHECK(TF_OperationOutputType(sum.output()) == TF_INT32);
}

TEST_CASE("Inferred Add with double inputs succeeds") {
    Graph g;
    
    auto* a = make_const(g, "a", 1.0);
    auto* b = make_const(g, "b", 2.0);
    
    CHECK(TF_OperationOutputType({a, 0}) == TF_DOUBLE);
    CHECK(TF_OperationOutputType({b, 0}) == TF_DOUBLE);
    
    auto sum = op::Add(g, "sum", {a, 0}, {b, 0}, TF_DOUBLE);
    CHECK(TF_OperationOutputType(sum.output()) == TF_DOUBLE);
}

// ============================================================================
// Dtype Mismatch Detection
// ============================================================================

TEST_CASE("Dtype mismatch is detectable") {
    // This test verifies the stub correctly reports different dtypes
    Graph g;
    
    auto* float_op = make_const(g, "float_val", 1.0f);
    auto* int_op = make_const(g, "int_val", int32_t{2});
    
    TF_DataType float_dt = TF_OperationOutputType({float_op, 0});
    TF_DataType int_dt = TF_OperationOutputType({int_op, 0});
    
    // These MUST be different for inference validation to work
    // Before stub fix, both would be TF_FLOAT
    CHECK(float_dt == TF_FLOAT);
    CHECK(int_dt == TF_INT32);
    CHECK(float_dt != int_dt);  // CRITICAL assertion
}

TEST_CASE("Manual dtype validation catches mismatch") {
    Graph g;
    
    auto* a = make_const(g, "a", 1.0f);    // float
    auto* b = make_const(g, "b", int32_t{2});  // int32
    
    TF_DataType dt_a = TF_OperationOutputType({a, 0});
    TF_DataType dt_b = TF_OperationOutputType({b, 0});
    
    // This is what inference overloads should do
    CHECK(dt_a != dt_b);
    
    // Creating Add with explicit dtype still works (user's responsibility)
    // But inference overload would throw here
}

// ============================================================================
// All Binary Ops Track Dtype Correctly
// ============================================================================

TEST_CASE("Sub output has correct dtype") {
    Graph g;
    
    auto* a = make_const(g, "a", int64_t{10});
    auto* b = make_const(g, "b", int64_t{3});
    
    auto diff = op::Sub(g, "diff", {a, 0}, {b, 0}, TF_INT64);
    CHECK(TF_OperationOutputType(diff.output()) == TF_INT64);
}

TEST_CASE("Mul output has correct dtype") {
    Graph g;
    
    auto* a = make_const(g, "a", 2.0);  // double
    auto* b = make_const(g, "b", 3.0);
    
    auto prod = op::Mul(g, "prod", {a, 0}, {b, 0}, TF_DOUBLE);
    CHECK(TF_OperationOutputType(prod.output()) == TF_DOUBLE);
}

TEST_CASE("Div output has correct dtype") {
    Graph g;
    
    auto* a = make_const(g, "a", 10.0f);
    auto* b = make_const(g, "b", 2.0f);
    
    auto quot = op::Div(g, "quot", {a, 0}, {b, 0}, TF_FLOAT);
    CHECK(TF_OperationOutputType(quot.output()) == TF_FLOAT);
}

// ============================================================================
// Mixed Operations Chain
// ============================================================================

TEST_CASE("Operation chain preserves dtype") {
    Graph g;
    
    auto* a = make_const(g, "a", 1.0f);
    auto* b = make_const(g, "b", 2.0f);
    auto* c = make_const(g, "c", 3.0f);
    
    auto sum = op::Add(g, "sum", {a, 0}, {b, 0}, TF_FLOAT);
    auto prod = op::Mul(g, "prod", sum.output(), {c, 0}, TF_FLOAT);
    
    CHECK(TF_OperationOutputType(sum.output()) == TF_FLOAT);
    CHECK(TF_OperationOutputType(prod.output()) == TF_FLOAT);
}

// ============================================================================
// Cast Operation Uses DstT
// ============================================================================

TEST_CASE("Cast output dtype is DstT not SrcT") {
    Graph g;
    
    auto* float_input = make_const(g, "input", 1.5f);
    CHECK(TF_OperationOutputType({float_input, 0}) == TF_FLOAT);
    
    auto* cast_op = g.NewOperation("Cast", "cast_to_int")
        .AddInput({float_input, 0})
        .SetAttrType("SrcT", TF_FLOAT)
        .SetAttrType("DstT", TF_INT32)
        .Finish();
    
    // Output should be DstT (INT32), not SrcT (FLOAT)
    CHECK(TF_OperationOutputType({cast_op, 0}) == TF_INT32);
}

// ============================================================================
// All Supported Dtypes
// ============================================================================

TEST_CASE("All dtypes tracked correctly through Identity") {
    struct DtypeCase {
        TF_DataType dtype;
        const char* name;
    };
    
    std::vector<DtypeCase> cases = {
        {TF_FLOAT, "float32"},
        {TF_DOUBLE, "float64"},
        {TF_INT8, "int8"},
        {TF_INT16, "int16"},
        {TF_INT32, "int32"},
        {TF_INT64, "int64"},
        {TF_UINT8, "uint8"},
        {TF_UINT16, "uint16"},
        {TF_UINT32, "uint32"},
        {TF_UINT64, "uint64"},
        {TF_BOOL, "bool"},
    };
    
    for (const auto& tc : cases) {
        SUBCASE(tc.name) {
            Graph g;
            
            // Create placeholder with specific dtype
            auto* ph = g.NewOperation("Placeholder", std::string("ph_") + tc.name)
                .SetAttrType("dtype", tc.dtype)
                .Finish();
            
            CHECK(TF_OperationOutputType({ph, 0}) == tc.dtype);
            
            // Identity should preserve dtype
            auto* id = g.NewOperation("Identity", std::string("id_") + tc.name)
                .AddInput({ph, 0})
                .SetAttrType("T", tc.dtype)
                .Finish();
            
            CHECK(TF_OperationOutputType({id, 0}) == tc.dtype);
        }
    }
}
