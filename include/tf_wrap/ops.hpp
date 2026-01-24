// ============================================================================
// tf_wrap/ops.hpp - Auto-generated TensorFlow Operations Wrapper
// Generated: 2026-01-22T17:39:33.280529
// Total operations: 160
// ============================================================================
//
// This file provides type-safe C++20 wrappers for TensorFlow operations.
// Each op is wrapped as a function that creates the operation in a graph.
//
// Usage:
//   using namespace tf_wrap::ops;
//   
//   Graph graph;
//   auto t1 = Tensor::FromScalar<float>(1.0f);
//   auto t2 = Tensor::FromScalar<float>(2.0f);
//   
//   auto c1 = Const(graph, "c1", t1.handle(), TF_FLOAT);
//   auto c2 = Const(graph, "c2", t2.handle(), TF_FLOAT);
//   auto sum = Add(graph, "sum", c1, c2, TF_FLOAT);
//
// ============================================================================

#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <optional>
#include <stdexcept>
#include <cstdint>

extern "C" {
#include <tensorflow/c/c_api.h>
}

#include "tf_wrap/graph.hpp"
#include "tf_wrap/tensor.hpp"
#include "tf_wrap/status.hpp"

namespace tf_wrap {
namespace ops {

// ============================================================================
// Op Result - Wrapper for operation outputs
// ============================================================================

/// Result of an operation - holds the TF_Operation* and provides output access
class OpResult {
public:
    explicit OpResult(TF_Operation* op) : op_(op) {
        if (!op_) throw std::runtime_error("OpResult: null operation");
    }
    
    /// Get the underlying operation
    [[nodiscard]] TF_Operation* op() const noexcept { return op_; }
    
    /// Get output at index (default 0)
    [[nodiscard]] TF_Output output(int index = 0) const noexcept {
        return TF_Output{op_, index};
    }
    
    /// Implicit conversion to TF_Output (for output 0)
    operator TF_Output() const noexcept { return output(0); }
    
    /// Get number of outputs
    [[nodiscard]] int num_outputs() const noexcept {
        return TF_OperationNumOutputs(op_);
    }
    
    /// Get operation name
    [[nodiscard]] std::string name() const {
        return TF_OperationName(op_);
    }

private:
    TF_Operation* op_;
};


// ============================================================================
// Math Operations
// ============================================================================

/// Returns x + y element-wise

[[nodiscard]] inline OpResult Add(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Add", std::string(name))
        .AddInput(x)
        .AddInput(y)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns x + y element-wise (with broadcasting)

[[nodiscard]] inline OpResult AddV2(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("AddV2", std::string(name))
        .AddInput(x)
        .AddInput(y)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns x - y element-wise

[[nodiscard]] inline OpResult Sub(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Sub", std::string(name))
        .AddInput(x)
        .AddInput(y)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns x * y element-wise

[[nodiscard]] inline OpResult Mul(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Mul", std::string(name))
        .AddInput(x)
        .AddInput(y)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns x / y element-wise

[[nodiscard]] inline OpResult Div(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Div", std::string(name))
        .AddInput(x)
        .AddInput(y)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns x / y element-wise for real types

[[nodiscard]] inline OpResult RealDiv(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("RealDiv", std::string(name))
        .AddInput(x)
        .AddInput(y)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns floor(x / y) element-wise

[[nodiscard]] inline OpResult FloorDiv(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("FloorDiv", std::string(name))
        .AddInput(x)
        .AddInput(y)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns x % y element-wise

[[nodiscard]] inline OpResult Mod(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Mod", std::string(name))
        .AddInput(x)
        .AddInput(y)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns x^y element-wise

[[nodiscard]] inline OpResult Pow(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Pow", std::string(name))
        .AddInput(x)
        .AddInput(y)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns max(x, y) element-wise

[[nodiscard]] inline OpResult Maximum(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Maximum", std::string(name))
        .AddInput(x)
        .AddInput(y)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns min(x, y) element-wise

[[nodiscard]] inline OpResult Minimum(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Minimum", std::string(name))
        .AddInput(x)
        .AddInput(y)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns -x element-wise

[[nodiscard]] inline OpResult Neg(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Neg", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns |x| element-wise

[[nodiscard]] inline OpResult Abs(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Abs", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns sign of x element-wise

[[nodiscard]] inline OpResult Sign(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Sign", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns 1/x element-wise

[[nodiscard]] inline OpResult Reciprocal(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Reciprocal", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns x^2 element-wise

[[nodiscard]] inline OpResult Square(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Square", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns sqrt(x) element-wise

[[nodiscard]] inline OpResult Sqrt(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Sqrt", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns 1/sqrt(x) element-wise

[[nodiscard]] inline OpResult Rsqrt(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Rsqrt", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns e^x element-wise

[[nodiscard]] inline OpResult Exp(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Exp", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns e^x - 1 element-wise

[[nodiscard]] inline OpResult Expm1(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Expm1", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns ln(x) element-wise

[[nodiscard]] inline OpResult Log(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Log", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns ln(1 + x) element-wise

[[nodiscard]] inline OpResult Log1p(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Log1p", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns sin(x) element-wise

[[nodiscard]] inline OpResult Sin(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Sin", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns cos(x) element-wise

[[nodiscard]] inline OpResult Cos(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Cos", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns tan(x) element-wise

[[nodiscard]] inline OpResult Tan(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Tan", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns asin(x) element-wise

[[nodiscard]] inline OpResult Asin(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Asin", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns acos(x) element-wise

[[nodiscard]] inline OpResult Acos(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Acos", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns atan(x) element-wise

[[nodiscard]] inline OpResult Atan(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Atan", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns sinh(x) element-wise

[[nodiscard]] inline OpResult Sinh(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Sinh", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns cosh(x) element-wise

[[nodiscard]] inline OpResult Cosh(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Cosh", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns tanh(x) element-wise

[[nodiscard]] inline OpResult Tanh(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Tanh", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns ceil(x) element-wise

[[nodiscard]] inline OpResult Ceil(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Ceil", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns floor(x) element-wise

[[nodiscard]] inline OpResult Floor(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Floor", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns round(x) element-wise

[[nodiscard]] inline OpResult Round(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Round", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns round to nearest integer element-wise

[[nodiscard]] inline OpResult Rint(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Rint", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns 1/(1+e^(-x)) element-wise

[[nodiscard]] inline OpResult Sigmoid(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Sigmoid", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}


// ============================================================================
// Matrix Operations
// ============================================================================

/// Matrix multiplication

[[nodiscard]] inline OpResult MatMul(
    Graph& graph,
    std::string_view name,
    TF_Output a,
    TF_Output b,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("MatMul", std::string(name))
        .AddInput(a)
        .AddInput(b)
        .SetAttrType("T", T)
        .Finish());
}

/// Batched matrix multiplication

[[nodiscard]] inline OpResult BatchMatMul(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("BatchMatMul", std::string(name))
        .AddInput(x)
        .AddInput(y)
        .SetAttrType("T", T)
        .Finish());
}

/// Batched matrix multiplication with broadcasting

[[nodiscard]] inline OpResult BatchMatMulV2(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("BatchMatMulV2", std::string(name))
        .AddInput(x)
        .AddInput(y)
        .SetAttrType("T", T)
        .Finish());
}

/// Permutes dimensions according to perm

[[nodiscard]] inline OpResult Transpose(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output perm,
    TF_DataType T,
    TF_DataType Tperm) {
    return OpResult(
        graph.NewOperation("Transpose", std::string(name))
        .AddInput(x)
        .AddInput(perm)
        .SetAttrType("T", T)
        .SetAttrType("Tperm", Tperm)
        .Finish());
}

/// Matrix inverse

[[nodiscard]] inline OpResult MatrixInverse(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("MatrixInverse", std::string(name))
        .AddInput(input)
        .SetAttrType("T", T)
        .Finish());
}

/// Matrix determinant

[[nodiscard]] inline OpResult MatrixDeterminant(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("MatrixDeterminant", std::string(name))
        .AddInput(input)
        .SetAttrType("T", T)
        .Finish());
}

/// Cholesky decomposition

[[nodiscard]] inline OpResult Cholesky(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Cholesky", std::string(name))
        .AddInput(input)
        .SetAttrType("T", T)
        .Finish());
}

/// QR decomposition

[[nodiscard]] inline OpResult Qr(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Qr", std::string(name))
        .AddInput(input)
        .SetAttrType("T", T)
        .Finish());
}

/// SVD decomposition

[[nodiscard]] inline OpResult Svd(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Svd", std::string(name))
        .AddInput(input)
        .SetAttrType("T", T)
        .Finish());
}

/// Einstein summation

[[nodiscard]] inline OpResult Einsum(
    Graph& graph,
    std::string_view name,
    std::span<const TF_Output> inputs,
    std::string_view equation,
    TF_DataType T,
    int64_t N) {
    return OpResult(
        graph.NewOperation("Einsum", std::string(name))
        .AddInputList(inputs)
        .SetAttrString("equation", equation)
        .SetAttrType("T", T)
        .SetAttrInt("N", N)
        .Finish());
}


// ============================================================================
// Reduction Operations
// ============================================================================

/// Sum along axis

[[nodiscard]] inline OpResult Sum(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output axis,
    TF_DataType T,
    TF_DataType Tidx) {
    return OpResult(
        graph.NewOperation("Sum", std::string(name))
        .AddInput(input)
        .AddInput(axis)
        .SetAttrType("T", T)
        .SetAttrType("Tidx", Tidx)
        .Finish());
}

/// Product along axis

[[nodiscard]] inline OpResult Prod(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output axis,
    TF_DataType T,
    TF_DataType Tidx) {
    return OpResult(
        graph.NewOperation("Prod", std::string(name))
        .AddInput(input)
        .AddInput(axis)
        .SetAttrType("T", T)
        .SetAttrType("Tidx", Tidx)
        .Finish());
}

/// Mean along axis

[[nodiscard]] inline OpResult Mean(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output axis,
    TF_DataType T,
    TF_DataType Tidx) {
    return OpResult(
        graph.NewOperation("Mean", std::string(name))
        .AddInput(input)
        .AddInput(axis)
        .SetAttrType("T", T)
        .SetAttrType("Tidx", Tidx)
        .Finish());
}

/// Maximum along axis

[[nodiscard]] inline OpResult Max(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output axis,
    TF_DataType T,
    TF_DataType Tidx) {
    return OpResult(
        graph.NewOperation("Max", std::string(name))
        .AddInput(input)
        .AddInput(axis)
        .SetAttrType("T", T)
        .SetAttrType("Tidx", Tidx)
        .Finish());
}

/// Minimum along axis

[[nodiscard]] inline OpResult Min(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output axis,
    TF_DataType T,
    TF_DataType Tidx) {
    return OpResult(
        graph.NewOperation("Min", std::string(name))
        .AddInput(input)
        .AddInput(axis)
        .SetAttrType("T", T)
        .SetAttrType("Tidx", Tidx)
        .Finish());
}

/// Logical AND along axis

[[nodiscard]] inline OpResult All(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output axis,
    TF_DataType Tidx) {
    return OpResult(
        graph.NewOperation("All", std::string(name))
        .AddInput(input)
        .AddInput(axis)
        .SetAttrType("Tidx", Tidx)
        .Finish());
}

/// Logical OR along axis

[[nodiscard]] inline OpResult Any(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output axis,
    TF_DataType Tidx) {
    return OpResult(
        graph.NewOperation("Any", std::string(name))
        .AddInput(input)
        .AddInput(axis)
        .SetAttrType("Tidx", Tidx)
        .Finish());
}

/// Index of maximum along axis

[[nodiscard]] inline OpResult ArgMax(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output axis,
    TF_DataType T,
    TF_DataType Tidx,
    TF_DataType output_type) {
    return OpResult(
        graph.NewOperation("ArgMax", std::string(name))
        .AddInput(input)
        .AddInput(axis)
        .SetAttrType("T", T)
        .SetAttrType("Tidx", Tidx)
        .SetAttrType("output_type", output_type)
        .Finish());
}

/// Index of minimum along axis

[[nodiscard]] inline OpResult ArgMin(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output axis,
    TF_DataType T,
    TF_DataType Tidx,
    TF_DataType output_type) {
    return OpResult(
        graph.NewOperation("ArgMin", std::string(name))
        .AddInput(input)
        .AddInput(axis)
        .SetAttrType("T", T)
        .SetAttrType("Tidx", Tidx)
        .SetAttrType("output_type", output_type)
        .Finish());
}


// ============================================================================
// Comparison Operations
// ============================================================================

/// Returns x == y element-wise

[[nodiscard]] inline OpResult Equal(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Equal", std::string(name))
        .AddInput(x)
        .AddInput(y)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns x != y element-wise

[[nodiscard]] inline OpResult NotEqual(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("NotEqual", std::string(name))
        .AddInput(x)
        .AddInput(y)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns x < y element-wise

[[nodiscard]] inline OpResult Less(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Less", std::string(name))
        .AddInput(x)
        .AddInput(y)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns x <= y element-wise

[[nodiscard]] inline OpResult LessEqual(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("LessEqual", std::string(name))
        .AddInput(x)
        .AddInput(y)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns x > y element-wise

[[nodiscard]] inline OpResult Greater(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Greater", std::string(name))
        .AddInput(x)
        .AddInput(y)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns x >= y element-wise

[[nodiscard]] inline OpResult GreaterEqual(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("GreaterEqual", std::string(name))
        .AddInput(x)
        .AddInput(y)
        .SetAttrType("T", T)
        .Finish());
}

/// Returns x AND y element-wise

[[nodiscard]] inline OpResult LogicalAnd(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y) {
    return OpResult(
        graph.NewOperation("LogicalAnd", std::string(name))
        .AddInput(x)
        .AddInput(y)
        .Finish());
}

/// Returns x OR y element-wise

[[nodiscard]] inline OpResult LogicalOr(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y) {
    return OpResult(
        graph.NewOperation("LogicalOr", std::string(name))
        .AddInput(x)
        .AddInput(y)
        .Finish());
}

/// Returns NOT x element-wise

[[nodiscard]] inline OpResult LogicalNot(
    Graph& graph,
    std::string_view name,
    TF_Output x) {
    return OpResult(graph.NewOperation("LogicalNot", std::string(name)).AddInput(x).Finish());
}


// ============================================================================
// Neural Network Operations
// ============================================================================

/// ReLU activation: max(0, x)

[[nodiscard]] inline OpResult Relu(
    Graph& graph,
    std::string_view name,
    TF_Output features,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Relu", std::string(name))
        .AddInput(features)
        .SetAttrType("T", T)
        .Finish());
}

/// ReLU6 activation: min(max(0, x), 6)

[[nodiscard]] inline OpResult Relu6(
    Graph& graph,
    std::string_view name,
    TF_Output features,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Relu6", std::string(name))
        .AddInput(features)
        .SetAttrType("T", T)
        .Finish());
}

/// Leaky ReLU activation

[[nodiscard]] inline OpResult LeakyRelu(
    Graph& graph,
    std::string_view name,
    TF_Output features,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("LeakyRelu", std::string(name))
        .AddInput(features)
        .SetAttrType("T", T)
        .Finish());
}

/// ELU activation

[[nodiscard]] inline OpResult Elu(
    Graph& graph,
    std::string_view name,
    TF_Output features,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Elu", std::string(name))
        .AddInput(features)
        .SetAttrType("T", T)
        .Finish());
}

/// SELU activation

[[nodiscard]] inline OpResult Selu(
    Graph& graph,
    std::string_view name,
    TF_Output features,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Selu", std::string(name))
        .AddInput(features)
        .SetAttrType("T", T)
        .Finish());
}

/// Softmax activation

[[nodiscard]] inline OpResult Softmax(
    Graph& graph,
    std::string_view name,
    TF_Output logits,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Softmax", std::string(name))
        .AddInput(logits)
        .SetAttrType("T", T)
        .Finish());
}

/// Log-softmax activation

[[nodiscard]] inline OpResult LogSoftmax(
    Graph& graph,
    std::string_view name,
    TF_Output logits,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("LogSoftmax", std::string(name))
        .AddInput(logits)
        .SetAttrType("T", T)
        .Finish());
}

/// Softplus activation: ln(1 + e^x)

[[nodiscard]] inline OpResult Softplus(
    Graph& graph,
    std::string_view name,
    TF_Output features,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Softplus", std::string(name))
        .AddInput(features)
        .SetAttrType("T", T)
        .Finish());
}

/// Softsign activation: x / (|x| + 1)

[[nodiscard]] inline OpResult Softsign(
    Graph& graph,
    std::string_view name,
    TF_Output features,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Softsign", std::string(name))
        .AddInput(features)
        .SetAttrType("T", T)
        .Finish());
}

/// Adds bias to value

[[nodiscard]] inline OpResult BiasAdd(
    Graph& graph,
    std::string_view name,
    TF_Output value,
    TF_Output bias,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("BiasAdd", std::string(name))
        .AddInput(value)
        .AddInput(bias)
        .SetAttrType("T", T)
        .Finish());
}

/// 2D convolution

[[nodiscard]] inline OpResult Conv2D(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output filter,
    TF_DataType T,
    std::span<const int64_t> strides,
    std::string_view padding) {
    return OpResult(
        graph.NewOperation("Conv2D", std::string(name))
        .AddInput(input)
        .AddInput(filter)
        .SetAttrType("T", T)
        .SetAttrIntList("strides", strides)
        .SetAttrString("padding", padding)
        .Finish());
}

/// Conv2D input gradient (transposed convolution)

[[nodiscard]] inline OpResult Conv2DBackpropInput(
    Graph& graph,
    std::string_view name,
    TF_Output input_sizes,
    TF_Output filter,
    TF_Output out_backprop,
    TF_DataType T,
    std::span<const int64_t> strides,
    std::string_view padding) {
    return OpResult(
        graph.NewOperation("Conv2DBackpropInput", std::string(name))
        .AddInput(input_sizes)
        .AddInput(filter)
        .AddInput(out_backprop)
        .SetAttrType("T", T)
        .SetAttrIntList("strides", strides)
        .SetAttrString("padding", padding)
        .Finish());
}

/// Depthwise 2D convolution

[[nodiscard]] inline OpResult DepthwiseConv2dNative(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output filter,
    TF_DataType T,
    std::span<const int64_t> strides,
    std::string_view padding) {
    return OpResult(
        graph.NewOperation("DepthwiseConv2dNative", std::string(name))
        .AddInput(input)
        .AddInput(filter)
        .SetAttrType("T", T)
        .SetAttrIntList("strides", strides)
        .SetAttrString("padding", padding)
        .Finish());
}

/// Max pooling

[[nodiscard]] inline OpResult MaxPool(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_DataType T,
    std::span<const int64_t> ksize,
    std::span<const int64_t> strides,
    std::string_view padding) {
    return OpResult(
        graph.NewOperation("MaxPool", std::string(name))
        .AddInput(input)
        .SetAttrType("T", T)
        .SetAttrIntList("ksize", ksize)
        .SetAttrIntList("strides", strides)
        .SetAttrString("padding", padding)
        .Finish());
}

/// Average pooling

[[nodiscard]] inline OpResult AvgPool(
    Graph& graph,
    std::string_view name,
    TF_Output value,
    TF_DataType T,
    std::span<const int64_t> ksize,
    std::span<const int64_t> strides,
    std::string_view padding) {
    return OpResult(
        graph.NewOperation("AvgPool", std::string(name))
        .AddInput(value)
        .SetAttrType("T", T)
        .SetAttrIntList("ksize", ksize)
        .SetAttrIntList("strides", strides)
        .SetAttrString("padding", padding)
        .Finish());
}

/// 3D max pooling

[[nodiscard]] inline OpResult MaxPool3D(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_DataType T,
    std::span<const int64_t> ksize,
    std::span<const int64_t> strides,
    std::string_view padding) {
    return OpResult(
        graph.NewOperation("MaxPool3D", std::string(name))
        .AddInput(input)
        .SetAttrType("T", T)
        .SetAttrIntList("ksize", ksize)
        .SetAttrIntList("strides", strides)
        .SetAttrString("padding", padding)
        .Finish());
}

/// 3D average pooling

[[nodiscard]] inline OpResult AvgPool3D(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_DataType T,
    std::span<const int64_t> ksize,
    std::span<const int64_t> strides,
    std::string_view padding) {
    return OpResult(
        graph.NewOperation("AvgPool3D", std::string(name))
        .AddInput(input)
        .SetAttrType("T", T)
        .SetAttrIntList("ksize", ksize)
        .SetAttrIntList("strides", strides)
        .SetAttrString("padding", padding)
        .Finish());
}

/// Fused batch normalization

[[nodiscard]] inline OpResult FusedBatchNorm(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output scale,
    TF_Output offset,
    TF_Output mean,
    TF_Output variance,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("FusedBatchNorm", std::string(name))
        .AddInput(x)
        .AddInput(scale)
        .AddInput(offset)
        .AddInput(mean)
        .AddInput(variance)
        .SetAttrType("T", T)
        .Finish());
}

/// Fused batch normalization V3

[[nodiscard]] inline OpResult FusedBatchNormV3(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output scale,
    TF_Output offset,
    TF_Output mean,
    TF_Output variance,
    TF_DataType T,
    TF_DataType U) {
    return OpResult(
        graph.NewOperation("FusedBatchNormV3", std::string(name))
        .AddInput(x)
        .AddInput(scale)
        .AddInput(offset)
        .AddInput(mean)
        .AddInput(variance)
        .SetAttrType("T", T)
        .SetAttrType("U", U)
        .Finish());
}

/// Local response normalization

[[nodiscard]] inline OpResult LRN(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("LRN", std::string(name))
        .AddInput(input)
        .SetAttrType("T", T)
        .Finish());
}

// NOTE: "Dropout" op does not exist in TensorFlow C API.
// Dropout is typically implemented using primitive ops:
//
// Example dropout implementation (keep_prob = 1 - rate):
//   1. Generate random values: RandomUniform(graph, "rand", shape, TF_FLOAT)
//   2. Create mask: Greater(graph, "mask", rand_output, rate_output, TF_FLOAT)
//   3. Cast mask to float: Cast(graph, "mask_float", mask_output, TF_BOOL, TF_FLOAT)
//   4. Scale factor: RealDiv(graph, "scale", one_output, keep_prob_output, TF_FLOAT)
//   5. Apply: Mul(graph, "dropped", Mul(..., x, mask_float), scale, TF_FLOAT)
//
// Or use Select() with a random boolean mask and zeros.

/// Softmax cross entropy loss

[[nodiscard]] inline OpResult SoftmaxCrossEntropyWithLogits(
    Graph& graph,
    std::string_view name,
    TF_Output features,
    TF_Output labels,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("SoftmaxCrossEntropyWithLogits", std::string(name))
        .AddInput(features)
        .AddInput(labels)
        .SetAttrType("T", T)
        .Finish());
}

/// Sparse softmax cross entropy loss

[[nodiscard]] inline OpResult SparseSoftmaxCrossEntropyWithLogits(
    Graph& graph,
    std::string_view name,
    TF_Output features,
    TF_Output labels,
    TF_DataType T,
    TF_DataType Tlabels) {
    return OpResult(
        graph.NewOperation("SparseSoftmaxCrossEntropyWithLogits", std::string(name))
        .AddInput(features)
        .AddInput(labels)
        .SetAttrType("T", T)
        .SetAttrType("Tlabels", Tlabels)
        .Finish());
}


// ============================================================================
// Array Operations
// ============================================================================

/// Constant tensor

[[nodiscard]] inline OpResult Const(
    Graph& graph,
    std::string_view name,
    TF_Tensor* value,
    TF_DataType dtype) {
    return OpResult(
        graph.NewOperation("Const", std::string(name))
        .SetAttrTensor("value", value)
        .SetAttrType("dtype", dtype)
        .Finish());
}

/// Placeholder for feeding data

[[nodiscard]] inline OpResult Placeholder(
    Graph& graph,
    std::string_view name,
    TF_DataType dtype,
    std::span<const int64_t> shape) {
    return OpResult(
        graph.NewOperation("Placeholder", std::string(name))
        .SetAttrType("dtype", dtype)
        .SetAttrShape("shape", shape)
        .Finish());
}

/// Identity function

[[nodiscard]] inline OpResult Identity(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Identity", std::string(name))
        .AddInput(input)
        .SetAttrType("T", T)
        .Finish());
}

/// Identity for multiple tensors

[[nodiscard]] inline OpResult IdentityN(
    Graph& graph,
    std::string_view name,
    std::span<const TF_Output> input) {
    return OpResult(graph.NewOperation("IdentityN", std::string(name)).AddInputList(input).Finish());
}

/// Reshape tensor

[[nodiscard]] inline OpResult Reshape(
    Graph& graph,
    std::string_view name,
    TF_Output tensor,
    TF_Output shape,
    TF_DataType T,
    TF_DataType Tshape) {
    return OpResult(
        graph.NewOperation("Reshape", std::string(name))
        .AddInput(tensor)
        .AddInput(shape)
        .SetAttrType("T", T)
        .SetAttrType("Tshape", Tshape)
        .Finish());
}

/// Remove size-1 dimensions

[[nodiscard]] inline OpResult Squeeze(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Squeeze", std::string(name))
        .AddInput(input)
        .SetAttrType("T", T)
        .Finish());
}

/// Insert dimension of size 1

[[nodiscard]] inline OpResult ExpandDims(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output axis,
    TF_DataType T,
    TF_DataType Tdim) {
    return OpResult(
        graph.NewOperation("ExpandDims", std::string(name))
        .AddInput(input)
        .AddInput(axis)
        .SetAttrType("T", T)
        .SetAttrType("Tdim", Tdim)
        .Finish());
}

/// Concatenate tensors

[[nodiscard]] inline OpResult Concat(
    Graph& graph,
    std::string_view name,
    TF_Output concat_dim,
    std::span<const TF_Output> values,
    TF_DataType T,
    int64_t N) {
    return OpResult(
        graph.NewOperation("Concat", std::string(name))
        .AddInput(concat_dim)
        .AddInputList(values)
        .SetAttrType("T", T)
        .SetAttrInt("N", N)
        .Finish());
}

/// Concatenate tensors V2

[[nodiscard]] inline OpResult ConcatV2(
    Graph& graph,
    std::string_view name,
    std::span<const TF_Output> values,
    TF_Output axis,
    TF_DataType T,
    int64_t N,
    TF_DataType Tidx) {
    return OpResult(
        graph.NewOperation("ConcatV2", std::string(name))
        .AddInputList(values)
        .AddInput(axis)
        .SetAttrType("T", T)
        .SetAttrInt("N", N)
        .SetAttrType("Tidx", Tidx)
        .Finish());
}

/// Split tensor into subtensors

[[nodiscard]] inline OpResult Split(
    Graph& graph,
    std::string_view name,
    TF_Output split_dim,
    TF_Output value,
    TF_DataType T,
    int64_t num_split) {
    return OpResult(
        graph.NewOperation("Split", std::string(name))
        .AddInput(split_dim)
        .AddInput(value)
        .SetAttrType("T", T)
        .SetAttrInt("num_split", num_split)
        .Finish());
}

/// Split tensor with variable sizes

[[nodiscard]] inline OpResult SplitV(
    Graph& graph,
    std::string_view name,
    TF_Output value,
    TF_Output size_splits,
    TF_Output split_dim,
    TF_DataType T,
    int64_t num_split,
    TF_DataType Tlen) {
    return OpResult(
        graph.NewOperation("SplitV", std::string(name))
        .AddInput(value)
        .AddInput(size_splits)
        .AddInput(split_dim)
        .SetAttrType("T", T)
        .SetAttrInt("num_split", num_split)
        .SetAttrType("Tlen", Tlen)
        .Finish());
}

/// Slice from tensor

[[nodiscard]] inline OpResult Slice(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output begin,
    TF_Output size,
    TF_DataType T,
    TF_DataType Index) {
    return OpResult(
        graph.NewOperation("Slice", std::string(name))
        .AddInput(input)
        .AddInput(begin)
        .AddInput(size)
        .SetAttrType("T", T)
        .SetAttrType("Index", Index)
        .Finish());
}

/// Strided slice from tensor

[[nodiscard]] inline OpResult StridedSlice(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output begin,
    TF_Output end,
    TF_Output strides,
    TF_DataType T,
    TF_DataType Index) {
    return OpResult(
        graph.NewOperation("StridedSlice", std::string(name))
        .AddInput(input)
        .AddInput(begin)
        .AddInput(end)
        .AddInput(strides)
        .SetAttrType("T", T)
        .SetAttrType("Index", Index)
        .Finish());
}

/// Gather slices from tensor

[[nodiscard]] inline OpResult Gather(
    Graph& graph,
    std::string_view name,
    TF_Output params,
    TF_Output indices,
    TF_DataType Tparams,
    TF_DataType Tindices) {
    return OpResult(
        graph.NewOperation("Gather", std::string(name))
        .AddInput(params)
        .AddInput(indices)
        .SetAttrType("Tparams", Tparams)
        .SetAttrType("Tindices", Tindices)
        .Finish());
}

/// Gather slices with axis

[[nodiscard]] inline OpResult GatherV2(
    Graph& graph,
    std::string_view name,
    TF_Output params,
    TF_Output indices,
    TF_Output axis,
    TF_DataType Tparams,
    TF_DataType Tindices,
    TF_DataType Taxis) {
    return OpResult(
        graph.NewOperation("GatherV2", std::string(name))
        .AddInput(params)
        .AddInput(indices)
        .AddInput(axis)
        .SetAttrType("Tparams", Tparams)
        .SetAttrType("Tindices", Tindices)
        .SetAttrType("Taxis", Taxis)
        .Finish());
}

/// Gather slices with N-dimensional indices

[[nodiscard]] inline OpResult GatherNd(
    Graph& graph,
    std::string_view name,
    TF_Output params,
    TF_Output indices,
    TF_DataType Tparams,
    TF_DataType Tindices) {
    return OpResult(
        graph.NewOperation("GatherNd", std::string(name))
        .AddInput(params)
        .AddInput(indices)
        .SetAttrType("Tparams", Tparams)
        .SetAttrType("Tindices", Tindices)
        .Finish());
}

/// Scatter updates into tensor

[[nodiscard]] inline OpResult ScatterNd(
    Graph& graph,
    std::string_view name,
    TF_Output indices,
    TF_Output updates,
    TF_Output shape,
    TF_DataType T,
    TF_DataType Tindices) {
    return OpResult(
        graph.NewOperation("ScatterNd", std::string(name))
        .AddInput(indices)
        .AddInput(updates)
        .AddInput(shape)
        .SetAttrType("T", T)
        .SetAttrType("Tindices", Tindices)
        .Finish());
}

/// Tile tensor

[[nodiscard]] inline OpResult Tile(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output multiples,
    TF_DataType T,
    TF_DataType Tmultiples) {
    return OpResult(
        graph.NewOperation("Tile", std::string(name))
        .AddInput(input)
        .AddInput(multiples)
        .SetAttrType("T", T)
        .SetAttrType("Tmultiples", Tmultiples)
        .Finish());
}

/// Pad tensor with zeros

[[nodiscard]] inline OpResult Pad(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output paddings,
    TF_DataType T,
    TF_DataType Tpaddings) {
    return OpResult(
        graph.NewOperation("Pad", std::string(name))
        .AddInput(input)
        .AddInput(paddings)
        .SetAttrType("T", T)
        .SetAttrType("Tpaddings", Tpaddings)
        .Finish());
}

/// Pad tensor with constant value

[[nodiscard]] inline OpResult PadV2(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output paddings,
    TF_Output constant_values,
    TF_DataType T,
    TF_DataType Tpaddings) {
    return OpResult(
        graph.NewOperation("PadV2", std::string(name))
        .AddInput(input)
        .AddInput(paddings)
        .AddInput(constant_values)
        .SetAttrType("T", T)
        .SetAttrType("Tpaddings", Tpaddings)
        .Finish());
}

/// Pad tensor with mirrored values

[[nodiscard]] inline OpResult MirrorPad(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output paddings,
    TF_DataType T,
    TF_DataType Tpaddings,
    std::string_view mode) {
    return OpResult(
        graph.NewOperation("MirrorPad", std::string(name))
        .AddInput(input)
        .AddInput(paddings)
        .SetAttrType("T", T)
        .SetAttrType("Tpaddings", Tpaddings)
        .SetAttrString("mode", mode)
        .Finish());
}

/// Reverse tensor along axes

[[nodiscard]] inline OpResult ReverseV2(
    Graph& graph,
    std::string_view name,
    TF_Output tensor,
    TF_Output axis,
    TF_DataType T,
    TF_DataType Tidx) {
    return OpResult(
        graph.NewOperation("ReverseV2", std::string(name))
        .AddInput(tensor)
        .AddInput(axis)
        .SetAttrType("T", T)
        .SetAttrType("Tidx", Tidx)
        .Finish());
}

/// Stack tensors along axis (pack)

[[nodiscard]] inline OpResult Pack(
    Graph& graph,
    std::string_view name,
    std::span<const TF_Output> values,
    TF_DataType T,
    int64_t N) {
    return OpResult(
        graph.NewOperation("Pack", std::string(name))
        .AddInputList(values)
        .SetAttrType("T", T)
        .SetAttrInt("N", N)
        .Finish());
}

/// Unstack tensor along axis (unpack)

[[nodiscard]] inline OpResult Unpack(
    Graph& graph,
    std::string_view name,
    TF_Output value,
    TF_DataType T,
    int64_t num) {
    return OpResult(
        graph.NewOperation("Unpack", std::string(name))
        .AddInput(value)
        .SetAttrType("T", T)
        .SetAttrInt("num", num)
        .Finish());
}

/// Get tensor shape

[[nodiscard]] inline OpResult Shape(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_DataType T,
    TF_DataType out_type) {
    return OpResult(
        graph.NewOperation("Shape", std::string(name))
        .AddInput(input)
        .SetAttrType("T", T)
        .SetAttrType("out_type", out_type)
        .Finish());
}

/// Get shapes of multiple tensors

[[nodiscard]] inline OpResult ShapeN(
    Graph& graph,
    std::string_view name,
    std::span<const TF_Output> input,
    TF_DataType T,
    int64_t N,
    TF_DataType out_type) {
    return OpResult(
        graph.NewOperation("ShapeN", std::string(name))
        .AddInputList(input)
        .SetAttrType("T", T)
        .SetAttrInt("N", N)
        .SetAttrType("out_type", out_type)
        .Finish());
}

/// Get tensor rank

[[nodiscard]] inline OpResult Rank(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Rank", std::string(name))
        .AddInput(input)
        .SetAttrType("T", T)
        .Finish());
}

/// Get number of elements

[[nodiscard]] inline OpResult Size(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_DataType T,
    TF_DataType out_type) {
    return OpResult(
        graph.NewOperation("Size", std::string(name))
        .AddInput(input)
        .SetAttrType("T", T)
        .SetAttrType("out_type", out_type)
        .Finish());
}

/// Fill tensor with scalar value

[[nodiscard]] inline OpResult Fill(
    Graph& graph,
    std::string_view name,
    TF_Output dims,
    TF_Output value,
    TF_DataType T,
    TF_DataType index_type) {
    return OpResult(
        graph.NewOperation("Fill", std::string(name))
        .AddInput(dims)
        .AddInput(value)
        .SetAttrType("T", T)
        .SetAttrType("index_type", index_type)
        .Finish());
}

// NOTE: "Zeros" op does not exist in TensorFlow C API.
// To create a tensor of zeros, use one of these alternatives:
//
// 1. Fill() with a zero constant:
//    auto zero = Const(graph, "zero", zero_tensor.handle(), TF_FLOAT);
//    auto zeros = Fill(graph, "zeros", shape, zero.output(), TF_FLOAT, TF_INT32);
//
// 2. ZerosLike() to match another tensor's shape:
//    auto zeros = ZerosLike(graph, "zeros", other_tensor_output, TF_FLOAT);

/// Create tensor of zeros with same shape

[[nodiscard]] inline OpResult ZerosLike(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("ZerosLike", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Create tensor of ones with same shape

[[nodiscard]] inline OpResult OnesLike(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("OnesLike", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Create range [start, limit) with delta step

[[nodiscard]] inline OpResult Range(
    Graph& graph,
    std::string_view name,
    TF_Output start,
    TF_Output limit,
    TF_Output delta,
    TF_DataType Tidx) {
    return OpResult(
        graph.NewOperation("Range", std::string(name))
        .AddInput(start)
        .AddInput(limit)
        .AddInput(delta)
        .SetAttrType("Tidx", Tidx)
        .Finish());
}

/// Create linearly spaced values

[[nodiscard]] inline OpResult LinSpace(
    Graph& graph,
    std::string_view name,
    TF_Output start,
    TF_Output stop,
    TF_Output num,
    TF_DataType T,
    TF_DataType Tidx) {
    return OpResult(
        graph.NewOperation("LinSpace", std::string(name))
        .AddInput(start)
        .AddInput(stop)
        .AddInput(num)
        .SetAttrType("T", T)
        .SetAttrType("Tidx", Tidx)
        .Finish());
}

/// Broadcast to shape

[[nodiscard]] inline OpResult BroadcastTo(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output shape,
    TF_DataType T,
    TF_DataType Tidx) {
    return OpResult(
        graph.NewOperation("BroadcastTo", std::string(name))
        .AddInput(input)
        .AddInput(shape)
        .SetAttrType("T", T)
        .SetAttrType("Tidx", Tidx)
        .Finish());
}

/// Returns indices of true elements

[[nodiscard]] inline OpResult Where(
    Graph& graph,
    std::string_view name,
    TF_Output condition,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Where", std::string(name))
        .AddInput(condition)
        .SetAttrType("T", T)
        .Finish());
}

/// Select elements based on condition

[[nodiscard]] inline OpResult SelectV2(
    Graph& graph,
    std::string_view name,
    TF_Output condition,
    TF_Output t,
    TF_Output e,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("SelectV2", std::string(name))
        .AddInput(condition)
        .AddInput(t)
        .AddInput(e)
        .SetAttrType("T", T)
        .Finish());
}


// ============================================================================
// Cast and Type Operations
// ============================================================================

/// Cast tensor to different type

[[nodiscard]] inline OpResult Cast(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType SrcT,
    TF_DataType DstT) {
    return OpResult(
        graph.NewOperation("Cast", std::string(name))
        .AddInput(x)
        .SetAttrType("SrcT", SrcT)
        .SetAttrType("DstT", DstT)
        .Finish());
}

/// Bitcast without copying data

[[nodiscard]] inline OpResult Bitcast(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_DataType T,
    TF_DataType type) {
    return OpResult(
        graph.NewOperation("Bitcast", std::string(name))
        .AddInput(input)
        .SetAttrType("T", T)
        .SetAttrType("type", type)
        .Finish());
}

/// Check for NaN/Inf values

[[nodiscard]] inline OpResult CheckNumerics(
    Graph& graph,
    std::string_view name,
    TF_Output tensor,
    TF_DataType T,
    std::string_view message) {
    return OpResult(
        graph.NewOperation("CheckNumerics", std::string(name))
        .AddInput(tensor)
        .SetAttrType("T", T)
        .SetAttrString("message", message)
        .Finish());
}


// ============================================================================
// Random Operations
// ============================================================================

/// Uniform random values [0, 1)

[[nodiscard]] inline OpResult RandomUniform(
    Graph& graph,
    std::string_view name,
    TF_Output shape,
    TF_DataType dtype,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("RandomUniform", std::string(name))
        .AddInput(shape)
        .SetAttrType("dtype", dtype)
        .SetAttrType("T", T)
        .Finish());
}

/// Standard normal random values

[[nodiscard]] inline OpResult RandomStandardNormal(
    Graph& graph,
    std::string_view name,
    TF_Output shape,
    TF_DataType dtype,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("RandomStandardNormal", std::string(name))
        .AddInput(shape)
        .SetAttrType("dtype", dtype)
        .SetAttrType("T", T)
        .Finish());
}

/// Truncated normal random values

[[nodiscard]] inline OpResult TruncatedNormal(
    Graph& graph,
    std::string_view name,
    TF_Output shape,
    TF_DataType dtype,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("TruncatedNormal", std::string(name))
        .AddInput(shape)
        .SetAttrType("dtype", dtype)
        .SetAttrType("T", T)
        .Finish());
}

/// Randomly shuffle tensor

[[nodiscard]] inline OpResult RandomShuffle(
    Graph& graph,
    std::string_view name,
    TF_Output value,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("RandomShuffle", std::string(name))
        .AddInput(value)
        .SetAttrType("T", T)
        .Finish());
}

/// Draw samples from multinomial distribution

[[nodiscard]] inline OpResult Multinomial(
    Graph& graph,
    std::string_view name,
    TF_Output logits,
    TF_Output num_samples,
    TF_DataType T,
    TF_DataType output_dtype) {
    return OpResult(
        graph.NewOperation("Multinomial", std::string(name))
        .AddInput(logits)
        .AddInput(num_samples)
        .SetAttrType("T", T)
        .SetAttrType("output_dtype", output_dtype)
        .Finish());
}


// ============================================================================
// Image Operations
// ============================================================================

/// Resize images using bilinear interpolation

[[nodiscard]] inline OpResult ResizeBilinear(
    Graph& graph,
    std::string_view name,
    TF_Output images,
    TF_Output size,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("ResizeBilinear", std::string(name))
        .AddInput(images)
        .AddInput(size)
        .SetAttrType("T", T)
        .Finish());
}

/// Resize images using nearest neighbor

[[nodiscard]] inline OpResult ResizeNearestNeighbor(
    Graph& graph,
    std::string_view name,
    TF_Output images,
    TF_Output size,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("ResizeNearestNeighbor", std::string(name))
        .AddInput(images)
        .AddInput(size)
        .SetAttrType("T", T)
        .Finish());
}

/// Resize images using bicubic interpolation

[[nodiscard]] inline OpResult ResizeBicubic(
    Graph& graph,
    std::string_view name,
    TF_Output images,
    TF_Output size,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("ResizeBicubic", std::string(name))
        .AddInput(images)
        .AddInput(size)
        .SetAttrType("T", T)
        .Finish());
}

/// Extract and resize crops from images

[[nodiscard]] inline OpResult CropAndResize(
    Graph& graph,
    std::string_view name,
    TF_Output image,
    TF_Output boxes,
    TF_Output box_ind,
    TF_Output crop_size,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("CropAndResize", std::string(name))
        .AddInput(image)
        .AddInput(boxes)
        .AddInput(box_ind)
        .AddInput(crop_size)
        .SetAttrType("T", T)
        .Finish());
}

/// Non-maximum suppression for object detection

[[nodiscard]] inline OpResult NonMaxSuppression(
    Graph& graph,
    std::string_view name,
    TF_Output boxes,
    TF_Output scores,
    TF_Output max_output_size) {
    return OpResult(
        graph.NewOperation("NonMaxSuppression", std::string(name))
        .AddInput(boxes)
        .AddInput(scores)
        .AddInput(max_output_size)
        .Finish());
}

/// Non-maximum suppression V3

[[nodiscard]] inline OpResult NonMaxSuppressionV3(
    Graph& graph,
    std::string_view name,
    TF_Output boxes,
    TF_Output scores,
    TF_Output max_output_size,
    TF_Output iou_threshold,
    TF_Output score_threshold) {
    return OpResult(
        graph.NewOperation("NonMaxSuppressionV3", std::string(name))
        .AddInput(boxes)
        .AddInput(scores)
        .AddInput(max_output_size)
        .AddInput(iou_threshold)
        .AddInput(score_threshold)
        .Finish());
}

/// Decode JPEG image

[[nodiscard]] inline OpResult DecodeJpeg(
    Graph& graph,
    std::string_view name,
    TF_Output contents) {
    return OpResult(graph.NewOperation("DecodeJpeg", std::string(name)).AddInput(contents).Finish());
}

/// Decode PNG image

[[nodiscard]] inline OpResult DecodePng(
    Graph& graph,
    std::string_view name,
    TF_Output contents,
    TF_DataType dtype) {
    return OpResult(
        graph.NewOperation("DecodePng", std::string(name))
        .AddInput(contents)
        .SetAttrType("dtype", dtype)
        .Finish());
}

/// Encode image as JPEG

[[nodiscard]] inline OpResult EncodeJpeg(
    Graph& graph,
    std::string_view name,
    TF_Output image) {
    return OpResult(graph.NewOperation("EncodeJpeg", std::string(name)).AddInput(image).Finish());
}

/// Encode image as PNG

[[nodiscard]] inline OpResult EncodePng(
    Graph& graph,
    std::string_view name,
    TF_Output image,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("EncodePng", std::string(name))
        .AddInput(image)
        .SetAttrType("T", T)
        .Finish());
}


// ============================================================================
// Control Flow Operations
// ============================================================================

/// No operation (placeholder)

[[nodiscard]] inline OpResult NoOp(
    Graph& graph,
    std::string_view name) {
    return OpResult(graph.NewOperation("NoOp", std::string(name)).Finish());
}

/// Stop gradient propagation

[[nodiscard]] inline OpResult StopGradient(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("StopGradient", std::string(name))
        .AddInput(input)
        .SetAttrType("T", T)
        .Finish());
}

/// Prevent gradient propagation with message

[[nodiscard]] inline OpResult PreventGradient(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("PreventGradient", std::string(name))
        .AddInput(input)
        .SetAttrType("T", T)
        .Finish());
}

/// Print tensor values for debugging

[[nodiscard]] inline OpResult Print(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    std::span<const TF_Output> data,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Print", std::string(name))
        .AddInput(input)
        .AddInputList(data)
        .SetAttrType("T", T)
        .Finish());
}

/// Assert condition is true

[[nodiscard]] inline OpResult Assert(
    Graph& graph,
    std::string_view name,
    TF_Output condition,
    std::span<const TF_Output> data) {
    return OpResult(
        graph.NewOperation("Assert", std::string(name))
        .AddInput(condition)
        .AddInputList(data)
        .Finish());
}


// ============================================================================
// String Operations
// ============================================================================

/// Join strings

[[nodiscard]] inline OpResult StringJoin(
    Graph& graph,
    std::string_view name,
    std::span<const TF_Output> inputs,
    int64_t N) {
    return OpResult(
        graph.NewOperation("StringJoin", std::string(name))
        .AddInputList(inputs)
        .SetAttrInt("N", N)
        .Finish());
}

/// Split strings

[[nodiscard]] inline OpResult StringSplit(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output delimiter) {
    return OpResult(
        graph.NewOperation("StringSplit", std::string(name))
        .AddInput(input)
        .AddInput(delimiter)
        .Finish());
}

/// Replace regex pattern

[[nodiscard]] inline OpResult RegexReplace(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output pattern,
    TF_Output rewrite) {
    return OpResult(
        graph.NewOperation("RegexReplace", std::string(name))
        .AddInput(input)
        .AddInput(pattern)
        .AddInput(rewrite)
        .Finish());
}


// ============================================================================
// File I/O Operations
// ============================================================================

/// Read entire file contents

[[nodiscard]] inline OpResult ReadFile(
    Graph& graph,
    std::string_view name,
    TF_Output filename) {
    return OpResult(graph.NewOperation("ReadFile", std::string(name)).AddInput(filename).Finish());
}

/// Write contents to file

[[nodiscard]] inline OpResult WriteFile(
    Graph& graph,
    std::string_view name,
    TF_Output filename,
    TF_Output contents) {
    return OpResult(
        graph.NewOperation("WriteFile", std::string(name))
        .AddInput(filename)
        .AddInput(contents)
        .Finish());
}

/// Find files matching pattern

[[nodiscard]] inline OpResult MatchingFiles(
    Graph& graph,
    std::string_view name,
    TF_Output pattern) {
    return OpResult(graph.NewOperation("MatchingFiles", std::string(name)).AddInput(pattern).Finish());
}


// ============================================================================
// Variable Operations
// ============================================================================

/// Create variable

[[nodiscard]] inline OpResult Variable(
    Graph& graph,
    std::string_view name,
    std::span<const int64_t> shape,
    TF_DataType dtype) {
    return OpResult(
        graph.NewOperation("Variable", std::string(name))
        .SetAttrShape("shape", shape)
        .SetAttrType("dtype", dtype)
        .Finish());
}

/// Create variable V2

[[nodiscard]] inline OpResult VariableV2(
    Graph& graph,
    std::string_view name,
    std::span<const int64_t> shape,
    TF_DataType dtype) {
    return OpResult(
        graph.NewOperation("VariableV2", std::string(name))
        .SetAttrShape("shape", shape)
        .SetAttrType("dtype", dtype)
        .Finish());
}

/// Create variable handle

[[nodiscard]] inline OpResult VarHandleOp(
    Graph& graph,
    std::string_view name,
    TF_DataType dtype,
    std::span<const int64_t> shape) {
    return OpResult(
        graph.NewOperation("VarHandleOp", std::string(name))
        .SetAttrType("dtype", dtype)
        .SetAttrShape("shape", shape)
        .Finish());
}

/// Read variable value

[[nodiscard]] inline OpResult ReadVariableOp(
    Graph& graph,
    std::string_view name,
    TF_Output resource,
    TF_DataType dtype) {
    return OpResult(
        graph.NewOperation("ReadVariableOp", std::string(name))
        .AddInput(resource)
        .SetAttrType("dtype", dtype)
        .Finish());
}

/// Assign value to variable

[[nodiscard]] inline OpResult AssignVariableOp(
    Graph& graph,
    std::string_view name,
    TF_Output resource,
    TF_Output value,
    TF_DataType dtype) {
    return OpResult(
        graph.NewOperation("AssignVariableOp", std::string(name))
        .AddInput(resource)
        .AddInput(value)
        .SetAttrType("dtype", dtype)
        .Finish());
}

/// Add value to variable

[[nodiscard]] inline OpResult AssignAddVariableOp(
    Graph& graph,
    std::string_view name,
    TF_Output resource,
    TF_Output value,
    TF_DataType dtype) {
    return OpResult(
        graph.NewOperation("AssignAddVariableOp", std::string(name))
        .AddInput(resource)
        .AddInput(value)
        .SetAttrType("dtype", dtype)
        .Finish());
}

/// Subtract value from variable

[[nodiscard]] inline OpResult AssignSubVariableOp(
    Graph& graph,
    std::string_view name,
    TF_Output resource,
    TF_Output value,
    TF_DataType dtype) {
    return OpResult(
        graph.NewOperation("AssignSubVariableOp", std::string(name))
        .AddInput(resource)
        .AddInput(value)
        .SetAttrType("dtype", dtype)
        .Finish());
}


} // namespace ops
} // namespace tf_wrap
