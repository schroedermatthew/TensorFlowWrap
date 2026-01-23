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
//   FastGraph graph;
//   auto t1 = FastTensor::FromScalar<float>(1.0f);
//   auto t2 = FastTensor::FromScalar<float>(2.0f);
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Add(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult AddV2(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Sub(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Mul(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Div(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult RealDiv(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult FloorDiv(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Mod(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Pow(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Maximum(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Minimum(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Neg(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Abs(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Sign(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Reciprocal(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Square(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Sqrt(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Rsqrt(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Exp(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Expm1(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Log(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Log1p(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Sin(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Cos(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Tan(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Asin(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Acos(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Atan(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Sinh(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Cosh(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Tanh(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Ceil(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Floor(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Round(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Rint(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Sigmoid(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult MatMul(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult BatchMatMul(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult BatchMatMulV2(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Transpose(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult MatrixInverse(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult MatrixDeterminant(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Cholesky(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Qr(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Svd(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Einsum(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Sum(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Prod(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Mean(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Max(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Min(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult All(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Any(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult ArgMax(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult ArgMin(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Equal(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult NotEqual(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Less(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult LessEqual(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Greater(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult GreaterEqual(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult LogicalAnd(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult LogicalOr(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult LogicalNot(
    Graph<GraphPolicy>& graph,
    std::string_view name,
    TF_Output x) {
    return OpResult(graph.NewOperation("LogicalNot", std::string(name)).AddInput(x).Finish());
}


// ============================================================================
// Neural Network Operations
// ============================================================================

/// ReLU activation: max(0, x)
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Relu(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Relu6(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult LeakyRelu(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Elu(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Selu(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Softmax(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult LogSoftmax(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Softplus(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Softsign(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult BiasAdd(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Conv2D(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Conv2DBackpropInput(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult DepthwiseConv2dNative(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult MaxPool(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult AvgPool(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult MaxPool3D(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult AvgPool3D(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult FusedBatchNorm(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult FusedBatchNormV3(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult LRN(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult SoftmaxCrossEntropyWithLogits(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult SparseSoftmaxCrossEntropyWithLogits(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Const(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Placeholder(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Identity(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult IdentityN(
    Graph<GraphPolicy>& graph,
    std::string_view name,
    std::span<const TF_Output> input) {
    return OpResult(graph.NewOperation("IdentityN", std::string(name)).AddInputList(input).Finish());
}

/// Reshape tensor
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Reshape(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Squeeze(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult ExpandDims(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Concat(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult ConcatV2(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Split(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult SplitV(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Slice(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult StridedSlice(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Gather(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult GatherV2(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult GatherNd(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult ScatterNd(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Tile(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Pad(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult PadV2(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult MirrorPad(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult ReverseV2(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Pack(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Unpack(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Shape(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult ShapeN(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Rank(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Size(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Fill(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult ZerosLike(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult OnesLike(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Range(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult LinSpace(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult BroadcastTo(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Where(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult SelectV2(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Cast(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Bitcast(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult CheckNumerics(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult RandomUniform(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult RandomStandardNormal(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult TruncatedNormal(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult RandomShuffle(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Multinomial(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult ResizeBilinear(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult ResizeNearestNeighbor(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult ResizeBicubic(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult CropAndResize(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult NonMaxSuppression(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult NonMaxSuppressionV3(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult DecodeJpeg(
    Graph<GraphPolicy>& graph,
    std::string_view name,
    TF_Output contents) {
    return OpResult(graph.NewOperation("DecodeJpeg", std::string(name)).AddInput(contents).Finish());
}

/// Decode PNG image
template<typename GraphPolicy>
[[nodiscard]] inline OpResult DecodePng(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult EncodeJpeg(
    Graph<GraphPolicy>& graph,
    std::string_view name,
    TF_Output image) {
    return OpResult(graph.NewOperation("EncodeJpeg", std::string(name)).AddInput(image).Finish());
}

/// Encode image as PNG
template<typename GraphPolicy>
[[nodiscard]] inline OpResult EncodePng(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult NoOp(
    Graph<GraphPolicy>& graph,
    std::string_view name) {
    return OpResult(graph.NewOperation("NoOp", std::string(name)).Finish());
}

/// Stop gradient propagation
template<typename GraphPolicy>
[[nodiscard]] inline OpResult StopGradient(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult PreventGradient(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Print(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Assert(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult StringJoin(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult StringSplit(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult RegexReplace(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult ReadFile(
    Graph<GraphPolicy>& graph,
    std::string_view name,
    TF_Output filename) {
    return OpResult(graph.NewOperation("ReadFile", std::string(name)).AddInput(filename).Finish());
}

/// Write contents to file
template<typename GraphPolicy>
[[nodiscard]] inline OpResult WriteFile(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult MatchingFiles(
    Graph<GraphPolicy>& graph,
    std::string_view name,
    TF_Output pattern) {
    return OpResult(graph.NewOperation("MatchingFiles", std::string(name)).AddInput(pattern).Finish());
}


// ============================================================================
// Variable Operations
// ============================================================================

/// Create variable
template<typename GraphPolicy>
[[nodiscard]] inline OpResult Variable(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult VariableV2(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult VarHandleOp(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult ReadVariableOp(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult AssignVariableOp(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult AssignAddVariableOp(
    Graph<GraphPolicy>& graph,
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
template<typename GraphPolicy>
[[nodiscard]] inline OpResult AssignSubVariableOp(
    Graph<GraphPolicy>& graph,
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
