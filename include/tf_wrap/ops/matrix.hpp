// tf_wrap/ops/matrix.hpp
// Matrix operation wrappers (generated)

#pragma once

#include "tf_wrap/ops/common.hpp"

namespace tf_wrap {
namespace ops {

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



} // namespace ops
} // namespace tf_wrap
