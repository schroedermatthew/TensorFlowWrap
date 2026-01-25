// tf_wrap/facade_ops.hpp
// Optional graph-building helpers built on top of the generated op wrappers.
//
// This header is intentionally NOT included by tf_wrap/core.hpp.
// Include it when you want dtype-inferred convenience helpers (Scalar, Add, ...).

#pragma once

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>

#include "tf_wrap/facade.hpp"
#include "tf_wrap/format.hpp"
#include "tf_wrap/ops/array.hpp"
#include "tf_wrap/ops/math.hpp"
#include "tf_wrap/ops/matrix.hpp"

namespace tf_wrap {
namespace facade {

// ============================================================================
// Endpoint helper
// ============================================================================

[[nodiscard]] inline Endpoint EndpointFrom(const ops::OpResult& result)
{
    return Endpoint(result.output(0));
}

// ============================================================================
// Facade helpers - Create common ops with dtype inference
// ============================================================================

/// Create a scalar constant
template<TensorScalar T>
[[nodiscard]] inline ops::OpResult Scalar(
    Graph& graph,
    std::string_view name,
    T value)
{
    auto tensor = Tensor::FromScalar<T>(value);
    return ops::Const(graph, name, tensor.handle(), tf_dtype_v<T>);
}

/// Create a constant from tensor
template<TensorScalar T>
[[nodiscard]] inline ops::OpResult Const(
    Graph& graph,
    std::string_view name,
    const Tensor& tensor)
{
    return ops::Const(graph, name, tensor.handle(), tf_dtype_v<T>);
}

/// Create a constant from an existing Tensor (dtype inferred at runtime)
[[nodiscard]] inline ops::OpResult Const(
    Graph& graph,
    std::string_view name,
    const Tensor& tensor)
{
    return ops::Const(graph, name, tensor.handle(), tensor.dtype());
}

/// Create a placeholder
template<TensorScalar T>
[[nodiscard]] inline ops::OpResult Placeholder(
    Graph& graph,
    std::string_view name,
    std::span<const std::int64_t> shape = {})
{
    return ops::Placeholder(graph, name, tf_dtype_v<T>, shape);
}

/// Create an identity (pass-through)
template<TensorScalar T>
[[nodiscard]] inline ops::OpResult Identity(
    Graph& graph,
    std::string_view name,
    TF_Output input)
{
    return ops::Identity(graph, name, input, tf_dtype_v<T>);
}

// ============================================================================
// Dtype-inferred binary operations
// ============================================================================

namespace detail {

/// Get dtype from TF_Output and validate both inputs match
inline TF_DataType get_binary_dtype(TF_Output x, TF_Output y, const char* op_name)
{
    const TF_DataType dx = TF_OperationOutputType(x);
    const TF_DataType dy = TF_OperationOutputType(y);

    if (dx != dy) {
        throw std::invalid_argument(tf_wrap::detail::format(
            "{}: input dtypes must match, got {} and {}",
            op_name, dtype_name(dx), dtype_name(dy)));
    }

    return dx;
}

} // namespace detail

/// Add with dtype inference
[[nodiscard]] inline ops::OpResult Add(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y)
{
    return ops::Add(graph, name, x, y, detail::get_binary_dtype(x, y, "Add"));
}

/// AddV2 with dtype inference
[[nodiscard]] inline ops::OpResult AddV2(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y)
{
    return ops::AddV2(graph, name, x, y, detail::get_binary_dtype(x, y, "AddV2"));
}

/// Sub with dtype inference
[[nodiscard]] inline ops::OpResult Sub(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y)
{
    return ops::Sub(graph, name, x, y, detail::get_binary_dtype(x, y, "Sub"));
}

/// Mul with dtype inference
[[nodiscard]] inline ops::OpResult Mul(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y)
{
    return ops::Mul(graph, name, x, y, detail::get_binary_dtype(x, y, "Mul"));
}

/// Div with dtype inference
[[nodiscard]] inline ops::OpResult Div(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y)
{
    return ops::Div(graph, name, x, y, detail::get_binary_dtype(x, y, "Div"));
}

/// MatMul with dtype inference
[[nodiscard]] inline ops::OpResult MatMul(
    Graph& graph,
    std::string_view name,
    TF_Output a,
    TF_Output b)
{
    return ops::MatMul(graph, name, a, b, detail::get_binary_dtype(a, b, "MatMul"));
}

} // namespace facade
} // namespace tf_wrap
