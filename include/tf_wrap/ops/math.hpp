// tf_wrap/ops/math.hpp
// Math operation wrappers (generated)

#pragma once

#include "tf_wrap/ops/common.hpp"

namespace tf_wrap {
namespace ops {

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



} // namespace ops
} // namespace tf_wrap
