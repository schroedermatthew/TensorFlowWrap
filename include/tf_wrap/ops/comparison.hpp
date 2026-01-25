// tf_wrap/ops/comparison.hpp
// Comparison operation wrappers (generated)

#pragma once

#include "tf_wrap/ops/common.hpp"

namespace tf_wrap {
namespace ops {

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



} // namespace ops
} // namespace tf_wrap
