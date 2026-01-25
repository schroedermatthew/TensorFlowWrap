// tf_wrap/ops/reduction.hpp
// Reduction operation wrappers (generated)

#pragma once

#include "tf_wrap/ops/common.hpp"

namespace tf_wrap {
namespace ops {

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



} // namespace ops
} // namespace tf_wrap
