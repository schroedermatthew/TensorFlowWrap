// tf_wrap/ops/control_flow.hpp
// Control Flow operation wrappers (generated)

#pragma once

#include "tf_wrap/ops/common.hpp"

namespace tf_wrap {
namespace ops {

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



} // namespace ops
} // namespace tf_wrap
