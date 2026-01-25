// tf_wrap/ops/cast.hpp
// Cast and Type operation wrappers (generated)

#pragma once

#include "tf_wrap/ops/common.hpp"

namespace tf_wrap {
namespace ops {

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



} // namespace ops
} // namespace tf_wrap
