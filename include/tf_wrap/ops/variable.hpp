// tf_wrap/ops/variable.hpp
// Variable operation wrappers (generated)

#pragma once

#include "tf_wrap/ops/common.hpp"

namespace tf_wrap {
namespace ops {

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
