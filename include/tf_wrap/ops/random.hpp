// tf_wrap/ops/random.hpp
// Random operation wrappers (generated)

#pragma once

#include "tf_wrap/ops/common.hpp"

namespace tf_wrap {
namespace ops {

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



} // namespace ops
} // namespace tf_wrap
