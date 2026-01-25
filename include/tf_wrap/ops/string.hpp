// tf_wrap/ops/string.hpp
// String operation wrappers (generated)

#pragma once

#include "tf_wrap/ops/common.hpp"

namespace tf_wrap {
namespace ops {

// ============================================================================
// String Operations
// ============================================================================

/// Join strings

[[nodiscard]] inline OpResult StringJoin(
    Graph& graph,
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

[[nodiscard]] inline OpResult StringSplit(
    Graph& graph,
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

[[nodiscard]] inline OpResult RegexReplace(
    Graph& graph,
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



} // namespace ops
} // namespace tf_wrap
