// tf_wrap/ops/io.hpp
// File I/O operation wrappers (generated)

#pragma once

#include "tf_wrap/ops/common.hpp"

namespace tf_wrap {
namespace ops {

// ============================================================================
// File I/O Operations
// ============================================================================

/// Read entire file contents

[[nodiscard]] inline OpResult ReadFile(
    Graph& graph,
    std::string_view name,
    TF_Output filename) {
    return OpResult(graph.NewOperation("ReadFile", std::string(name)).AddInput(filename).Finish());
}

/// Write contents to file

[[nodiscard]] inline OpResult WriteFile(
    Graph& graph,
    std::string_view name,
    TF_Output filename,
    TF_Output contents) {
    return OpResult(
        graph.NewOperation("WriteFile", std::string(name))
        .AddInput(filename)
        .AddInput(contents)
        .Finish());
}

/// Find files matching pattern

[[nodiscard]] inline OpResult MatchingFiles(
    Graph& graph,
    std::string_view name,
    TF_Output pattern) {
    return OpResult(graph.NewOperation("MatchingFiles", std::string(name)).AddInput(pattern).Finish());
}



} // namespace ops
} // namespace tf_wrap
