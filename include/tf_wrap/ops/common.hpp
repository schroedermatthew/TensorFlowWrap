// tf_wrap/ops/common.hpp
// Common types for TensorFlow op wrappers

#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <optional>
#include <stdexcept>
#include <cstdint>
extern "C" {
#include <tensorflow/c/c_api.h>
}
#include "tf_wrap/graph.hpp"
#include "tf_wrap/tensor.hpp"
#include "tf_wrap/status.hpp"

namespace tf_wrap {
namespace ops {

// ============================================================================
// Op Result - Wrapper for operation outputs
// ============================================================================

/// Result of an operation - holds the TF_Operation* and provides output access
class OpResult {
public:
    explicit OpResult(TF_Operation* op) : op_(op) {
        if (!op_) throw std::runtime_error("OpResult: null operation");
    }
    
    /// Get the underlying operation
    [[nodiscard]] TF_Operation* op() const noexcept { return op_; }
    
    /// Get output at index (default 0)
    ///
    /// Throws std::out_of_range if index is invalid.
    [[nodiscard]] TF_Output output(int index = 0) const {
        const int n = num_outputs();
        if (index < 0 || index >= n) {
            throw std::out_of_range(tf_wrap::detail::format(
                "OpResult::output: index {} out of range for '{}' (has {} outputs)",
                index, name(), n));
        }
        return TF_Output{op_, index};
    }

    /// Get output at index without bounds checking (caller must ensure validity).
    [[nodiscard]] TF_Output output_unchecked(int index = 0) const noexcept {
        return TF_Output{op_, index};
    }
    
    /// Implicit conversion to TF_Output (for output 0)
    operator TF_Output() const { return output(0); }
    
    /// Get number of outputs
    [[nodiscard]] int num_outputs() const noexcept {
        return TF_OperationNumOutputs(op_);
    }
    
    /// Get operation name
    [[nodiscard]] std::string name() const {
        return TF_OperationName(op_);
    }

private:
    TF_Operation* op_;
};



} // namespace ops
} // namespace tf_wrap
