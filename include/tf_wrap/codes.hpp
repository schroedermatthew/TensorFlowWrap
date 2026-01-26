// tf_wrap/codes.hpp
// Shared helpers for TensorFlow status/error codes
//
// Centralizes TF_Code -> string mapping to avoid duplication.

#pragma once

extern "C" {
#include <tensorflow/c/c_api.h>
}

namespace tf_wrap {

/// Convert a TensorFlow TF_Code to a stable string name.
///
/// This is a pure function and never throws.
[[nodiscard]] constexpr const char* code_to_string(TF_Code code) noexcept {
    switch (code) {
        case TF_OK:                  return "OK";
        case TF_CANCELLED:           return "CANCELLED";
        case TF_UNKNOWN:             return "UNKNOWN";
        case TF_INVALID_ARGUMENT:    return "INVALID_ARGUMENT";
        case TF_DEADLINE_EXCEEDED:   return "DEADLINE_EXCEEDED";
        case TF_NOT_FOUND:           return "NOT_FOUND";
        case TF_ALREADY_EXISTS:      return "ALREADY_EXISTS";
        case TF_PERMISSION_DENIED:   return "PERMISSION_DENIED";
        case TF_UNAUTHENTICATED:     return "UNAUTHENTICATED";
        case TF_RESOURCE_EXHAUSTED:  return "RESOURCE_EXHAUSTED";
        case TF_FAILED_PRECONDITION: return "FAILED_PRECONDITION";
        case TF_ABORTED:             return "ABORTED";
        case TF_OUT_OF_RANGE:        return "OUT_OF_RANGE";
        case TF_UNIMPLEMENTED:       return "UNIMPLEMENTED";
        case TF_INTERNAL:            return "INTERNAL";
        case TF_UNAVAILABLE:         return "UNAVAILABLE";
        case TF_DATA_LOSS:           return "DATA_LOSS";
        default:                     return "UNKNOWN_CODE";
    }
}

} // namespace tf_wrap
