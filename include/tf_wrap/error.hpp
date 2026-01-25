// tf_wrap/error.hpp
// Structured exception type for TensorFlowWrap
//
// Goals:
// - Preserve compatibility with code that catches std::runtime_error
// - Expose TF_Code, context, optional op name/index, and source location

#pragma once

#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>

extern "C" {
#include <tensorflow/c/c_api.h>
}

#include "tf_wrap/format.hpp"

namespace tf_wrap {

enum class ErrorSource {
    TensorFlow,
    Wrapper,
};

/// Structured error for both TensorFlow status failures and wrapper-level
/// validation errors.
///
/// This derives from std::runtime_error for backward compatibility.
class Error : public std::runtime_error {
public:
    Error(
        ErrorSource source,
        TF_Code code,
        std::string_view context,
        std::string_view message,
        std::string_view op_name = {},
        int index = -1,
        std::source_location loc = std::source_location::current())
        : std::runtime_error(build_what_(source, code, context, message, op_name, index, loc))
        , source_(source)
        , code_(code)
        , context_(context)
        , op_name_(op_name)
        , index_(index)
        , loc_(loc)
    {}

    [[nodiscard]] ErrorSource source() const noexcept { return source_; }
    [[nodiscard]] TF_Code code() const noexcept { return code_; }
    [[nodiscard]] const char* code_name() const noexcept { return code_to_string(code_); }

    [[nodiscard]] std::string_view context() const noexcept { return context_; }
    [[nodiscard]] std::string_view op_name() const noexcept { return op_name_; }
    [[nodiscard]] int index() const noexcept { return index_; }
    [[nodiscard]] std::source_location location() const noexcept { return loc_; }

    // Convenience factories

    [[nodiscard]] static Error TensorFlow(
        TF_Code code,
        std::string_view context,
        std::string_view message,
        std::source_location loc = std::source_location::current())
    {
        return Error(ErrorSource::TensorFlow, code, context, message, {}, -1, loc);
    }

    [[nodiscard]] static Error Wrapper(
        TF_Code code,
        std::string_view context,
        std::string_view message,
        std::string_view op_name = {},
        int index = -1,
        std::source_location loc = std::source_location::current())
    {
        return Error(ErrorSource::Wrapper, code, context, message, op_name, index, loc);
    }

    [[nodiscard]] static constexpr const char* code_to_string(TF_Code code) noexcept {
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

private:
    ErrorSource source_{ErrorSource::Wrapper};
    TF_Code code_{TF_UNKNOWN};
    std::string context_;
    std::string op_name_;
    int index_{-1};
    std::source_location loc_{};

    static std::string build_what_(
        ErrorSource source,
        TF_Code code,
        std::string_view context,
        std::string_view message,
        std::string_view op_name,
        int index,
        const std::source_location& loc)
    {
        const char* prefix = (source == ErrorSource::TensorFlow) ? "TF" : "WRAP";

        std::string op_part;
        if (!op_name.empty()) {
            if (index >= 0) {
                op_part = tf_wrap::detail::format(" op '{}:{}'", op_name, index);
            } else {
                op_part = tf_wrap::detail::format(" op '{}'", op_name);
            }
        }

        if (context.empty()) {
            return tf_wrap::detail::format(
                "[{}_{}]{} at {}:{} in {}: {}",
                prefix, code_to_string(code), op_part,
                loc.file_name(), loc.line(), loc.function_name(), message);
        }

        return tf_wrap::detail::format(
            "[{}_{}] {}{} at {}:{} in {}: {}",
            prefix, code_to_string(code), context, op_part,
            loc.file_name(), loc.line(), loc.function_name(), message);
    }
};

} // namespace tf_wrap
