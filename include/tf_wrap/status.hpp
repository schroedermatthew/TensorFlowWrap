// tf/status.hpp
// RAII wrapper for TF_Status with exception-based error handling
//
// Fixes applied:
// - P0: TF_Status always deleted (no leaks on success or error path)
// - P3: Human-readable error code names
// - V5: string_view safety fix (makes owned copy for null-termination)

#pragma once

#include <new>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

extern "C" {
#include <tensorflow/c/c_api.h>
}

#include "tf_wrap/format.hpp"
#include "tf_wrap/error.hpp"

namespace tf_wrap {

// ============================================================================
// Status - RAII wrapper for TF_Status*
// ============================================================================

class Status {
public:
    /// Create a new status (initially OK)
    Status() : st_(TF_NewStatus()) {
        if (!st_) throw std::bad_alloc();
    }

    ~Status() {
        if (st_) TF_DeleteStatus(st_);
    }

    // Non-copyable
    Status(const Status&) = delete;
    Status& operator=(const Status&) = delete;

    // Movable
    Status(Status&& other) noexcept : st_(other.st_) {
        other.st_ = nullptr;
    }

    Status& operator=(Status&& other) noexcept {
        if (this != &other) {
            if (st_) TF_DeleteStatus(st_);
            st_ = other.st_;
            other.st_ = nullptr;
        }
        return *this;
    }

    // ─────────────────────────────────────────────────────────────────
    // Accessors
    // ─────────────────────────────────────────────────────────────────

    [[nodiscard]] TF_Status* get() noexcept { return st_; }
    [[nodiscard]] TF_Status* handle() noexcept { return st_; }
    [[nodiscard]] const TF_Status* get() const noexcept { return st_; }
    [[nodiscard]] const TF_Status* handle() const noexcept { return st_; }
    [[nodiscard]] TF_Code code() const noexcept { return TF_GetCode(st_); }
    [[nodiscard]] const char* code_name() const noexcept { return code_to_string(code()); }
    [[nodiscard]] const char* message() const noexcept { return TF_Message(st_); }
    [[nodiscard]] bool ok() const noexcept { return code() == TF_OK; }
    [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
    [[nodiscard]] bool operator!() const noexcept { return !ok(); }

    // ─────────────────────────────────────────────────────────────────
    // Mutation
    // ─────────────────────────────────────────────────────────────────

    void reset() noexcept {
        TF_SetStatus(st_, TF_OK, "");
    }

    // ─────────────────────────────────────────────────────────────────
    // Error handling
    // ─────────────────────────────────────────────────────────────────

    void throw_if_error(const char* context,
                        std::source_location loc = std::source_location::current()) const
    {
        if (ok()) return;
        throw Error::from_status(*this, context, loc);
    }

    void throw_if_error(const std::string& context,
                        std::source_location loc = std::source_location::current()) const
    {
        throw_if_error(context.c_str(), loc);
    }

    // ─────────────────────────────────────────────────────────────────
    // Code name helper
    // ─────────────────────────────────────────────────────────────────

    [[nodiscard]] static const char* code_to_string(TF_Code code) noexcept {
        switch (code) {
        case TF_OK: return "TF_OK";
        case TF_CANCELLED: return "TF_CANCELLED";
        case TF_UNKNOWN: return "TF_UNKNOWN";
        case TF_INVALID_ARGUMENT: return "TF_INVALID_ARGUMENT";
        case TF_DEADLINE_EXCEEDED: return "TF_DEADLINE_EXCEEDED";
        case TF_NOT_FOUND: return "TF_NOT_FOUND";
        case TF_ALREADY_EXISTS: return "TF_ALREADY_EXISTS";
        case TF_PERMISSION_DENIED: return "TF_PERMISSION_DENIED";
        case TF_UNAUTHENTICATED: return "TF_UNAUTHENTICATED";
        case TF_RESOURCE_EXHAUSTED: return "TF_RESOURCE_EXHAUSTED";
        case TF_FAILED_PRECONDITION: return "TF_FAILED_PRECONDITION";
        case TF_ABORTED: return "TF_ABORTED";
        case TF_OUT_OF_RANGE: return "TF_OUT_OF_RANGE";
        case TF_UNIMPLEMENTED: return "TF_UNIMPLEMENTED";
        case TF_INTERNAL: return "TF_INTERNAL";
        case TF_UNAVAILABLE: return "TF_UNAVAILABLE";
        case TF_DATA_LOSS: return "TF_DATA_LOSS";
        default: return "TF_<unknown>";
        }
    }

private:
    TF_Status* st_{nullptr};
};

} // namespace tf_wrap
