// tf_wrap/tensor_name.hpp
// Shared parsing for "op:index" tensor name strings.
//
// This is used by both the core Session API and the facade layer.
//
// Parsing rules:
// - Trim leading/trailing ASCII whitespace.
// - If the last ':' is followed only by digits, treat it as an output index.
// - Otherwise, treat ':' as part of the operation name.

#pragma once

#include <algorithm>
#include <cctype>
#include <charconv>
#include <stdexcept>
#include <string>
#include <string_view>

#include "tf_wrap/format.hpp"

namespace tf_wrap::detail {

struct ParsedTensorName {
    std::string op;
    int index{0};
    bool had_explicit_index{false};
};

/// Parse a tensor name string like "op_name:0" or "op_name".
///
/// Throws std::invalid_argument on parse errors.
[[nodiscard]] inline ParsedTensorName parse_tensor_name(std::string_view s) {
    ParsedTensorName result;

    // Trim whitespace
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
        s.remove_prefix(1);
    }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
        s.remove_suffix(1);
    }

    if (s.empty()) {
        throw std::invalid_argument("parse_tensor_name: empty string");
    }

    // Find the last colon
    const auto colon_pos = s.rfind(':');

    if (colon_pos == std::string_view::npos) {
        // No colon - just op name
        result.op = std::string(s);
        result.index = 0;
        result.had_explicit_index = false;
    } else if (colon_pos == 0) {
        // Colon at start - invalid
        throw std::invalid_argument("parse_tensor_name: empty operation name");
    } else if (colon_pos == s.size() - 1) {
        // Colon at end with nothing after
        throw std::invalid_argument("parse_tensor_name: missing index after colon");
    } else {
        // Has colon - check if everything after is a valid non-negative integer
        std::string_view index_part = s.substr(colon_pos + 1);

        const bool all_digits = !index_part.empty() &&
            std::all_of(index_part.begin(), index_part.end(),
                [](unsigned char c) { return std::isdigit(c); });

        if (all_digits) {
            result.op = std::string(s.substr(0, colon_pos));

            int idx = 0;
            auto [ptr, ec] = std::from_chars(
                index_part.data(),
                index_part.data() + index_part.size(),
                idx);

            if (ec != std::errc{} || ptr != index_part.data() + index_part.size()) {
                throw std::invalid_argument(tf_wrap::detail::format(
                    "parse_tensor_name: invalid index '{}'", index_part));
            }

            result.index = idx;
            result.had_explicit_index = true;
        } else {
            // Colon is part of the op name (e.g. "scope/op:name")
            result.op = std::string(s);
            result.index = 0;
            result.had_explicit_index = false;
        }
    }

    if (result.op.empty()) {
        throw std::invalid_argument("parse_tensor_name: empty operation name");
    }

    return result;
}

} // namespace tf_wrap::detail
