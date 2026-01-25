// tf/format.hpp
// Small compatibility wrapper for std::format.
//
// Goal:
// - Keep using std::format when the standard library provides it.
// - Provide a fallback for toolchains where <format> is missing or incomplete
//   (common on some libstdc++ versions), without forcing callers to remove
//   std::format-style call sites.
//
// This is intentionally minimal: it supports only the formatting patterns used
// in this wrapper's error messages.

#pragma once

#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

// Prefer a feature-test when available.
#if defined(__cpp_lib_format) && (__cpp_lib_format >= 201907L)
    #include <format>
    #define TFWRAP_HAS_STD_FORMAT 1
#else
    #define TFWRAP_HAS_STD_FORMAT 0
#endif

namespace tf_wrap::detail {

// -----------------------------------------------------------------------------
// format() - std::format when available; otherwise a conservative fallback.
//
// Fallback behavior:
// - If std::format is unavailable, we keep the original format string visible
//   and append all arguments (space-separated) after a " |" marker.
// - This is intentionally conservative (it never tries to interpret full
//   std::format syntax).
// -----------------------------------------------------------------------------

#if TFWRAP_HAS_STD_FORMAT

template<class... Args>
[[nodiscard]] inline std::string format(std::format_string<Args...> fmt, Args&&... args)
{
    return std::format(fmt, std::forward<Args>(args)...);
}

#else

namespace detail_impl {
inline void append_one(std::ostringstream& os, std::string_view s) { os << s; }
inline void append_one(std::ostringstream& os, const char* s) { os << (s ? s : ""); }

template<class T>
inline void append_one(std::ostringstream& os, const T& v)
{
    os << v;
}

template<class... Args>
inline void append_all(std::ostringstream& os, Args&&... args)
{
    bool first = true;
    auto append_sep = [&]() {
        if (!first) os << ' ';
        first = false;
    };

    ((append_sep(), append_one(os, std::forward<Args>(args))), ...);
}

// Conservative fallback: keep fmt visible and append args.
template<class T>
inline std::string to_string_one(T&& v)
{
    std::ostringstream tmp;
    append_one(tmp, std::forward<T>(v));
    return tmp.str();
}

template<class... Args>
inline std::string braces_replace(std::string_view fmt, Args&&... args)
{
    // Convert args to strings once (so we can substitute them into the format string)
    std::vector<std::string> arg_strs;
    arg_strs.reserve(sizeof...(Args));
    (arg_strs.push_back(to_string_one(std::forward<Args>(args))), ...);

    std::ostringstream os;
    std::size_t i = 0;
    std::size_t arg_i = 0;

    while (i < fmt.size()) {
        const char c = fmt[i];

        // Escaped braces: "{{" -> "{", "}}" -> "}"
        if (c == '{' && (i + 1) < fmt.size() && fmt[i + 1] == '{') {
            os << '{';
            i += 2;
            continue;
        }
        if (c == '}' && (i + 1) < fmt.size() && fmt[i + 1] == '}') {
            os << '}';
            i += 2;
            continue;
        }

        // Replacement: "{}" -> next arg (if any)
        if (c == '{' && (i + 1) < fmt.size() && fmt[i + 1] == '}') {
            if (arg_i < arg_strs.size()) {
                os << arg_strs[arg_i++];
            } else {
                os << "{}";
            }
            i += 2;
            continue;
        }

        os << c;
        ++i;
    }

    // If there were extra args, keep old fallback behavior: append after a marker
    if (arg_i < arg_strs.size()) {
        os << " |";
        for (std::size_t j = arg_i; j < arg_strs.size(); ++j) {
            os << ' ' << arg_strs[j];
        }
    }

    return os.str();
}
} // namespace detail_impl

template<class... Args>
[[nodiscard]] inline std::string format(std::string_view fmt, Args&&... args)
{
    // Conservative: keep the original format string visible and append args.
    // This guarantees diagnostics are not lost even without std::format.
    return detail_impl::braces_replace(fmt, std::forward<Args>(args)...);
}

#endif

} // namespace tf_wrap::detail
