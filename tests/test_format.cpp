// test_format.cpp
// Comprehensive tests for tf_wrap::detail::format
//
// Framework: doctest
// Runs with: Stub TensorFlow (all platforms)
//
// These tests cover:
// - format(): basic strings, placeholders, multiple args, edge cases
// - Escaped braces: {{ and }}
// - Type formatting: strings, ints, floats, string_view

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "tf_wrap/format.hpp"

#include <string>
#include <string_view>

// Use a namespace alias to avoid ADL issues with std::string/std::string_view
// pulling in std::format on MSVC
namespace fmt = tf_wrap::detail;

// ============================================================================
// Basic Formatting Tests
// ============================================================================

TEST_CASE("format - empty string") {
    auto result = fmt::format("");
    CHECK(result.empty());
}

TEST_CASE("format - string without placeholders") {
    auto result = fmt::format("hello world");
    CHECK(result == "hello world");
}

TEST_CASE("format - single placeholder with int") {
    auto result = fmt::format("value is {}", 42);
    CHECK(result == "value is 42");
}

TEST_CASE("format - single placeholder with string literal") {
    auto result = fmt::format("name is {}", "Alice");
    CHECK(result == "name is Alice");
}

TEST_CASE("format - single placeholder with float") {
    auto result = fmt::format("pi is approximately {}", 3.14);
    // Float formatting varies, just check it contains 3.14
    CHECK(result.find("3.14") != std::string::npos);
}

TEST_CASE("format - single placeholder with negative int") {
    auto result = fmt::format("value is {}", -123);
    CHECK(result == "value is -123");
}

// ============================================================================
// Multiple Placeholders Tests
// ============================================================================

TEST_CASE("format - two placeholders") {
    auto result = fmt::format("{} + {} = 3", 1, 2);
    CHECK(result == "1 + 2 = 3");
}

TEST_CASE("format - three placeholders") {
    auto result = fmt::format("{}, {}, {}", "a", "b", "c");
    CHECK(result == "a, b, c");
}

TEST_CASE("format - mixed types") {
    auto result = fmt::format("name={}, age={}, score={}", "Alice", 30, 95.5);
    CHECK(result.find("name=Alice") != std::string::npos);
    CHECK(result.find("age=30") != std::string::npos);
    CHECK(result.find("score=95.5") != std::string::npos);
}

TEST_CASE("format - placeholders at start and end") {
    auto result = fmt::format("{} middle {}", "start", "end");
    CHECK(result == "start middle end");
}

TEST_CASE("format - adjacent placeholders") {
    auto result = fmt::format("{}{}{}", 1, 2, 3);
    CHECK(result == "123");
}

// ============================================================================
// Escaped Braces Tests
// ============================================================================

TEST_CASE("format - escaped open brace") {
    auto result = fmt::format("use {{}} for placeholders");
    CHECK(result == "use {} for placeholders");
}

TEST_CASE("format - escaped close brace") {
    auto result = fmt::format("end with }}");
    CHECK(result == "end with }");
}

TEST_CASE("format - escaped braces with placeholder") {
    auto result = fmt::format("{{{}}} is the value", 42);
    CHECK(result == "{42} is the value");
}

TEST_CASE("format - multiple escaped braces") {
    auto result = fmt::format("{{{{}}}}");
    CHECK(result == "{{}}");
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_CASE("format - empty placeholder only") {
    auto result = fmt::format("{}", 42);
    CHECK(result == "42");
}

TEST_CASE("format - null const char*") {
    const char* ptr = nullptr;
    auto result = fmt::format("value: {}", ptr);
    // Should not crash, empty string for null
    CHECK(result.find("value:") != std::string::npos);
}

TEST_CASE("format - special characters") {
    auto result = fmt::format("special: {}", "tab\ttab");
    CHECK(result.find("tab\ttab") != std::string::npos);
}

// ============================================================================
// Type Safety Tests
// ============================================================================

TEST_CASE("format - size_t value") {
    std::size_t val = 12345;
    auto result = fmt::format("size is {}", val);
    CHECK(result == "size is 12345");
}

TEST_CASE("format - int64_t value") {
    std::int64_t val = -9876543210LL;
    auto result = fmt::format("big number: {}", val);
    CHECK(result.find("-9876543210") != std::string::npos);
}

TEST_CASE("format - bool value") {
    auto result_true = fmt::format("flag is {}", true);
    auto result_false = fmt::format("flag is {}", false);
    // Bool may format as 1/0 or true/false depending on impl
    CHECK(!result_true.empty());
    CHECK(!result_false.empty());
}

// ============================================================================
// Error Message Use Case Tests
// ============================================================================

TEST_CASE("format - typical error message pattern") {
    auto result = fmt::format("operation '{}' failed with code {}", "MatMul", 3);
    CHECK(result.find("operation 'MatMul'") != std::string::npos);
    CHECK(result.find("code 3") != std::string::npos);
}

TEST_CASE("format - tensor shape pattern") {
    auto result = fmt::format("shape mismatch: expected {}, got {}", "[2,3,4]", "[2,3,5]");
    CHECK(result.find("[2,3,4]") != std::string::npos);
    CHECK(result.find("[2,3,5]") != std::string::npos);
}

TEST_CASE("format - index out of range pattern") {
    auto result = fmt::format("index {} out of range [0, {})", 10, 5);
    CHECK(result.find("index 10") != std::string::npos);
    CHECK(result.find("[0, 5)") != std::string::npos);
}
