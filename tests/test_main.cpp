// tests/test_main.cpp
// Dependency-free tests for the TensorFlow C++20 wrapper.
//
// Rationale:
// - This project is intended to be header-only and usable in environments
//   where external test dependencies (Catch2, GTest, etc.) are not allowed.
// - This file provides a minimal test harness with clear diagnostics.

#include "tf_wrap/all.hpp"

#include "tf_wrap/format.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

namespace tf_test {

struct TestCase {
    const char* name;
    void (*fn)();
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

struct Registrar {
    Registrar(const char* name, void (*fn)()) {
        registry().push_back(TestCase{name, fn});
    }
};

[[nodiscard]] inline std::string where(const char* file, int line) {
    return tf_wrap::detail::format("{}:{}", file, line);
}

inline void require_impl(bool cond, const char* expr, const char* file, int line) {
    if (cond) return;
    throw std::runtime_error(tf_wrap::detail::format(
        "REQUIRE failed: {} ({})", expr, where(file, line)));
}

inline void require_false_impl(bool cond, const char* expr, const char* file, int line) {
    if (!cond) return;
    throw std::runtime_error(tf_wrap::detail::format(
        "REQUIRE_FALSE failed: {} ({})", expr, where(file, line)));
}

template<class A, class B>
inline void require_eq_impl(const A& a, const B& b, const char* expr_a, const char* expr_b,
                            const char* file, int line) {
    if (a == b) return;
    throw std::runtime_error(tf_wrap::detail::format(
        "REQUIRE_EQ failed: {} == {} ({})", expr_a, expr_b, where(file, line)));
}

inline bool approx_equal(double a, double b, double rel = 1e-12, double abs = 1e-12) {
    const double diff = (a > b) ? (a - b) : (b - a);
    if (diff <= abs) return true;
    const double denom = (a > 0.0 ? a : -a) + (b > 0.0 ? b : -b) + abs;
    return diff <= rel * denom;
}

inline void require_approx_impl(double a, double b, const char* expr_a, const char* expr_b,
                                const char* file, int line) {
    if (approx_equal(a, b)) return;
    throw std::runtime_error(tf_wrap::detail::format(
        "REQUIRE_APPROX failed: {} ~= {} ({})", expr_a, expr_b, where(file, line)));
}

template<class Ex, class Fn>
inline void require_throws_as_impl(Fn&& fn, const char* expr, const char* ex_name,
                                  const char* file, int line) {
    try {
        std::forward<Fn>(fn)();
    } catch (const Ex&) {
        return;
    } catch (const std::exception& e) {
        throw std::runtime_error(tf_wrap::detail::format(
            "REQUIRE_THROWS_AS failed: {} threw different std::exception ({}) ({})",
            expr, typeid(e).name(), where(file, line)));
    } catch (...) {
        throw std::runtime_error(tf_wrap::detail::format(
            "REQUIRE_THROWS_AS failed: {} threw non-std exception ({})",
            expr, where(file, line)));
    }

    throw std::runtime_error(tf_wrap::detail::format(
        "REQUIRE_THROWS_AS failed: {} did not throw {} ({})",
        expr, ex_name, where(file, line)));
}

} // namespace tf_test

#define TF_TEST_JOIN2(a, b) a##b
#define TF_TEST_JOIN(a, b) TF_TEST_JOIN2(a, b)

#define TF_TEST_CASE(name_literal)                                                   \
    static void TF_TEST_JOIN(tf_test_fn_, __LINE__)();                               \
    static ::tf_test::Registrar TF_TEST_JOIN(tf_test_reg_, __LINE__)(                \
        (name_literal), &TF_TEST_JOIN(tf_test_fn_, __LINE__));                       \
    static void TF_TEST_JOIN(tf_test_fn_, __LINE__)()

#define TF_REQUIRE(expr) ::tf_test::require_impl((expr), #expr, __FILE__, __LINE__)
#define TF_REQUIRE_FALSE(expr) ::tf_test::require_false_impl((expr), #expr, __FILE__, __LINE__)
#define TF_REQUIRE_EQ(a, b) ::tf_test::require_eq_impl((a), (b), #a, #b, __FILE__, __LINE__)
#define TF_REQUIRE_APPROX(a, b) ::tf_test::require_approx_impl((a), (b), #a, #b, __FILE__, __LINE__)

#define TF_REQUIRE_THROWS_AS(expr, ex_type)                                          \
    ::tf_test::require_throws_as_impl<ex_type>([&]() { (void)(expr); }, #expr, #ex_type, __FILE__, __LINE__)

// ============================================================================
// Compile-time checks (replacement for Catch2 STATIC_REQUIRE)
// ============================================================================


// ============================================================================
// Runtime tests
// ============================================================================

TF_TEST_CASE("Status RAII prevents leaks") {
    for (int i = 0; i < 100; ++i) {
        tf_wrap::Status st;
        TF_REQUIRE(st.ok());
        TF_REQUIRE(st.code() == TF_OK);
    }
}

TF_TEST_CASE("Status reset() works") {
    tf_wrap::Status st;
    st.set(TF_INVALID_ARGUMENT, "test error");
    TF_REQUIRE_FALSE(st.ok());

    st.reset();
    TF_REQUIRE(st.ok());
}

TF_TEST_CASE("Status code names are correct") {
    TF_REQUIRE_EQ(std::string(tf_wrap::Status::code_to_string(TF_OK)), "OK");
    TF_REQUIRE_EQ(std::string(tf_wrap::Status::code_to_string(TF_INVALID_ARGUMENT)), "INVALID_ARGUMENT");
    TF_REQUIRE_EQ(std::string(tf_wrap::Status::code_to_string(TF_NOT_FOUND)), "NOT_FOUND");
}

TF_TEST_CASE("Status operator!() works") {
    tf_wrap::Status st;
    TF_REQUIRE_FALSE(!st);  // OK status

    st.set(TF_INVALID_ARGUMENT, "test error");
    TF_REQUIRE(!st);  // Error status

    if (!st) {
        TF_REQUIRE(st.code() == TF_INVALID_ARGUMENT);
    }
}

TF_TEST_CASE("Status set() accepts string_view") {
    tf_wrap::Status st;

    st.set(TF_CANCELLED);
    TF_REQUIRE(st.code() == TF_CANCELLED);

    std::string_view msg = "test message";
    st.set(TF_INTERNAL, msg);
    TF_REQUIRE(st.code() == TF_INTERNAL);
    TF_REQUIRE_EQ(std::string(st.message()), "test message");

    st.set(TF_UNAVAILABLE, std::string("temporary").substr(0, 4));
    TF_REQUIRE(st.code() == TF_UNAVAILABLE);
}

TF_TEST_CASE("Tensor FromVector works") {
    std::vector<std::int64_t> shape = {2, 3};
    std::vector<float> data = {1, 2, 3, 4, 5, 6};

    auto tensor = tf_wrap::Tensor::FromVector<float>(shape, data);

    TF_REQUIRE(tensor.dtype() == TF_FLOAT);
    TF_REQUIRE(tensor.shape() == shape);
    TF_REQUIRE(tensor.num_elements() == 6);
    TF_REQUIRE(tensor.rank() == 2);
}

TF_TEST_CASE("Tensor FromVector<bool> works") {
    // std::vector<bool> is a special bitfield type without .data()
    // This test verifies the fix for that limitation
    std::vector<std::int64_t> shape = {5};
    std::vector<bool> data = {true, false, true, false, true};

    auto tensor = tf_wrap::Tensor::FromVector<bool>(shape, data);

    TF_REQUIRE(tensor.dtype() == TF_BOOL);
    TF_REQUIRE(tensor.shape() == shape);
    TF_REQUIRE(tensor.num_elements() == 5);

    auto view = tensor.read<bool>();
    TF_REQUIRE(view[0] == true);
    TF_REQUIRE(view[1] == false);
    TF_REQUIRE(view[2] == true);
    TF_REQUIRE(view[3] == false);
    TF_REQUIRE(view[4] == true);
}

TF_TEST_CASE("Tensor FromScalar works") {
    auto tensor = tf_wrap::Tensor::FromScalar<double>(3.14);

    TF_REQUIRE(tensor.dtype() == TF_DOUBLE);
    TF_REQUIRE(tensor.num_elements() == 1);

    auto view = tensor.read<double>();
    TF_REQUIRE_APPROX(view[0], 3.14);
}

TF_TEST_CASE("Tensor FromRaw rejects null") {
    TF_REQUIRE_THROWS_AS(tf_wrap::Tensor::FromRaw(nullptr), std::invalid_argument);
}

TF_TEST_CASE("Tensor Zeros factory works") {
    auto tensor = tf_wrap::Tensor::Zeros<float>({10});

    auto view = tensor.read<float>();
    for (float x : view) {
        TF_REQUIRE(x == 0.0f);
    }
}

TF_TEST_CASE("Tensor Allocate factory works") {
    auto tensor = tf_wrap::Tensor::Allocate<std::int32_t>({100});

    TF_REQUIRE(tensor.num_elements() == 100);
    TF_REQUIRE(tensor.byte_size() == 400);
}

TF_TEST_CASE("Tensor read/write views work") {
    auto tensor = tf_wrap::Tensor::FromVector<float>({4}, {1, 2, 3, 4});

    {
        auto view = tensor.read<float>();
        TF_REQUIRE(view.size() == 4);
        TF_REQUIRE(view[0] == 1.0f);
        TF_REQUIRE(view[3] == 4.0f);
    }

    {
        auto view = tensor.write<float>();
        view[0] = 10.0f;
        view[3] = 40.0f;
    }

    {
        auto view = tensor.read<float>();
        TF_REQUIRE(view[0] == 10.0f);
        TF_REQUIRE(view[3] == 40.0f);
    }
}

TF_TEST_CASE("Tensor with_read/with_write work") {
    auto tensor = tf_wrap::Tensor::FromVector<float>({5}, {1, 2, 3, 4, 5});

    float sum = tensor.with_read<float>([](std::span<const float> s) {
        return std::accumulate(s.begin(), s.end(), 0.0f);
    });
    TF_REQUIRE(sum == 15.0f);

    tensor.with_write<float>([](std::span<float> s) {
        for (float& x : s) x *= 2.0f;
    });

    float new_sum = tensor.with_read<float>([](std::span<const float> s) {
        return std::accumulate(s.begin(), s.end(), 0.0f);
    });
    TF_REQUIRE(new_sum == 30.0f);
}

TF_TEST_CASE("Tensor dtype mismatch throws") {
    auto tensor = tf_wrap::Tensor::FromScalar<float>(1.0f);
    TF_REQUIRE_THROWS_AS(tensor.read<double>(), std::runtime_error);
    TF_REQUIRE_THROWS_AS(tensor.write<int>(), std::runtime_error);
}

TF_TEST_CASE("Tensor ToVector extracts data correctly") {
    std::vector<std::int64_t> shape = {2, 3};
    std::vector<float> data = {1, 2, 3, 4, 5, 6};

    auto tensor = tf_wrap::Tensor::FromVector<float>(shape, data);
    auto extracted = tensor.ToVector<float>();

    TF_REQUIRE_EQ(extracted.size(), 6u);
    TF_REQUIRE(extracted == data);
}

TF_TEST_CASE("Tensor ToScalar extracts single value") {
    auto tensor = tf_wrap::Tensor::FromScalar<double>(3.14159);
    double value = tensor.ToScalar<double>();
    TF_REQUIRE_APPROX(value, 3.14159);
}

TF_TEST_CASE("Tensor ToScalar throws for multi-element tensor") {
    auto tensor = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    TF_REQUIRE_THROWS_AS(tensor.ToScalar<float>(), std::runtime_error);
}

TF_TEST_CASE("Tensor ToVector dtype mismatch throws") {
    auto tensor = tf_wrap::Tensor::FromScalar<float>(1.0f);
    TF_REQUIRE_THROWS_AS(tensor.ToVector<double>(), std::runtime_error);
}

TF_TEST_CASE("Tensor Clone creates independent copy") {
    auto original = tf_wrap::Tensor::FromVector<float>({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto cloned = original.Clone();
    
    // Verify same shape and dtype
    TF_REQUIRE(cloned.dtype() == original.dtype());
    TF_REQUIRE(cloned.shape() == original.shape());
    TF_REQUIRE(cloned.num_elements() == original.num_elements());
    
    // Verify data was copied
    auto orig_data = original.ToVector<float>();
    auto clone_data = cloned.ToVector<float>();
    TF_REQUIRE(orig_data == clone_data);
    
    // Verify independence - modify clone, original unchanged
    {
        auto view = cloned.write<float>();
        view[0] = 999.0f;
    }
    
    auto orig_after = original.ToVector<float>();
    TF_REQUIRE(orig_after[0] == 1.0f);  // Original unchanged
    
    auto clone_after = cloned.ToVector<float>();
    TF_REQUIRE(clone_after[0] == 999.0f);  // Clone modified
}

TF_TEST_CASE("Tensor Clone of empty returns empty") {
    tf_wrap::Tensor empty;
    auto cloned = empty.Clone();
    
    TF_REQUIRE(cloned.empty());
    TF_REQUIRE(cloned.handle() == nullptr);
}

TF_TEST_CASE("Tensor Clone of scalar works") {
    auto original = tf_wrap::Tensor::FromScalar<int>(42);
    auto cloned = original.Clone();
    
    TF_REQUIRE(cloned.rank() == 0);
    TF_REQUIRE(cloned.num_elements() == 1);
    TF_REQUIRE(cloned.ToScalar<int>() == 42);
}

TF_TEST_CASE("Tensor dimension mismatch throws") {
    TF_REQUIRE_THROWS_AS(
        tf_wrap::Tensor::FromVector<float>({2, 3}, {1, 2, 3}),
        std::invalid_argument);
}

TF_TEST_CASE("All scalar types map to TF dtypes") {
    TF_REQUIRE(tf_wrap::tf_dtype_v<float> == TF_FLOAT);
    TF_REQUIRE(tf_wrap::tf_dtype_v<double> == TF_DOUBLE);
    TF_REQUIRE(tf_wrap::tf_dtype_v<std::int8_t> == TF_INT8);
    TF_REQUIRE(tf_wrap::tf_dtype_v<std::int16_t> == TF_INT16);
    TF_REQUIRE(tf_wrap::tf_dtype_v<std::int32_t> == TF_INT32);
    TF_REQUIRE(tf_wrap::tf_dtype_v<std::int64_t> == TF_INT64);
    TF_REQUIRE(tf_wrap::tf_dtype_v<std::uint8_t> == TF_UINT8);
    TF_REQUIRE(tf_wrap::tf_dtype_v<std::uint16_t> == TF_UINT16);
    TF_REQUIRE(tf_wrap::tf_dtype_v<std::uint32_t> == TF_UINT32);
    TF_REQUIRE(tf_wrap::tf_dtype_v<std::uint64_t> == TF_UINT64);
    TF_REQUIRE(tf_wrap::tf_dtype_v<bool> == TF_BOOL);
}


TF_TEST_CASE("SessionOptions RAII works") {
    tf_wrap::SessionOptions opts;
    TF_REQUIRE(opts.get() != nullptr);

    tf_wrap::SessionOptions opts2 = std::move(opts);
    TF_REQUIRE(opts2.get() != nullptr);
    TF_REQUIRE(opts.get() == nullptr);
}

TF_TEST_CASE("Graph creation and operation lookup") {
    tf_wrap::Graph graph;

    auto tensor = tf_wrap::Tensor::FromScalar<float>(1.0f);

    auto op = graph.NewOperation("Const", "test_const")
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();

    TF_REQUIRE(op != nullptr);

    auto found = graph.GetOperation("test_const");
    TF_REQUIRE(found.has_value());
    TF_REQUIRE(*found == op);

    auto not_found = graph.GetOperation("nonexistent");
    TF_REQUIRE_FALSE(not_found.has_value());
}

TF_TEST_CASE("Graph GetOperationOrThrow throws when not found") {
    tf_wrap::Graph graph;
    TF_REQUIRE_THROWS_AS(graph.GetOperationOrThrow("nonexistent"), std::runtime_error);
}

TF_TEST_CASE("tf_wrap::Output helper creates correct TF_Output") {
    tf_wrap::Graph graph;
    
    auto tensor = tf_wrap::Tensor::FromScalar<float>(1.0f);
    auto op = graph.NewOperation("Const", "test_const")
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Test Output from raw pointer
    TF_Output out1 = tf_wrap::Output(op, 0);
    TF_REQUIRE(out1.oper == op);
    TF_REQUIRE(out1.index == 0);
    
    // Test Output with different index
    TF_Output out2 = tf_wrap::Output(op, 1);
    TF_REQUIRE(out2.oper == op);
    TF_REQUIRE(out2.index == 1);
    
    // Test Output from Operation wrapper
    tf_wrap::Operation wrapped(op);
    TF_Output out3 = tf_wrap::Output(wrapped, 0);
    TF_REQUIRE(out3.oper == op);
    TF_REQUIRE(out3.index == 0);
}

// ============================================================================
// Debugging Features Tests
// ============================================================================

TF_TEST_CASE("Buffer RAII works correctly") {
    // Empty buffer
    tf_wrap::Buffer empty;
    TF_REQUIRE(empty.get() != nullptr);
    TF_REQUIRE(empty.empty());
    TF_REQUIRE(empty.length() == 0);
    
    // Buffer with data
    std::string data = "hello world";
    tf_wrap::Buffer with_data(data.data(), data.size());
    TF_REQUIRE(!with_data.empty());
    TF_REQUIRE(with_data.length() == data.size());
    
    // to_bytes
    auto bytes = with_data.to_bytes();
    TF_REQUIRE(bytes.size() == data.size());
    
    // Move
    tf_wrap::Buffer moved = std::move(with_data);
    TF_REQUIRE(!moved.empty());
    TF_REQUIRE(with_data.get() == nullptr);  // NOLINT: testing moved-from state
}

TF_TEST_CASE("Graph ToGraphDef serializes graph") {
    tf_wrap::Graph graph;
    
    // Add some operations
    auto t1 = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)graph.NewOperation("Const", "A")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto t2 = tf_wrap::Tensor::FromScalar<float>(2.0f);
    (void)graph.NewOperation("Const", "B")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Serialize
    auto graph_def = graph.ToGraphDef();
    TF_REQUIRE(!graph_def.empty());
    
#ifdef TF_WRAPPER_TF_STUB_ENABLED
    // In the stub, we get a text representation
    std::string content(graph_def.begin(), graph_def.end());
    TF_REQUIRE(content.find("STUB_GRAPH_DEF") != std::string::npos);
    TF_REQUIRE(content.find("num_operations: 2") != std::string::npos);
#else
    // In real TF, we get a protobuf binary - just check it's non-empty
    TF_REQUIRE(graph_def.size() > 10);  // Should be more than a few bytes
#endif
}

TF_TEST_CASE("Graph GetPlaceholders finds placeholder ops") {
    tf_wrap::Graph graph;
    
    // Add a Placeholder (note: in real TF you'd use Placeholder op type)
    auto t = tf_wrap::Tensor::FromScalar<float>(0.0f);
    (void)graph.NewOperation("Placeholder", "input")
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Add a Const (not a placeholder)
    (void)graph.NewOperation("Const", "weight")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto placeholders = graph.GetPlaceholders();
    TF_REQUIRE(placeholders.size() == 1u);
    TF_REQUIRE(placeholders[0].op_name == "input");
    TF_REQUIRE(placeholders[0].op_type == "Placeholder");
}

TF_TEST_CASE("Graph GetOperationsByType filters correctly") {
    tf_wrap::Graph graph;
    
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)graph.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    (void)graph.NewOperation("Const", "B")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    (void)graph.NewOperation("Placeholder", "X")
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto consts = graph.GetOperationsByType("Const");
    TF_REQUIRE(consts.size() == 2u);
    
    auto placeholders = graph.GetOperationsByType("Placeholder");
    TF_REQUIRE(placeholders.size() == 1u);
    
    auto none = graph.GetOperationsByType("NonExistent");
    TF_REQUIRE(none.empty());
}

TF_TEST_CASE("Graph DebugString produces readable output") {
    tf_wrap::Graph graph;
    
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)graph.NewOperation("Const", "MyConst")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    std::string debug = graph.DebugString();
    TF_REQUIRE(debug.find("1 operations") != std::string::npos);
    TF_REQUIRE(debug.find("MyConst") != std::string::npos);
    TF_REQUIRE(debug.find("Const") != std::string::npos);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    int failed = 0;
    const auto& tests = tf_test::registry();

    std::cout << "Running " << tests.size() << " tests...\n\n";

    for (const auto& tc : tests) {
        try {
            tc.fn();
            std::cout << "[PASS] " << tc.name << "\n";
        } catch (const std::exception& e) {
            ++failed;
            std::cout << "[FAIL] " << tc.name << "\n";
            std::cout << "       " << e.what() << "\n";
        } catch (...) {
            ++failed;
            std::cout << "[FAIL] " << tc.name << "\n";
            std::cout << "       (non-std exception)\n";
        }
    }

    std::cout << "\n";
    if (failed == 0) {
        std::cout << "✓ All tests passed.\n";
        return 0;
    }

    std::cout << "✗ " << failed << " test(s) failed.\n";
    return 1;
}
