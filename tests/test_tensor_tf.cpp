// test_tensor_tf.cpp
// Tensor tests with real TensorFlow C library
//
// Tests tensor creation, data access, and manipulation with real TF runtime.

#include "tf_wrap/tensor.hpp"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace tf_wrap;

// Simple test framework
static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    void test_##name(); \
    struct TestRunner_##name { \
        TestRunner_##name() { \
            std::cout << "Testing " #name "... " << std::flush; \
            tests_run++; \
            try { \
                test_##name(); \
                std::cout << "PASSED\n"; \
                tests_passed++; \
            } catch (const std::exception& e) { \
                std::cout << "FAILED: " << e.what() << "\n"; \
            } catch (...) { \
                std::cout << "FAILED: unknown exception\n"; \
            } \
        } \
    } test_runner_##name; \
    void test_##name()

#define REQUIRE(cond) \
    do { if (!(cond)) throw std::runtime_error("REQUIRE failed: " #cond); } while (0)

#define REQUIRE_CLOSE(a, b, eps) \
    do { \
        if (std::abs((a) - (b)) > (eps)) { \
            throw std::runtime_error("REQUIRE_CLOSE failed: " #a " vs " #b); \
        } \
    } while (0)

#define REQUIRE_THROWS(expr) \
    do { \
        bool threw = false; \
        try { (void)(expr); } catch (...) { threw = true; } \
        if (!threw) throw std::runtime_error("REQUIRE_THROWS failed: " #expr); \
    } while (0)

// ============================================================================
// FromScalar Tests - All dtypes
// ============================================================================

TEST(from_scalar_float) {
    auto t = Tensor::FromScalar<float>(3.14f);
    REQUIRE(t.valid());
    REQUIRE(t.dtype() == TF_FLOAT);
    REQUIRE(t.num_elements() == 1);
    REQUIRE(t.rank() == 0);
    REQUIRE_CLOSE(t.ToScalar<float>(), 3.14f, 0.001f);
}

TEST(from_scalar_double) {
    auto t = Tensor::FromScalar<double>(2.718281828);
    REQUIRE(t.dtype() == TF_DOUBLE);
    REQUIRE_CLOSE(t.ToScalar<double>(), 2.718281828, 0.0000001);
}

TEST(from_scalar_int32) {
    auto t = Tensor::FromScalar<std::int32_t>(-42);
    REQUIRE(t.dtype() == TF_INT32);
    REQUIRE(t.ToScalar<std::int32_t>() == -42);
}

TEST(from_scalar_int64) {
    auto t = Tensor::FromScalar<std::int64_t>(9223372036854775807LL);
    REQUIRE(t.dtype() == TF_INT64);
    REQUIRE(t.ToScalar<std::int64_t>() == 9223372036854775807LL);
}

TEST(from_scalar_uint8) {
    auto t = Tensor::FromScalar<std::uint8_t>(255);
    REQUIRE(t.dtype() == TF_UINT8);
    REQUIRE(t.ToScalar<std::uint8_t>() == 255);
}

TEST(from_scalar_bool_true) {
    auto t = Tensor::FromScalar<bool>(true);
    REQUIRE(t.dtype() == TF_BOOL);
    REQUIRE(t.ToScalar<bool>() == true);
}

TEST(from_scalar_bool_false) {
    auto t = Tensor::FromScalar<bool>(false);
    REQUIRE(t.dtype() == TF_BOOL);
    REQUIRE(t.ToScalar<bool>() == false);
}

TEST(from_scalar_complex64) {
    std::complex<float> val(1.0f, 2.0f);
    auto t = Tensor::FromScalar<std::complex<float>>(val);
    REQUIRE(t.dtype() == TF_COMPLEX64);
    auto result = t.ToScalar<std::complex<float>>();
    REQUIRE_CLOSE(result.real(), 1.0f, 0.001f);
    REQUIRE_CLOSE(result.imag(), 2.0f, 0.001f);
}

// ============================================================================
// FromVector Tests
// ============================================================================

TEST(from_vector_1d) {
    auto t = Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    REQUIRE(t.dtype() == TF_FLOAT);
    REQUIRE(t.num_elements() == 3);
    REQUIRE(t.rank() == 1);
    REQUIRE(t.shape()[0] == 3);
    
    auto view = t.read<float>();
    REQUIRE_CLOSE(view[0], 1.0f, 0.001f);
    REQUIRE_CLOSE(view[1], 2.0f, 0.001f);
    REQUIRE_CLOSE(view[2], 3.0f, 0.001f);
}

TEST(from_vector_2d) {
    auto t = Tensor::FromVector<std::int32_t>({2, 3}, {1, 2, 3, 4, 5, 6});
    REQUIRE(t.rank() == 2);
    REQUIRE(t.shape()[0] == 2);
    REQUIRE(t.shape()[1] == 3);
    REQUIRE(t.num_elements() == 6);
    
    auto vec = t.ToVector<std::int32_t>();
    REQUIRE(vec.size() == 6);
    REQUIRE(vec[0] == 1);
    REQUIRE(vec[5] == 6);
}

TEST(from_vector_3d) {
    std::vector<double> data(24);
    for (int i = 0; i < 24; ++i) data[i] = static_cast<double>(i);
    
    auto t = Tensor::FromVector<double>({2, 3, 4}, data);
    REQUIRE(t.rank() == 3);
    REQUIRE(t.num_elements() == 24);
}

TEST(from_vector_bool) {
    auto t = Tensor::FromVector<bool>({4}, {true, false, true, false});
    REQUIRE(t.dtype() == TF_BOOL);
    auto view = t.read<bool>();
    REQUIRE(view[0] == true);
    REQUIRE(view[1] == false);
    REQUIRE(view[2] == true);
    REQUIRE(view[3] == false);
}

TEST(from_vector_shape_mismatch_throws) {
    REQUIRE_THROWS(Tensor::FromVector<float>({2, 2}, {1.0f, 2.0f, 3.0f}));
}

// ============================================================================
// Zeros / Allocate Tests
// ============================================================================

TEST(zeros_float) {
    auto t = Tensor::Zeros<float>({2, 3});
    REQUIRE(t.num_elements() == 6);
    auto view = t.read<float>();
    for (std::size_t i = 0; i < 6; ++i) {
        REQUIRE(view[i] == 0.0f);
    }
}

TEST(allocate_and_write) {
    auto t = Tensor::Allocate<std::int32_t>({4});
    auto view = t.write<std::int32_t>();
    for (std::size_t i = 0; i < 4; ++i) {
        view[i] = static_cast<std::int32_t>(i * 10);
    }
    
    auto read_view = t.read<std::int32_t>();
    REQUIRE(read_view[0] == 0);
    REQUIRE(read_view[1] == 10);
    REQUIRE(read_view[2] == 20);
    REQUIRE(read_view[3] == 30);
}

// ============================================================================
// String Tensor Tests
// ============================================================================

TEST(from_string_basic) {
    auto t = Tensor::FromString("hello world");
    REQUIRE(t.dtype() == TF_STRING);
    REQUIRE(t.num_elements() == 1);
    REQUIRE(t.ToString() == "hello world");
}

TEST(from_string_empty) {
    auto t = Tensor::FromString("");
    REQUIRE(t.ToString() == "");
}

TEST(from_string_unicode) {
    auto t = Tensor::FromString("日本語テスト 🎉");
    REQUIRE(t.ToString() == "日本語テスト 🎉");
}

// ============================================================================
// Clone Tests
// ============================================================================

TEST(clone_independence) {
    auto original = Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    auto cloned = original.Clone();
    
    REQUIRE(cloned.dtype() == original.dtype());
    REQUIRE(cloned.shape() == original.shape());
    
    // Modify clone
    cloned.write<float>()[0] = 999.0f;
    
    // Original should be unchanged
    REQUIRE_CLOSE(original.read<float>()[0], 1.0f, 0.001f);
    REQUIRE_CLOSE(cloned.read<float>()[0], 999.0f, 0.001f);
}

TEST(clone_string) {
    auto original = Tensor::FromString("test string");
    auto cloned = original.Clone();
    REQUIRE(cloned.ToString() == "test string");
}

// ============================================================================
// Reshape Tests
// ============================================================================

TEST(reshape_1d_to_2d) {
    auto t = Tensor::FromVector<float>({6}, {1, 2, 3, 4, 5, 6});
    auto reshaped = t.reshape({2, 3});
    
    REQUIRE(reshaped.shape()[0] == 2);
    REQUIRE(reshaped.shape()[1] == 3);
    REQUIRE(reshaped.num_elements() == 6);
}

TEST(reshape_shares_data) {
    auto t = Tensor::FromVector<std::int32_t>({4}, {1, 2, 3, 4});
    auto reshaped = t.reshape({2, 2});
    
    // Modify through original
    t.write<std::int32_t>()[0] = 999;
    
    // Reshaped should see the change
    REQUIRE(reshaped.read<std::int32_t>()[0] == 999);
}

TEST(reshape_mismatch_throws) {
    auto t = Tensor::FromVector<float>({6}, {1, 2, 3, 4, 5, 6});
    REQUIRE_THROWS(t.reshape({2, 2}));  // 4 != 6
}

// ============================================================================
// Type Safety Tests
// ============================================================================

TEST(dtype_mismatch_read_throws) {
    auto t = Tensor::FromScalar<float>(1.0f);
    REQUIRE_THROWS(t.read<std::int32_t>());
}

TEST(dtype_mismatch_write_throws) {
    auto t = Tensor::FromScalar<float>(1.0f);
    REQUIRE_THROWS(t.write<double>());
}

TEST(dtype_mismatch_to_scalar_throws) {
    auto t = Tensor::FromScalar<std::int32_t>(42);
    REQUIRE_THROWS(t.ToScalar<float>());
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST(move_constructor) {
    auto t1 = Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    auto t2 = std::move(t1);
    
    REQUIRE(t2.valid());
    REQUIRE(t2.num_elements() == 3);
    REQUIRE(!t1.valid());
}

TEST(moved_from_state_safe) {
    auto t1 = Tensor::FromScalar<float>(1.0f);
    auto t2 = std::move(t1);
    
    REQUIRE(!t1.valid());
    REQUIRE(t1.empty());
    REQUIRE(t1.byte_size() == 0);
    REQUIRE(t1.handle() == nullptr);
}

// ============================================================================
// View Lifetime Tests
// ============================================================================

TEST(view_keeps_tensor_alive) {
    TensorView<const float> view = []{
        auto t = Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
        return t.read<float>();
    }();
    
    // View should still be valid
    REQUIRE(view.size() == 3);
    REQUIRE_CLOSE(view[0], 1.0f, 0.001f);
}

// ============================================================================
// Large Tensor Test
// ============================================================================

TEST(large_tensor) {
    auto t = Tensor::Zeros<float>({1000, 1000});
    REQUIRE(t.num_elements() == 1000000);
    REQUIRE(t.byte_size() == 1000000 * sizeof(float));
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== TensorFlowWrap Tensor Tests (Real TF) ===\n\n";
    
    // Tests run automatically via static initialization
    
    std::cout << "\n=== Results: " << tests_passed << "/" << tests_run << " passed ===\n";
    
    return (tests_passed == tests_run) ? 0 : 1;
}
