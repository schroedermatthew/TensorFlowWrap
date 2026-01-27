// test_tensor.cpp
// Comprehensive tests for tf_wrap::Tensor
//
// These tests cover:
// - Factory methods: FromScalar, FromVector, Zeros, Allocate, FromString, Clone
// - Data access: read/write views, ToScalar, ToVector, ToString
// - Shape operations: reshape, matches_shape
// - Type safety: dtype checking, type mismatches
// - Edge cases: empty tensors, moved-from state, large tensors
// - All supported dtypes

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "tf_wrap/tensor.hpp"

#include <cmath>
#include <complex>
#include <limits>
#include <vector>

using namespace tf_wrap;

// ============================================================================
// FromScalar Tests
// ============================================================================

TEST_CASE("Tensor::FromScalar - float") {
    auto t = Tensor::FromScalar<float>(3.14f);
    
    CHECK(t.valid());
    CHECK(t.dtype() == TF_FLOAT);
    CHECK(t.num_elements() == 1);
    CHECK(t.rank() == 0);
    CHECK(t.shape().empty());
    CHECK(t.ToScalar<float>() == doctest::Approx(3.14f));
}

TEST_CASE("Tensor::FromScalar - double") {
    auto t = Tensor::FromScalar<double>(2.718281828);
    
    CHECK(t.dtype() == TF_DOUBLE);
    CHECK(t.num_elements() == 1);
    CHECK(t.ToScalar<double>() == doctest::Approx(2.718281828));
}

TEST_CASE("Tensor::FromScalar - int32") {
    auto t = Tensor::FromScalar<std::int32_t>(-42);
    
    CHECK(t.dtype() == TF_INT32);
    CHECK(t.ToScalar<std::int32_t>() == -42);
}

TEST_CASE("Tensor::FromScalar - int64") {
    auto t = Tensor::FromScalar<std::int64_t>(9223372036854775807LL);
    
    CHECK(t.dtype() == TF_INT64);
    CHECK(t.ToScalar<std::int64_t>() == 9223372036854775807LL);
}

TEST_CASE("Tensor::FromScalar - uint8") {
    auto t = Tensor::FromScalar<std::uint8_t>(255);
    
    CHECK(t.dtype() == TF_UINT8);
    CHECK(t.ToScalar<std::uint8_t>() == 255);
}

TEST_CASE("Tensor::FromScalar - uint16") {
    auto t = Tensor::FromScalar<std::uint16_t>(65535);
    
    CHECK(t.dtype() == TF_UINT16);
    CHECK(t.ToScalar<std::uint16_t>() == 65535);
}

TEST_CASE("Tensor::FromScalar - uint32") {
    auto t = Tensor::FromScalar<std::uint32_t>(4294967295U);
    
    CHECK(t.dtype() == TF_UINT32);
    CHECK(t.ToScalar<std::uint32_t>() == 4294967295U);
}

TEST_CASE("Tensor::FromScalar - uint64") {
    auto t = Tensor::FromScalar<std::uint64_t>(18446744073709551615ULL);
    
    CHECK(t.dtype() == TF_UINT64);
    CHECK(t.ToScalar<std::uint64_t>() == 18446744073709551615ULL);
}

TEST_CASE("Tensor::FromScalar - int8") {
    auto t = Tensor::FromScalar<std::int8_t>(-128);
    
    CHECK(t.dtype() == TF_INT8);
    CHECK(t.ToScalar<std::int8_t>() == -128);
}

TEST_CASE("Tensor::FromScalar - int16") {
    auto t = Tensor::FromScalar<std::int16_t>(-32768);
    
    CHECK(t.dtype() == TF_INT16);
    CHECK(t.ToScalar<std::int16_t>() == -32768);
}

TEST_CASE("Tensor::FromScalar - bool true") {
    auto t = Tensor::FromScalar<bool>(true);
    
    CHECK(t.dtype() == TF_BOOL);
    CHECK(t.ToScalar<bool>() == true);
}

TEST_CASE("Tensor::FromScalar - bool false") {
    auto t = Tensor::FromScalar<bool>(false);
    
    CHECK(t.dtype() == TF_BOOL);
    CHECK(t.ToScalar<bool>() == false);
}

TEST_CASE("Tensor::FromScalar - complex64") {
    std::complex<float> val(1.0f, 2.0f);
    auto t = Tensor::FromScalar<std::complex<float>>(val);
    
    CHECK(t.dtype() == TF_COMPLEX64);
    auto result = t.ToScalar<std::complex<float>>();
    CHECK(result.real() == doctest::Approx(1.0f));
    CHECK(result.imag() == doctest::Approx(2.0f));
}

TEST_CASE("Tensor::FromScalar - complex128") {
    std::complex<double> val(3.0, 4.0);
    auto t = Tensor::FromScalar<std::complex<double>>(val);
    
    CHECK(t.dtype() == TF_COMPLEX128);
    auto result = t.ToScalar<std::complex<double>>();
    CHECK(result.real() == doctest::Approx(3.0));
    CHECK(result.imag() == doctest::Approx(4.0));
}

// ============================================================================
// FromVector Tests
// ============================================================================

TEST_CASE("Tensor::FromVector - 1D float") {
    auto t = Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    
    CHECK(t.valid());
    CHECK(t.dtype() == TF_FLOAT);
    CHECK(t.num_elements() == 3);
    CHECK(t.rank() == 1);
    CHECK(t.shape() == std::vector<std::int64_t>{3});
    
    auto view = t.read<float>();
    CHECK(view[0] == doctest::Approx(1.0f));
    CHECK(view[1] == doctest::Approx(2.0f));
    CHECK(view[2] == doctest::Approx(3.0f));
}

TEST_CASE("Tensor::FromVector - 2D int32") {
    auto t = Tensor::FromVector<std::int32_t>({2, 3}, {1, 2, 3, 4, 5, 6});
    
    CHECK(t.dtype() == TF_INT32);
    CHECK(t.num_elements() == 6);
    CHECK(t.rank() == 2);
    CHECK(t.shape() == std::vector<std::int64_t>{2, 3});
    
    auto vec = t.ToVector<std::int32_t>();
    CHECK(vec == std::vector<std::int32_t>{1, 2, 3, 4, 5, 6});
}

TEST_CASE("Tensor::FromVector - 3D double") {
    std::vector<double> data(24);
    for (int i = 0; i < 24; ++i) data[i] = static_cast<double>(i);
    
    auto t = Tensor::FromVector<double>({2, 3, 4}, data);
    
    CHECK(t.dtype() == TF_DOUBLE);
    CHECK(t.num_elements() == 24);
    CHECK(t.rank() == 3);
    CHECK(t.shape() == std::vector<std::int64_t>{2, 3, 4});
}

TEST_CASE("Tensor::FromVector - bool array") {
    auto t = Tensor::FromVector<bool>({4}, {true, false, true, false});
    
    CHECK(t.dtype() == TF_BOOL);
    CHECK(t.num_elements() == 4);
    
    auto view = t.read<bool>();
    CHECK(view[0] == true);
    CHECK(view[1] == false);
    CHECK(view[2] == true);
    CHECK(view[3] == false);
}

TEST_CASE("Tensor::FromVector - shape mismatch throws") {
    bool threw = false;
    try {
        auto t = Tensor::FromVector<float>({2, 2}, {1.0f, 2.0f, 3.0f});  // 4 expected, 3 given
        (void)t;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("Tensor::FromVector - empty tensor") {
    auto t = Tensor::FromVector<float>({0}, std::vector<float>{});
    
    CHECK(t.valid());
    CHECK(t.num_elements() == 0);
    CHECK(t.empty());
    CHECK(t.shape() == std::vector<std::int64_t>{0});
}

TEST_CASE("Tensor::FromVector - with std::vector") {
    std::vector<std::int64_t> shape = {2, 2};
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    
    auto t = Tensor::FromVector<float>(shape, data);
    
    CHECK(t.num_elements() == 4);
    CHECK(t.ToVector<float>() == data);
}

// ============================================================================
// Zeros / Allocate Tests
// ============================================================================

TEST_CASE("Tensor::Zeros - float") {
    auto t = Tensor::Zeros<float>({2, 3});
    
    CHECK(t.dtype() == TF_FLOAT);
    CHECK(t.num_elements() == 6);
    
    auto view = t.read<float>();
    for (std::size_t i = 0; i < 6; ++i) {
        CHECK(view[i] == 0.0f);
    }
}

TEST_CASE("Tensor::Zeros - int32") {
    auto t = Tensor::Zeros<std::int32_t>({10});
    
    CHECK(t.dtype() == TF_INT32);
    auto view = t.read<std::int32_t>();
    for (std::size_t i = 0; i < 10; ++i) {
        CHECK(view[i] == 0);
    }
}

TEST_CASE("Tensor::Zeros - empty shape") {
    auto t = Tensor::Zeros<float>({0, 5});
    
    CHECK(t.num_elements() == 0);
    CHECK(t.empty());
}

TEST_CASE("Tensor::Allocate - float") {
    auto t = Tensor::Allocate<float>({3, 3});
    
    CHECK(t.valid());
    CHECK(t.dtype() == TF_FLOAT);
    CHECK(t.num_elements() == 9);
    CHECK(t.byte_size() == 9 * sizeof(float));
}

TEST_CASE("Tensor::Allocate - write data") {
    auto t = Tensor::Allocate<std::int32_t>({4});
    
    auto view = t.write<std::int32_t>();
    for (std::size_t i = 0; i < 4; ++i) {
        view[i] = static_cast<std::int32_t>(i * 10);
    }
    
    auto read_view = t.read<std::int32_t>();
    CHECK(read_view[0] == 0);
    CHECK(read_view[1] == 10);
    CHECK(read_view[2] == 20);
    CHECK(read_view[3] == 30);
}

// ============================================================================
// String Tensor Tests
// ============================================================================

TEST_CASE("Tensor::FromString - basic") {
    auto t = Tensor::FromString("hello world");
    
    CHECK(t.valid());
    CHECK(t.dtype() == TF_STRING);
    CHECK(t.num_elements() == 1);
    CHECK(t.ToString() == "hello world");
}

TEST_CASE("Tensor::FromString - empty string") {
    auto t = Tensor::FromString("");
    
    CHECK(t.dtype() == TF_STRING);
    CHECK(t.ToString() == "");
}

TEST_CASE("Tensor::FromString - unicode") {
    auto t = Tensor::FromString("日本語テスト 🎉");
    
    CHECK(t.ToString() == "日本語テスト 🎉");
}

TEST_CASE("Tensor::ToString - wrong dtype throws") {
    auto t = Tensor::FromScalar<float>(1.0f);
    
    bool threw = false;
    try {
        auto s = t.ToString();
        (void)s;
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

// ============================================================================
// Clone Tests
// ============================================================================

TEST_CASE("Tensor::Clone - float") {
    auto original = Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    auto cloned = original.Clone();
    
    CHECK(cloned.valid());
    CHECK(cloned.dtype() == original.dtype());
    CHECK(cloned.shape() == original.shape());
    CHECK(cloned.ToVector<float>() == original.ToVector<float>());
    
    // Verify independence - modify clone
    cloned.write<float>()[0] = 999.0f;
    CHECK(original.read<float>()[0] == doctest::Approx(1.0f));
    CHECK(cloned.read<float>()[0] == doctest::Approx(999.0f));
}

TEST_CASE("Tensor::Clone - string") {
    auto original = Tensor::FromString("test string");
    auto cloned = original.Clone();
    
    CHECK(cloned.ToString() == "test string");
}

TEST_CASE("Tensor::Clone - empty tensor") {
    Tensor empty;
    auto cloned = empty.Clone();
    
    CHECK_FALSE(cloned.valid());
}

// ============================================================================
// Reshape Tests
// ============================================================================

TEST_CASE("Tensor::reshape - 1D to 2D") {
    auto t = Tensor::FromVector<float>({6}, {1, 2, 3, 4, 5, 6});
    auto reshaped = t.reshape({2, 3});
    
    CHECK(reshaped.valid());
    CHECK(reshaped.shape() == std::vector<std::int64_t>{2, 3});
    CHECK(reshaped.num_elements() == 6);
    
    // Verify data is shared (zero-copy)
    auto view = reshaped.read<float>();
    CHECK(view[0] == doctest::Approx(1.0f));
    CHECK(view[5] == doctest::Approx(6.0f));
}

TEST_CASE("Tensor::reshape - 2D to 1D") {
    auto t = Tensor::FromVector<std::int32_t>({2, 3}, {1, 2, 3, 4, 5, 6});
    auto reshaped = t.reshape({6});
    
    CHECK(reshaped.shape() == std::vector<std::int64_t>{6});
    CHECK(reshaped.ToVector<std::int32_t>() == std::vector<std::int32_t>{1, 2, 3, 4, 5, 6});
}

TEST_CASE("Tensor::reshape - to scalar") {
    auto t = Tensor::FromVector<float>({1}, {42.0f});
    auto reshaped = t.reshape({});  // scalar
    
    CHECK(reshaped.rank() == 0);
    CHECK(reshaped.num_elements() == 1);
    CHECK(reshaped.ToScalar<float>() == doctest::Approx(42.0f));
}

TEST_CASE("Tensor::reshape - element count mismatch throws") {
    auto t = Tensor::FromVector<float>({6}, {1, 2, 3, 4, 5, 6});
    
    bool threw = false;
    try {
        auto r = t.reshape({2, 2});  // 4 != 6
        (void)r;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("Tensor::reshape - shares data (zero-copy)") {
    auto t = Tensor::FromVector<std::int32_t>({4}, {1, 2, 3, 4});
    auto reshaped = t.reshape({2, 2});
    
    // Modify through original
    t.write<std::int32_t>()[0] = 999;
    
    // Reshaped should see the change
    CHECK(reshaped.read<std::int32_t>()[0] == 999);
}

// ============================================================================
// matches_shape Tests
// ============================================================================

TEST_CASE("Tensor::matches_shape - exact match") {
    auto t = Tensor::FromVector<float>({2, 3}, {1, 2, 3, 4, 5, 6});
    
    CHECK(t.matches_shape({2, 3}));
    CHECK_FALSE(t.matches_shape({3, 2}));
    CHECK_FALSE(t.matches_shape({6}));
    CHECK_FALSE(t.matches_shape({2, 3, 1}));
}

TEST_CASE("Tensor::matches_shape - scalar") {
    auto t = Tensor::FromScalar<float>(1.0f);
    
    CHECK(t.matches_shape({}));
    CHECK_FALSE(t.matches_shape({1}));
}

// ============================================================================
// Read/Write View Tests
// ============================================================================

TEST_CASE("TensorView - iteration") {
    auto t = Tensor::FromVector<std::int32_t>({5}, {10, 20, 30, 40, 50});
    
    auto view = t.read<std::int32_t>();
    
    std::int32_t sum = 0;
    for (auto val : view) {
        sum += val;
    }
    CHECK(sum == 150);
}

TEST_CASE("TensorView - at() bounds checking") {
    auto t = Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    
    auto view = t.read<float>();
    
    CHECK(view.at(0) == doctest::Approx(1.0f));
    CHECK(view.at(2) == doctest::Approx(3.0f));
    
    bool threw = false;
    try {
        auto v = view.at(3);
        (void)v;
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("TensorView - front/back") {
    auto t = Tensor::FromVector<std::int32_t>({4}, {1, 2, 3, 4});
    
    auto view = t.read<std::int32_t>();
    
    CHECK(view.front() == 1);
    CHECK(view.back() == 4);
}

TEST_CASE("TensorView - size/empty") {
    auto t = Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    
    auto view = t.read<float>();
    
    CHECK(view.size() == 3);
    CHECK(view.size_bytes() == 3 * sizeof(float));
    CHECK_FALSE(view.empty());
}

TEST_CASE("WriteView - modification") {
    auto t = Tensor::Allocate<float>({3});
    
    {
        auto view = t.write<float>();
        view[0] = 1.0f;
        view[1] = 2.0f;
        view[2] = 3.0f;
    }
    
    CHECK(t.ToVector<float>() == std::vector<float>{1.0f, 2.0f, 3.0f});
}

TEST_CASE("TensorView - keeps tensor alive") {
    TensorView<const float> view = []{
        auto t = Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
        return t.read<float>();  // Returns view, tensor goes out of scope
    }();
    
    // View should still be valid because it holds shared_ptr to state
    CHECK(view.size() == 3);
    CHECK(view[0] == doctest::Approx(1.0f));
    CHECK(view[1] == doctest::Approx(2.0f));
    CHECK(view[2] == doctest::Approx(3.0f));
}

// ============================================================================
// with_read / with_write Tests
// ============================================================================

TEST_CASE("Tensor::with_read") {
    auto t = Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    
    float sum = t.with_read<float>([](std::span<const float> data) {
        float s = 0;
        for (auto v : data) s += v;
        return s;
    });
    
    CHECK(sum == doctest::Approx(6.0f));
}

TEST_CASE("Tensor::with_write") {
    auto t = Tensor::Allocate<std::int32_t>({4});
    
    t.with_write<std::int32_t>([](std::span<std::int32_t> data) {
        for (std::size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<std::int32_t>(i * i);
        }
    });
    
    CHECK(t.ToVector<std::int32_t>() == std::vector<std::int32_t>{0, 1, 4, 9});
}

// ============================================================================
// Type Safety Tests
// ============================================================================

TEST_CASE("Tensor - dtype mismatch on read throws") {
    auto t = Tensor::FromScalar<float>(1.0f);
    
    bool threw = false;
    try {
        auto v = t.read<std::int32_t>();
        (void)v;
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("Tensor - dtype mismatch on write throws") {
    auto t = Tensor::FromScalar<float>(1.0f);
    
    bool threw = false;
    try {
        auto v = t.write<double>();
        (void)v;
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("Tensor - dtype mismatch on ToScalar throws") {
    auto t = Tensor::FromScalar<std::int32_t>(42);
    
    bool threw = false;
    try {
        auto v = t.ToScalar<float>();
        (void)v;
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("Tensor - dtype mismatch on ToVector throws") {
    auto t = Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    
    bool threw = false;
    try {
        auto v = t.ToVector<std::int32_t>();
        (void)v;
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_CASE("Tensor - move constructor") {
    auto t1 = Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    auto t2 = std::move(t1);
    
    CHECK(t2.valid());
    CHECK(t2.num_elements() == 3);
    CHECK_FALSE(t1.valid());  // Moved-from
}

TEST_CASE("Tensor - move assignment") {
    auto t1 = Tensor::FromScalar<float>(1.0f);
    auto t2 = Tensor::FromScalar<float>(2.0f);
    
    t1 = std::move(t2);
    
    CHECK(t1.valid());
    CHECK(t1.ToScalar<float>() == doctest::Approx(2.0f));
    CHECK_FALSE(t2.valid());  // Moved-from
}

TEST_CASE("Tensor - moved-from state is safe") {
    auto t1 = Tensor::FromScalar<float>(1.0f);
    auto t2 = std::move(t1);
    
    // These should not crash on moved-from tensor
    CHECK_FALSE(t1.valid());
    CHECK(t1.empty());
    CHECK(t1.byte_size() == 0);
    CHECK(t1.num_elements() == 0);
    CHECK(t1.handle() == nullptr);
    
    // dtype() on moved-from should throw
    bool threw = false;
    try {
        auto d = t1.dtype();
        (void)d;
    } catch (...) {
        threw = true;
    }
    CHECK(threw);
}

// ============================================================================
// Empty/Default Tensor Tests
// ============================================================================

TEST_CASE("Tensor - default constructed") {
    Tensor t;
    
    CHECK_FALSE(t.valid());
    CHECK(t.empty());
    CHECK(t.handle() == nullptr);
    CHECK(t.byte_size() == 0);
    CHECK(t.num_elements() == 0);
}

TEST_CASE("Tensor - operations on empty tensor throw") {
    Tensor t;
    
    auto throws_on = [&](auto fn) {
        bool threw = false;
        try { fn(); } catch (...) { threw = true; }
        return threw;
    };
    
    CHECK(throws_on([&]{ return t.dtype(); }));
    CHECK(throws_on([&]{ return t.read<float>(); }));
    CHECK(throws_on([&]{ return t.write<float>(); }));
    CHECK(throws_on([&]{ return t.ToScalar<float>(); }));
    CHECK(throws_on([&]{ return t.reshape({1}); }));
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_CASE("Tensor - very large tensor") {
    // 1M elements
    auto t = Tensor::Zeros<float>({1000, 1000});
    
    CHECK(t.num_elements() == 1000000);
    CHECK(t.byte_size() == 1000000 * sizeof(float));
}

TEST_CASE("Tensor - single element treated as array") {
    auto t = Tensor::FromVector<float>({1}, {42.0f});
    
    CHECK(t.rank() == 1);
    CHECK(t.num_elements() == 1);
    CHECK(t.shape() == std::vector<std::int64_t>{1});
    
    // Can read as array
    auto view = t.read<float>();
    CHECK(view.size() == 1);
    CHECK(view[0] == doctest::Approx(42.0f));
}

TEST_CASE("Tensor - high rank tensor") {
    auto t = Tensor::Zeros<float>({2, 2, 2, 2, 2});  // rank 5
    
    CHECK(t.rank() == 5);
    CHECK(t.num_elements() == 32);
    CHECK(t.shape() == std::vector<std::int64_t>{2, 2, 2, 2, 2});
}

TEST_CASE("Tensor - negative dimension throws") {
    bool threw = false;
    try {
        auto t = Tensor::Zeros<float>({2, -1, 3});
        (void)t;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

// ============================================================================
// dtype_name Tests
// ============================================================================

TEST_CASE("dtype_name - all types") {
    CHECK(std::string(dtype_name(TF_FLOAT)) == "float32");
    CHECK(std::string(dtype_name(TF_DOUBLE)) == "float64");
    CHECK(std::string(dtype_name(TF_INT8)) == "int8");
    CHECK(std::string(dtype_name(TF_INT16)) == "int16");
    CHECK(std::string(dtype_name(TF_INT32)) == "int32");
    CHECK(std::string(dtype_name(TF_INT64)) == "int64");
    CHECK(std::string(dtype_name(TF_UINT8)) == "uint8");
    CHECK(std::string(dtype_name(TF_UINT16)) == "uint16");
    CHECK(std::string(dtype_name(TF_UINT32)) == "uint32");
    CHECK(std::string(dtype_name(TF_UINT64)) == "uint64");
    CHECK(std::string(dtype_name(TF_BOOL)) == "bool");
    CHECK(std::string(dtype_name(TF_STRING)) == "string");
    CHECK(std::string(dtype_name(TF_COMPLEX64)) == "complex64");
    CHECK(std::string(dtype_name(TF_COMPLEX128)) == "complex128");
}

TEST_CASE("Tensor::dtype_name method") {
    auto t = Tensor::FromScalar<float>(1.0f);
    CHECK(std::string(t.dtype_name()) == "float32");
}

// ============================================================================
// tf_dtype_v Tests
// ============================================================================

TEST_CASE("tf_dtype_v - compile time mapping") {
    static_assert(tf_dtype_v<float> == TF_FLOAT);
    static_assert(tf_dtype_v<double> == TF_DOUBLE);
    static_assert(tf_dtype_v<std::int32_t> == TF_INT32);
    static_assert(tf_dtype_v<std::int64_t> == TF_INT64);
    static_assert(tf_dtype_v<bool> == TF_BOOL);
    static_assert(tf_dtype_v<std::complex<float>> == TF_COMPLEX64);
    static_assert(tf_dtype_v<std::complex<double>> == TF_COMPLEX128);
}

// ============================================================================
// Keepalive Tests
// ============================================================================

TEST_CASE("Tensor::keepalive - extends lifetime") {
    std::shared_ptr<const void> keepalive;
    
    {
        auto t = Tensor::FromScalar<float>(42.0f);
        keepalive = t.keepalive();
    }
    // Tensor t is destroyed, but keepalive holds the state
    
    // The handle should still be valid (though we can't easily verify without UB)
    CHECK(keepalive != nullptr);
}

// ============================================================================
// AdoptMalloc Tests  
// ============================================================================

TEST_CASE("Tensor::AdoptMalloc - basic") {
    float* data = static_cast<float*>(std::malloc(3 * sizeof(float)));
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    
    std::vector<std::int64_t> shape = {3};
    auto t = Tensor::AdoptMalloc<float>(shape, data, 3 * sizeof(float));
    
    CHECK(t.valid());
    CHECK(t.dtype() == TF_FLOAT);
    CHECK(t.num_elements() == 3);
    
    auto vec = t.ToVector<float>();
    CHECK(vec == std::vector<float>{1.0f, 2.0f, 3.0f});
    
    // data is freed when tensor is destroyed
}

TEST_CASE("Tensor::AdoptMalloc - byte_len mismatch throws") {
    float* data = static_cast<float*>(std::malloc(3 * sizeof(float)));
    
    std::vector<std::int64_t> shape = {3};
    bool threw = false;
    try {
        auto t = Tensor::AdoptMalloc<float>(shape, data, 2 * sizeof(float));  // Wrong size
        (void)t;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    
    std::free(data);  // Clean up since AdoptMalloc didn't take ownership
}

TEST_CASE("Tensor::AdoptMalloc - null data with zero size ok") {
    std::vector<std::int64_t> shape = {0};
    auto t = Tensor::AdoptMalloc<float>(shape, nullptr, 0);
    
    CHECK(t.valid());
    CHECK(t.num_elements() == 0);
}

TEST_CASE("Tensor::AdoptMalloc - null data with non-zero size throws") {
    std::vector<std::int64_t> shape = {3};
    bool threw = false;
    try {
        auto t = Tensor::AdoptMalloc<float>(shape, nullptr, 12);
        (void)t;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}
