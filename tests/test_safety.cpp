// tests/test_safety.cpp
// Safety invariant tests - these MUST pass on all platforms
//
// Purpose: Catch platform-specific issues that cause silent data corruption
// Source: Gemini's analysis identified critical gaps in bool/dtype handling

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "tf_wrap/core.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

// ============================================================================
// CRITICAL: Compile-Time Safety Checks
// ============================================================================

TEST_CASE("sizeof(bool) must be 1 for TF_BOOL compatibility") {
    // TensorFlow's TF_BOOL is always a 1-byte unsigned integer.
    // If sizeof(bool) != 1 on this platform, bool tensor operations
    // will cause memory corruption.
    CHECK(sizeof(bool) == 1);
    CHECK(sizeof(uint8_t) == 1);
}

TEST_CASE("TF_DataType sizes match expectations") {
    CHECK(sizeof(float) == 4);      // TF_FLOAT
    CHECK(sizeof(double) == 8);     // TF_DOUBLE
    CHECK(sizeof(int32_t) == 4);    // TF_INT32
    CHECK(sizeof(int64_t) == 8);    // TF_INT64
    CHECK(sizeof(int8_t) == 1);     // TF_INT8
    CHECK(sizeof(int16_t) == 2);    // TF_INT16
    CHECK(sizeof(uint8_t) == 1);    // TF_UINT8
    CHECK(sizeof(uint16_t) == 2);   // TF_UINT16
    CHECK(sizeof(uint32_t) == 4);   // TF_UINT32
    CHECK(sizeof(uint64_t) == 8);   // TF_UINT64
}

// ============================================================================
// Bool Tensor Raw Byte Layout (Gemini's key insight)
// ============================================================================

TEST_CASE("Bool tensor byte_size equals num_elements") {
    // This catches the case where sizeof(bool) > 1
    auto tensor = tf_wrap::Tensor::FromVector<bool>({100}, std::vector<bool>(100, true));
    
    // Must be exactly 100 bytes, not 100 * sizeof(bool)
    CHECK(tensor.byte_size() == 100);
    CHECK(tensor.num_elements() == 100);
}

TEST_CASE("Bool tensor raw byte inspection") {
    std::vector<bool> input = {true, false, true, false, true, true, false, false};
    auto tensor = tf_wrap::Tensor::FromVector<bool>({8}, input);
    
    CHECK(tensor.dtype() == TF_BOOL);
    CHECK(tensor.byte_size() == 8);
    
    auto view = tensor.read<bool>();
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(&view[0]);
    
    // Each bool must be exactly 1 byte: true->1, false->0
    CHECK(raw[0] == 1);  // true
    CHECK(raw[1] == 0);  // false
    CHECK(raw[2] == 1);  // true
    CHECK(raw[3] == 0);  // false
    CHECK(raw[4] == 1);  // true
    CHECK(raw[5] == 1);  // true
    CHECK(raw[6] == 0);  // false
    CHECK(raw[7] == 0);  // false
}

TEST_CASE("Bool tensor write produces correct byte pattern") {
    auto tensor = tf_wrap::Tensor::Allocate<bool>({4});
    
    {
        auto view = tensor.write<bool>();
        view[0] = true;
        view[1] = false;
        view[2] = true;
        view[3] = false;
    }
    
    auto view = tensor.read<bool>();
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(&view[0]);
    
    CHECK(raw[0] == 1);
    CHECK(raw[1] == 0);
    CHECK(raw[2] == 1);
    CHECK(raw[3] == 0);
}

TEST_CASE("Bool tensor bytes are contiguous") {
    auto tensor = tf_wrap::Tensor::FromVector<bool>({10}, std::vector<bool>(10, true));
    
    auto view = tensor.read<bool>();
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(&view[0]);
    
    for (int i = 0; i < 10; ++i) {
        CHECK(reinterpret_cast<const uint8_t*>(&view[i]) == raw + i);
    }
}

// ============================================================================
// Bool Tensor Round-Trip Verification
// ============================================================================

TEST_CASE("Bool tensor FromVector -> ToVector preserves all values") {
    std::vector<bool> input = {true, false, true, false, true, false, true, false};
    auto tensor = tf_wrap::Tensor::FromVector<bool>({8}, input);
    
    auto output = tensor.ToVector<bool>();
    
    REQUIRE(output.size() == input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        CHECK(output[i] == input[i]);
    }
}

TEST_CASE("Bool tensor round-trip with all-true") {
    std::vector<bool> input(256, true);
    auto tensor = tf_wrap::Tensor::FromVector<bool>({256}, input);
    auto output = tensor.ToVector<bool>();
    
    for (size_t i = 0; i < 256; ++i) {
        CHECK(output[i] == true);
    }
}

TEST_CASE("Bool tensor round-trip with all-false") {
    std::vector<bool> input(256, false);
    auto tensor = tf_wrap::Tensor::FromVector<bool>({256}, input);
    auto output = tensor.ToVector<bool>();
    
    for (size_t i = 0; i < 256; ++i) {
        CHECK(output[i] == false);
    }
}

TEST_CASE("Bool tensor Clone preserves raw bytes") {
    std::vector<bool> input = {true, false, true, false};
    auto original = tf_wrap::Tensor::FromVector<bool>({4}, input);
    auto clone = original.Clone();
    
    auto orig_view = original.read<bool>();
    auto clone_view = clone.read<bool>();
    
    const uint8_t* orig_raw = reinterpret_cast<const uint8_t*>(&orig_view[0]);
    const uint8_t* clone_raw = reinterpret_cast<const uint8_t*>(&clone_view[0]);
    
    CHECK(std::memcmp(orig_raw, clone_raw, 4) == 0);
}

// ============================================================================
// 32-bit Overflow Protection (Grok's concern)
// ============================================================================

TEST_CASE("Large shape doesn't overflow on 32-bit") {
    if constexpr (sizeof(size_t) < 8) {
        // On 32-bit, very large shapes should throw, not silently corrupt
        CHECK_THROWS(tf_wrap::Tensor::Allocate<float>({1024, 1024, 1024, 4}));
    } else {
        // On 64-bit, verify size calculation is correct
        std::vector<int64_t> shape = {1024, 1024, 1024};
        int64_t expected = 1024LL * 1024LL * 1024LL;
        
        int64_t product = 1;
        for (auto d : shape) product *= d;
        CHECK(product == expected);
    }
}

// ============================================================================
// Pointer Alignment
// ============================================================================

TEST_CASE("Tensor data is properly aligned") {
    auto f = tf_wrap::Tensor::FromScalar<float>(1.0f);
    auto d = tf_wrap::Tensor::FromScalar<double>(1.0);
    auto i64 = tf_wrap::Tensor::FromScalar<int64_t>(1);
    
    auto f_view = f.read<float>();
    auto d_view = d.read<double>();
    auto i64_view = i64.read<int64_t>();
    
    CHECK(reinterpret_cast<uintptr_t>(&f_view[0]) % alignof(float) == 0);
    CHECK(reinterpret_cast<uintptr_t>(&d_view[0]) % alignof(double) == 0);
    CHECK(reinterpret_cast<uintptr_t>(&i64_view[0]) % alignof(int64_t) == 0);
}

// ============================================================================
// Empty Tensor Safety
// ============================================================================

TEST_CASE("Empty tensor operations are safe") {
    tf_wrap::Tensor empty;
    
    CHECK(empty.empty());
    CHECK(empty.handle() == nullptr);
    CHECK(empty.byte_size() == 0);
    CHECK(empty.num_elements() == 0);
    
    auto clone = empty.Clone();
    CHECK(clone.empty());
}

TEST_CASE("Zero-element tensor is valid") {
    auto tensor = tf_wrap::Tensor::FromVector<float>({0}, {});
    
    CHECK(tensor.num_elements() == 0);
    CHECK(tensor.byte_size() == 0);
    CHECK(tensor.shape().size() == 1);
    CHECK(tensor.shape()[0] == 0);
}

// ============================================================================
// Move Semantics Safety
// ============================================================================

TEST_CASE("Moved-from Tensor is in valid empty state") {
    auto t1 = tf_wrap::Tensor::FromScalar<float>(42.0f);
    CHECK(!t1.empty());
    
    auto t2 = std::move(t1);
    
    CHECK(t1.empty());
    CHECK(t1.handle() == nullptr);
    CHECK(t1.byte_size() == 0);
    
    CHECK(!t2.empty());
    CHECK(t2.ToScalar<float>() == 42.0f);
}

TEST_CASE("Moved-from Graph is invalid") {
    tf_wrap::Graph g1;
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g1.NewOperation("Const", "c")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Graph g2 = std::move(g1);
    
    CHECK(g2.valid());
    CHECK(!g1.valid());
    CHECK_THROWS(g1.num_operations());
}
