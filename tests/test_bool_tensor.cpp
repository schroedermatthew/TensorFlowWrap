// tests/test_bool_tensor.cpp
// Comprehensive bool tensor tests
//
// Purpose: Verify bool tensors work correctly despite std::vector<bool> weirdness
// Source: Gemini identified bool size trap as CRITICAL bug

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "tf_wrap/all.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

// ============================================================================
// Basic Bool Tensor Creation
// ============================================================================

TEST_CASE("Bool tensor from scalar") {
    auto t_true = tf_wrap::Tensor::FromScalar<bool>(true);
    auto t_false = tf_wrap::Tensor::FromScalar<bool>(false);
    
    CHECK(t_true.dtype() == TF_BOOL);
    CHECK(t_false.dtype() == TF_BOOL);
    CHECK(t_true.num_elements() == 1);
    CHECK(t_false.num_elements() == 1);
    
    CHECK(t_true.ToScalar<bool>() == true);
    CHECK(t_false.ToScalar<bool>() == false);
}

TEST_CASE("Bool tensor from vector") {
    std::vector<bool> input = {true, false, true, false};
    auto tensor = tf_wrap::Tensor::FromVector<bool>({4}, input);
    
    CHECK(tensor.dtype() == TF_BOOL);
    CHECK(tensor.num_elements() == 4);
    CHECK(tensor.byte_size() == 4);  // 1 byte per element
    
    auto output = tensor.ToVector<bool>();
    REQUIRE(output.size() == 4);
    CHECK(output[0] == true);
    CHECK(output[1] == false);
    CHECK(output[2] == true);
    CHECK(output[3] == false);
}

TEST_CASE("Bool tensor 2D") {
    std::vector<bool> input = {true, false, true, false, true, false};
    auto tensor = tf_wrap::Tensor::FromVector<bool>({2, 3}, input);
    
    CHECK(tensor.dtype() == TF_BOOL);
    CHECK(tensor.shape().size() == 2);
    CHECK(tensor.shape()[0] == 2);
    CHECK(tensor.shape()[1] == 3);
    CHECK(tensor.num_elements() == 6);
    CHECK(tensor.byte_size() == 6);
}

TEST_CASE("Bool tensor 3D") {
    std::vector<bool> input(24, true);
    auto tensor = tf_wrap::Tensor::FromVector<bool>({2, 3, 4}, input);
    
    CHECK(tensor.shape().size() == 3);
    CHECK(tensor.num_elements() == 24);
    CHECK(tensor.byte_size() == 24);
}

// ============================================================================
// Read/Write Views
// ============================================================================

TEST_CASE("Bool tensor read view") {
    std::vector<bool> input = {true, false, true, false, true};
    auto tensor = tf_wrap::Tensor::FromVector<bool>({5}, input);
    
    auto view = tensor.read<bool>();
    CHECK(view.size() == 5);
    CHECK(view[0] == true);
    CHECK(view[1] == false);
    CHECK(view[2] == true);
    CHECK(view[3] == false);
    CHECK(view[4] == true);
}

TEST_CASE("Bool tensor write view") {
    auto tensor = tf_wrap::Tensor::Allocate<bool>({4});
    
    {
        auto view = tensor.write<bool>();
        view[0] = true;
        view[1] = false;
        view[2] = true;
        view[3] = false;
    }
    
    auto read = tensor.read<bool>();
    CHECK(read[0] == true);
    CHECK(read[1] == false);
    CHECK(read[2] == true);
    CHECK(read[3] == false);
}

TEST_CASE("Bool tensor modify via write view") {
    auto tensor = tf_wrap::Tensor::FromVector<bool>({4}, {false, false, false, false});
    
    {
        auto view = tensor.write<bool>();
        view[1] = true;
        view[3] = true;
    }
    
    auto output = tensor.ToVector<bool>();
    CHECK(output[0] == false);
    CHECK(output[1] == true);
    CHECK(output[2] == false);
    CHECK(output[3] == true);
}

// ============================================================================
// Raw Byte Layout (CRITICAL - Gemini)
// ============================================================================

TEST_CASE("Bool tensor raw bytes are 0x00/0x01") {
    std::vector<bool> input = {true, false, true, false};
    auto tensor = tf_wrap::Tensor::FromVector<bool>({4}, input);
    
    auto view = tensor.read<bool>();
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(&view[0]);
    
    CHECK(raw[0] == 1);  // true  = 0x01
    CHECK(raw[1] == 0);  // false = 0x00
    CHECK(raw[2] == 1);  // true  = 0x01
    CHECK(raw[3] == 0);  // false = 0x00
}

TEST_CASE("Bool tensor write produces correct raw bytes") {
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

TEST_CASE("Bool tensor elements are 1 byte apart") {
    auto tensor = tf_wrap::Tensor::FromVector<bool>({8}, std::vector<bool>(8, true));
    
    auto view = tensor.read<bool>();
    const uint8_t* base = reinterpret_cast<const uint8_t*>(&view[0]);
    
    for (int i = 0; i < 8; ++i) {
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&view[i]);
        CHECK(ptr == base + i);
    }
}

// ============================================================================
// Clone
// ============================================================================

TEST_CASE("Bool tensor Clone preserves values") {
    std::vector<bool> input = {true, false, true, false};
    auto original = tf_wrap::Tensor::FromVector<bool>({4}, input);
    auto clone = original.Clone();
    
    CHECK(clone.dtype() == TF_BOOL);
    CHECK(clone.num_elements() == 4);
    
    auto output = clone.ToVector<bool>();
    CHECK(output[0] == true);
    CHECK(output[1] == false);
    CHECK(output[2] == true);
    CHECK(output[3] == false);
}

TEST_CASE("Bool tensor Clone is independent") {
    auto original = tf_wrap::Tensor::FromVector<bool>({4}, {true, true, true, true});
    auto clone = original.Clone();
    
    {
        auto view = clone.write<bool>();
        view[0] = false;
        view[1] = false;
    }
    
    // Original unchanged
    auto orig_out = original.ToVector<bool>();
    CHECK(orig_out[0] == true);
    CHECK(orig_out[1] == true);
    
    // Clone modified
    auto clone_out = clone.ToVector<bool>();
    CHECK(clone_out[0] == false);
    CHECK(clone_out[1] == false);
}

// ============================================================================
// Large Bool Tensor
// ============================================================================

TEST_CASE("Large bool tensor round-trip") {
    const size_t N = 10000;
    std::vector<bool> input(N);
    for (size_t i = 0; i < N; ++i) {
        input[i] = (i % 2 == 0);
    }
    
    auto tensor = tf_wrap::Tensor::FromVector<bool>({static_cast<int64_t>(N)}, input);
    
    CHECK(tensor.byte_size() == N);
    CHECK(tensor.num_elements() == N);
    
    auto output = tensor.ToVector<bool>();
    REQUIRE(output.size() == N);
    
    for (size_t i = 0; i < N; ++i) {
        CHECK(output[i] == (i % 2 == 0));
    }
}

// ============================================================================
// Bool Tensor Through Graph
// ============================================================================

TEST_CASE("Bool tensor as graph constant") {
    tf_wrap::Graph g;
    
    auto tensor = tf_wrap::Tensor::FromVector<bool>({4}, {true, false, true, false});
    
    auto* op = g.NewOperation("Const", "bool_const")
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", TF_BOOL)
        .Finish();
    
    CHECK(TF_OperationOutputType({op, 0}) == TF_BOOL);
}

TEST_CASE("Bool tensor through Identity") {
    tf_wrap::Graph g;
    
    auto tensor = tf_wrap::Tensor::FromVector<bool>({4}, {true, false, true, false});
    
    auto* const_op = g.NewOperation("Const", "input")
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", TF_BOOL)
        .Finish();
    
    auto* id_op = g.NewOperation("Identity", "output")
        .AddInput({const_op, 0})
        .SetAttrType("T", TF_BOOL)
        .Finish();
    
    CHECK(TF_OperationOutputType({const_op, 0}) == TF_BOOL);
    CHECK(TF_OperationOutputType({id_op, 0}) == TF_BOOL);
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"output", 0}}, {});
    
    CHECK(results[0].dtype() == TF_BOOL);
    auto output = results[0].ToVector<bool>();
    CHECK(output[0] == true);
    CHECK(output[1] == false);
    CHECK(output[2] == true);
    CHECK(output[3] == false);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_CASE("Empty bool tensor") {
    auto tensor = tf_wrap::Tensor::FromVector<bool>({0}, {});
    
    CHECK(tensor.dtype() == TF_BOOL);
    CHECK(tensor.num_elements() == 0);
    CHECK(tensor.byte_size() == 0);
}

TEST_CASE("Single element bool tensor") {
    auto t = tf_wrap::Tensor::FromVector<bool>({1}, {true});
    
    CHECK(t.num_elements() == 1);
    CHECK(t.byte_size() == 1);
    CHECK(t.ToVector<bool>()[0] == true);
}
