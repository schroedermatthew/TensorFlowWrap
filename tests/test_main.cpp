// tests/test_main.cpp
// Unit tests for the TensorFlow C++20 wrapper using doctest.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "tf_wrap/all.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <string>
#include <vector>

// ============================================================================
// Status tests
// ============================================================================

TEST_CASE("Status RAII prevents leaks") {
    for (int i = 0; i < 100; ++i) {
        tf_wrap::Status st;
        CHECK(st.ok());
        CHECK(st.code() == TF_OK);
    }
}

TEST_CASE("Status reset() works") {
    tf_wrap::Status st;
    st.set(TF_INVALID_ARGUMENT, "test error");
    CHECK_FALSE(st.ok());

    st.reset();
    CHECK(st.ok());
}

TEST_CASE("Status code names are correct") {
    CHECK(std::string(tf_wrap::Status::code_to_string(TF_OK)) == "OK");
    CHECK(std::string(tf_wrap::Status::code_to_string(TF_INVALID_ARGUMENT)) == "INVALID_ARGUMENT");
    CHECK(std::string(tf_wrap::Status::code_to_string(TF_NOT_FOUND)) == "NOT_FOUND");
}

TEST_CASE("Status operator!() works") {
    tf_wrap::Status st;
    CHECK_FALSE(!st);  // OK status

    st.set(TF_INVALID_ARGUMENT, "test error");
    CHECK(!st);  // Error status

    if (!st) {
        CHECK(st.code() == TF_INVALID_ARGUMENT);
    }
}

TEST_CASE("Status set() accepts string_view") {
    tf_wrap::Status st;

    st.set(TF_CANCELLED);
    CHECK(st.code() == TF_CANCELLED);

    std::string_view msg = "test message";
    st.set(TF_INTERNAL, msg);
    CHECK(st.code() == TF_INTERNAL);
    CHECK(std::string(st.message()) == "test message");

    st.set(TF_UNAVAILABLE, std::string("temporary").substr(0, 4));
    CHECK(st.code() == TF_UNAVAILABLE);
}

// ============================================================================
// Tensor tests
// ============================================================================

TEST_CASE("Tensor FromVector works") {
    std::vector<std::int64_t> shape = {2, 3};
    std::vector<float> data = {1, 2, 3, 4, 5, 6};

    auto tensor = tf_wrap::Tensor::FromVector<float>(shape, data);

    CHECK(tensor.dtype() == TF_FLOAT);
    CHECK(tensor.shape() == shape);
    CHECK(tensor.num_elements() == 6);
    CHECK(tensor.rank() == 2);
}

TEST_CASE("Tensor FromVector<bool> works") {
    std::vector<std::int64_t> shape = {4};
    std::vector<bool> data = {true, false, true, false};

    auto tensor = tf_wrap::Tensor::FromVector<bool>(shape, data);

    CHECK(tensor.dtype() == TF_BOOL);
    CHECK(tensor.shape() == shape);
    CHECK(tensor.num_elements() == 4);

    auto view = tensor.read<bool>();
    CHECK(view[0] == true);
    CHECK(view[1] == false);
    CHECK(view[2] == true);
    CHECK(view[3] == false);
}

TEST_CASE("Tensor FromScalar works") {
    auto tensor = tf_wrap::Tensor::FromScalar<int32_t>(42);

    CHECK(tensor.dtype() == TF_INT32);
    CHECK(tensor.rank() == 0);
    CHECK(tensor.num_elements() == 1);
    CHECK(tensor.ToScalar<int32_t>() == 42);
}

TEST_CASE("Tensor FromRaw rejects null") {
    CHECK_THROWS_AS(tf_wrap::Tensor::FromRaw(nullptr), std::invalid_argument);
}

TEST_CASE("Tensor Zeros factory works") {
    auto tensor = tf_wrap::Tensor::Zeros<float>({2, 3});

    CHECK(tensor.dtype() == TF_FLOAT);
    CHECK(tensor.num_elements() == 6);

    auto view = tensor.read<float>();
    for (auto v : view) {
        CHECK(v == 0.0f);
    }
}

TEST_CASE("Tensor Allocate factory works") {
    auto tensor = tf_wrap::Tensor::Allocate<double>({3, 3});
    CHECK(tensor.dtype() == TF_DOUBLE);
    CHECK(tensor.num_elements() == 9);
    CHECK(tensor.byte_size() == 9 * sizeof(double));
}

TEST_CASE("Tensor read/write views work") {
    auto tensor = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});

    {
        auto view = tensor.read<float>();
        CHECK(view.size() == 3);
        CHECK(view[0] == 1.0f);
        CHECK(view[1] == 2.0f);
        CHECK(view[2] == 3.0f);
    }

    {
        auto view = tensor.write<float>();
        view[0] = 10.0f;
        view[1] = 20.0f;
        view[2] = 30.0f;
    }

    {
        auto view = tensor.read<float>();
        CHECK(view[0] == 10.0f);
        CHECK(view[1] == 20.0f);
        CHECK(view[2] == 30.0f);
    }
}

TEST_CASE("Tensor with_read/with_write work") {
    auto tensor = tf_wrap::Tensor::FromVector<int32_t>({4}, {1, 2, 3, 4});

    int sum = tensor.with_read<int32_t>([](std::span<const int32_t> data) {
        return std::accumulate(data.begin(), data.end(), 0);
    });
    CHECK(sum == 10);

    tensor.with_write<int32_t>([](std::span<int32_t> data) {
        for (auto& v : data) v *= 2;
    });

    sum = tensor.with_read<int32_t>([](std::span<const int32_t> data) {
        return std::accumulate(data.begin(), data.end(), 0);
    });
    CHECK(sum == 20);
}

TEST_CASE("Tensor dtype mismatch throws") {
    auto tensor = tf_wrap::Tensor::FromScalar<float>(1.0f);
    CHECK_THROWS_AS(tensor.read<int32_t>(), std::runtime_error);
    CHECK_THROWS_AS(tensor.write<int32_t>(), std::runtime_error);
}

TEST_CASE("Tensor ToVector extracts data correctly") {
    std::vector<double> original = {1.5, 2.5, 3.5, 4.5};
    auto tensor = tf_wrap::Tensor::FromVector<double>({4}, original);
    
    auto extracted = tensor.ToVector<double>();
    CHECK(extracted == original);
}

TEST_CASE("Tensor ToScalar extracts single value") {
    auto tensor = tf_wrap::Tensor::FromScalar<int64_t>(12345);
    CHECK(tensor.ToScalar<int64_t>() == 12345);
}

TEST_CASE("Tensor ToScalar throws for multi-element tensor") {
    auto tensor = tf_wrap::Tensor::FromVector<float>({2}, {1.0f, 2.0f});
    CHECK_THROWS_AS(tensor.ToScalar<float>(), std::runtime_error);
}

TEST_CASE("Tensor ToVector dtype mismatch throws") {
    auto tensor = tf_wrap::Tensor::FromScalar<float>(1.0f);
    CHECK_THROWS_AS(tensor.ToVector<int32_t>(), std::runtime_error);
}

TEST_CASE("Tensor Clone creates independent copy") {
    auto original = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    auto cloned = original.Clone();
    
    CHECK(cloned.dtype() == original.dtype());
    CHECK(cloned.shape() == original.shape());
    CHECK(cloned.ToVector<float>() == original.ToVector<float>());
    
    // Verify independence
    cloned.write<float>()[0] = 999.0f;
    CHECK(original.read<float>()[0] == 1.0f);
    CHECK(cloned.read<float>()[0] == 999.0f);
}

TEST_CASE("Tensor Clone of empty returns empty") {
    tf_wrap::Tensor empty;
    auto cloned = empty.Clone();
    CHECK_FALSE(cloned.valid());
    CHECK(cloned.handle() == nullptr);
}

TEST_CASE("Tensor Clone of scalar works") {
    auto original = tf_wrap::Tensor::FromScalar<int32_t>(42);
    auto cloned = original.Clone();
    
    CHECK(cloned.ToScalar<int32_t>() == 42);
    CHECK(cloned.rank() == 0);
}

TEST_CASE("Tensor dimension mismatch throws") {
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    CHECK_THROWS_AS(
        tf_wrap::Tensor::FromVector<float>({2, 2}, data),
        std::invalid_argument
    );
}

TEST_CASE("All scalar types map to TF dtypes") {
    CHECK(tf_wrap::tf_dtype_v<float> == TF_FLOAT);
    CHECK(tf_wrap::tf_dtype_v<double> == TF_DOUBLE);
    CHECK(tf_wrap::tf_dtype_v<std::int8_t> == TF_INT8);
    CHECK(tf_wrap::tf_dtype_v<std::int16_t> == TF_INT16);
    CHECK(tf_wrap::tf_dtype_v<std::int32_t> == TF_INT32);
    CHECK(tf_wrap::tf_dtype_v<std::int64_t> == TF_INT64);
    CHECK(tf_wrap::tf_dtype_v<std::uint8_t> == TF_UINT8);
    CHECK(tf_wrap::tf_dtype_v<std::uint16_t> == TF_UINT16);
    CHECK(tf_wrap::tf_dtype_v<std::uint32_t> == TF_UINT32);
    CHECK(tf_wrap::tf_dtype_v<std::uint64_t> == TF_UINT64);
    CHECK(tf_wrap::tf_dtype_v<bool> == TF_BOOL);
}

// ============================================================================
// SessionOptions tests
// ============================================================================

TEST_CASE("SessionOptions RAII works") {
    tf_wrap::SessionOptions opts;
    CHECK(opts.get() != nullptr);

    tf_wrap::SessionOptions with_data;
    CHECK(with_data.get() != nullptr);

    tf_wrap::SessionOptions moved = std::move(with_data);
    CHECK(moved.get() != nullptr);
    CHECK(with_data.get() == nullptr);  // NOLINT: testing moved-from state
}

// ============================================================================
// Graph tests
// ============================================================================

TEST_CASE("Graph creation and operation lookup") {
    tf_wrap::Graph graph;
    CHECK(graph.handle() != nullptr);

    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)graph.NewOperation("Const", "my_const")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();

    auto op = graph.GetOperation("my_const");
    CHECK(op.has_value());

    auto missing = graph.GetOperation("nonexistent");
    CHECK_FALSE(missing.has_value());
}

TEST_CASE("Graph GetOperationOrThrow throws when not found") {
    tf_wrap::Graph graph;
    CHECK_THROWS_AS(graph.GetOperationOrThrow("nonexistent"), std::runtime_error);
}

TEST_CASE("tf_wrap::Output helper creates correct TF_Output") {
    tf_wrap::Graph graph;
    
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    TF_Operation* op = graph.NewOperation("Const", "test_op")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    TF_Output output = tf_wrap::Output(op, 0);
    CHECK(output.oper == op);
    CHECK(output.index == 0);
    
    TF_Output output2 = tf_wrap::Output(op, 2);
    CHECK(output2.index == 2);
}

TEST_CASE("Buffer RAII works correctly") {
    tf_wrap::Buffer buf;
    CHECK(buf.get() != nullptr);
    
    tf_wrap::Buffer moved = std::move(buf);
    CHECK(moved.get() != nullptr);
    CHECK(buf.get() == nullptr);  // NOLINT: testing moved-from state
}

TEST_CASE("Graph ToGraphDef serializes graph") {
    tf_wrap::Graph graph;
    
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)graph.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto data = graph.ToGraphDef();
    CHECK(!data.empty());
    CHECK(data.size() > 0);
}

TEST_CASE("Graph GetPlaceholders finds placeholder ops") {
    tf_wrap::Graph graph;
    
    (void)graph.NewOperation("Placeholder", "input")
        .SetAttrType("dtype", TF_FLOAT)
        .SetAttrShape("shape", {-1, 10})
        .Finish();
    
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)graph.NewOperation("Const", "weight")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto placeholders = graph.GetPlaceholders();
    CHECK(placeholders.size() == 1);
    CHECK(placeholders[0].op_name == "input");
}

TEST_CASE("Graph GetOperationsByType filters correctly") {
    tf_wrap::Graph graph;
    
    auto t1 = tf_wrap::Tensor::FromScalar<float>(1.0f);
    auto t2 = tf_wrap::Tensor::FromScalar<float>(2.0f);
    
    (void)graph.NewOperation("Const", "c1")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    (void)graph.NewOperation("Const", "c2")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    (void)graph.NewOperation("Placeholder", "p1")
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto consts = graph.GetOperationsByType("Const");
    CHECK(consts.size() == 2);
    
    auto placeholders = graph.GetOperationsByType("Placeholder");
    CHECK(placeholders.size() == 1);
    
    auto none = graph.GetOperationsByType("NonExistent");
    CHECK(none.empty());
}

TEST_CASE("Graph DebugString produces readable output") {
    tf_wrap::Graph graph;
    
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)graph.NewOperation("Const", "MyConst")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    std::string debug = graph.DebugString();
    CHECK(debug.find("1 operations") != std::string::npos);
    CHECK(debug.find("MyConst") != std::string::npos);
    CHECK(debug.find("Const") != std::string::npos);
}
