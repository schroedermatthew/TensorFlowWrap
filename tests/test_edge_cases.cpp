// tests/test_edge_cases.cpp
//
// Edge-case tests for the production inference wrapper.
//
// This suite intentionally avoids string-based hot-path APIs. We resolve names
// once (Session::resolve) and then run using TF_Output handles.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

// Suppress warnings from CHECK_THROWS_AS with [[nodiscard]] functions
#if defined(_MSC_VER)
#pragma warning(disable: 4834)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

#include "tf_wrap/core.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<tf_wrap::Fetch> F(tf_wrap::Session& s, std::initializer_list<const char*> names) {
    std::vector<tf_wrap::Fetch> fetches;
    fetches.reserve(names.size());
    for (const char* n : names) {
        fetches.emplace_back(s.resolve(n));
    }
    return fetches;
}

} // namespace

TEST_CASE("Graph mutation after Session creation must throw") {
    tf_wrap::Graph g;

    auto a = tf_wrap::Tensor::FromVector<float>({2}, {1.0f, 2.0f});
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", a.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();

    CHECK_FALSE(g.is_frozen());

    tf_wrap::Session s(g);
    CHECK(g.is_frozen());

    // Any mutation attempt must throw (graph is immutable once session exists)
    auto b = tf_wrap::Tensor::FromScalar<float>(0.0f);
    CHECK_THROWS(g.NewOperation("Const", "B")
        .SetAttrTensor("value", b.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish());

    // Session should still work (fetch A)
    auto outs = s.Run({}, F(s, {"A:0"}), {});
    REQUIRE(outs.size() == 1);
    CHECK((outs[0].ToVector<float>() == std::vector<float>{1.0f, 2.0f}));
}

TEST_CASE("Session move leaves source invalid") {
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();

    tf_wrap::Session s1(g);
    tf_wrap::Session s2(std::move(s1));

    CHECK(s2.handle() != nullptr);
    CHECK(s1.handle() == nullptr);
}

TEST_CASE("Multiple Sessions from same Graph work") {
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(42.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();

    tf_wrap::Session s1(g);
    tf_wrap::Session s2(g);
    tf_wrap::Session s3(g);

    auto r1 = s1.Run({}, F(s1, {"A:0"}), {});
    auto r2 = s2.Run({}, F(s2, {"A:0"}), {});
    auto r3 = s3.Run({}, F(s3, {"A:0"}), {});

    REQUIRE(r1.size() == 1);
    REQUIRE(r2.size() == 1);
    REQUIRE(r3.size() == 1);

    CHECK(r1[0].ToScalar<float>() == 42.0f);
    CHECK(r2[0].ToScalar<float>() == 42.0f);
    CHECK(r3[0].ToScalar<float>() == 42.0f);
}

TEST_CASE("Session::resolve throws on non-existent op and on out-of-range index") {
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();

    tf_wrap::Session s(g);

    CHECK_THROWS_AS(s.resolve("DoesNotExist:0"), tf_wrap::Error);
    CHECK_THROWS_AS(s.resolve("A:999"), tf_wrap::Error);

    // If the suffix after ':' isn't numeric, it's treated as part of the op name
    // and should fail with NOT_FOUND in this case.
    CHECK_THROWS_AS(s.resolve("A:foo"), tf_wrap::Error);
}

TEST_CASE("validate_output rejects outputs from another graph") {
    tf_wrap::Graph g1;
    (void)g1.NewOperation("Const", "A")
        .SetAttrTensor("value", tf_wrap::Tensor::FromScalar<float>(1.0f).handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();

    tf_wrap::Graph g2;
    (void)g2.NewOperation("Const", "B")
        .SetAttrTensor("value", tf_wrap::Tensor::FromScalar<float>(2.0f).handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();

    tf_wrap::Session s1(g1);
    tf_wrap::Session s2(g2);

    auto out_b = s2.resolve("B:0");

    CHECK_THROWS_AS(s1.validate_output(out_b), tf_wrap::Error);
}

TEST_CASE("Session Run with empty fetches") {
    tf_wrap::Graph g;
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", tf_wrap::Tensor::FromScalar<float>(1.0f).handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();

    tf_wrap::Session s(g);

    // Real TensorFlow typically requires at least one fetch or target and throws INVALID_ARGUMENT.
    // Stub mode returns empty results. Both behaviors are acceptable.
    std::vector<tf_wrap::Fetch> empty_fetches;

    try {
        auto results = s.Run({}, empty_fetches, {});
        CHECK(results.empty());
    } catch (const std::exception& e) {
        std::string msg = e.what();
        CHECK(msg.find("INVALID_ARGUMENT") != std::string::npos);
    }
}

TEST_CASE("Session::Run fetches multiple outputs in single call") {
    tf_wrap::Graph g;

    // Create two independent constants
    auto t1 = tf_wrap::Tensor::FromScalar<float>(10.0f);
    (void)g.NewOperation("Const", "const1")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();

    auto t2 = tf_wrap::Tensor::FromScalar<float>(20.0f);
    (void)g.NewOperation("Const", "const2")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();

    tf_wrap::Session s(g);

    // Fetch both outputs in single Run call
    auto results = s.Run({}, F(s, {"const1:0", "const2:0"}), {});

    REQUIRE(results.size() == 2);
    CHECK(results[0].ToScalar<float>() == doctest::Approx(10.0f));
    CHECK(results[1].ToScalar<float>() == doctest::Approx(20.0f));
}

TEST_CASE("ReadView keeps data alive after tensor moved") {
    auto tensor = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    auto view = tensor.read<float>();

    // Move tensor away
    tf_wrap::Tensor moved = std::move(tensor);
    CHECK(!tensor.valid());  // NOLINT: testing moved-from state

    // View should still be valid due to keepalive
    CHECK(view.size() == 3);
    CHECK(view[0] == 1.0f);
    CHECK(view[1] == 2.0f);
    CHECK(view[2] == 3.0f);
}

TEST_CASE("Empty tensor operations are safe") {
    tf_wrap::Tensor empty;

    CHECK_FALSE(empty.valid());
    CHECK(empty.handle() == nullptr);

    // Clone of empty returns empty
    auto cloned = empty.Clone();
    CHECK_FALSE(cloned.valid());

    // Move of empty is safe
    tf_wrap::Tensor moved = std::move(empty);
    CHECK_FALSE(moved.valid());
}
