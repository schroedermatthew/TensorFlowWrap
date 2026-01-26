// tests/test_ops.cpp
//
// Production inference build: no generated op-wrappers.
//
// These tests validate that we can still build graphs via the raw operation
// builder and execute them via Session::Run (works for both stub and real TF).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

// Suppress warnings from CHECK_THROWS_AS with [[nodiscard]] functions
#if defined(_MSC_VER)
#pragma warning(disable: 4834)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

#include "tf_wrap/core.hpp"

#include <vector>

namespace {

TF_Operation* MakePlaceholder(tf_wrap::Graph& g, const char* name, TF_DataType dtype) {
    return g.NewOperation("Placeholder", name)
        .SetAttrType("dtype", dtype)
        .Finish();
}

TF_Operation* MakeConstScalar(tf_wrap::Graph& g, const char* name, float value) {
    auto t = tf_wrap::Tensor::FromScalar<float>(value);
    return g.NewOperation("Const", name)
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
}

} // namespace

TEST_CASE("Graph + Session: AddV2 executes") {
    tf_wrap::Graph g;

    auto* x = MakePlaceholder(g, "X", TF_FLOAT);
    auto* y = MakePlaceholder(g, "Y", TF_FLOAT);

    auto* add = g.NewOperation("AddV2", "AddXY")
        .AddInput(x, 0)
        .AddInput(y, 0)
        .SetAttrType("T", TF_FLOAT)
        .Finish();

    tf_wrap::Session s(g);

    auto tx = tf_wrap::Tensor::FromScalar<float>(1.5f);
    auto ty = tf_wrap::Tensor::FromScalar<float>(2.5f);

    std::vector<tf_wrap::Feed> feeds{
        tf_wrap::Feed{TF_Output{x, 0}, tx},
        tf_wrap::Feed{TF_Output{y, 0}, ty},
    };

    std::vector<tf_wrap::Fetch> fetches{
        tf_wrap::Fetch{TF_Output{add, 0}},
    };

    auto outs = s.Run(feeds, fetches);
    REQUIRE(outs.size() == 1);
    CHECK(outs[0].ToScalar<float>() == doctest::Approx(4.0f));
}

TEST_CASE("Session::resolve resolves by name and validates output index") {
    tf_wrap::Graph g;
    auto* c = MakeConstScalar(g, "C", 7.0f);
    tf_wrap::Session s(g);

    const TF_Output out0 = s.resolve("C:0");
    CHECK(out0.oper == c);
    CHECK(out0.index == 0);

    CHECK_THROWS_AS(s.resolve("C:999"), tf_wrap::Error);
}
