// tests/test_real_tf.cpp
//
// Minimal integration tests against the real TensorFlow C library.
//
// Goal: validate that the *production* wrapper API works end-to-end with a
// real TF runtime (no stub):
//  - Build a tiny graph
//  - Resolve outputs once
//  - Run inference via handle-based Session::Run
//  - Exercise BatchRunStacked and RunContext reuse

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

// Suppress warnings from CHECK_THROWS_AS with [[nodiscard]] functions
#if defined(_MSC_VER)
#pragma warning(disable: 4834)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

#include "tf_wrap/core.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace {

TF_Operation* MakeConstVec(tf_wrap::Graph& g, const char* name,
                           std::vector<float> values) {
    const std::int64_t n = static_cast<std::int64_t>(values.size());
    auto t = tf_wrap::Tensor::FromVector<float>({n}, values);
    return g.NewOperation("Const", name)
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
}

TF_Operation* MakePlaceholder(tf_wrap::Graph& g, const char* name) {
    return g.NewOperation("Placeholder", name)
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
}

} // namespace

TEST_CASE("const_and_identity") {
    tf_wrap::Graph g;

    auto* x = MakeConstVec(g, "X", {1.0f, 2.0f, 3.0f});

    (void)g.NewOperation("Identity", "Y")
        .AddInput(x, 0)
        .SetAttrType("T", TF_FLOAT)
        .Finish();

    tf_wrap::Session s(g);

    auto results = s.Run({}, std::vector<tf_wrap::Fetch>{tf_wrap::Fetch{s.resolve("Y:0")}}, {});
    REQUIRE(results.size() == 1);

    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 3);
    CHECK(v[0] == doctest::Approx(1.0f));
    CHECK(v[1] == doctest::Approx(2.0f));
    CHECK(v[2] == doctest::Approx(3.0f));
}

TEST_CASE("batch_run_stacked_true_batching") {
    tf_wrap::Graph g;

    auto* in = MakePlaceholder(g, "input");

    (void)g.NewOperation("Identity", "output")
        .AddInput(in, 0)
        .SetAttrType("T", TF_FLOAT)
        .Finish();

    tf_wrap::Session s(g);

    std::vector<tf_wrap::Tensor> inputs;
    inputs.push_back(tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f}));
    inputs.push_back(tf_wrap::Tensor::FromVector<float>({3}, {4.0f, 5.0f, 6.0f}));
    inputs.push_back(tf_wrap::Tensor::FromVector<float>({3}, {7.0f, 8.0f, 9.0f}));

    auto out = s.BatchRunStacked(
        s.resolve("input:0"),
        std::span<const tf_wrap::Tensor>(inputs),
        s.resolve("output:0"));

    REQUIRE(out.size() == inputs.size());

    for (std::size_t i = 0; i < out.size(); ++i) {
        CHECK(out[i].ToVector<float>() == inputs[i].ToVector<float>());
    }
}

TEST_CASE("run_context_reuse") {
    tf_wrap::Graph g;

    auto* in = MakePlaceholder(g, "in");
    (void)g.NewOperation("Identity", "out")
        .AddInput(in, 0)
        .SetAttrType("T", TF_FLOAT)
        .Finish();

    tf_wrap::Session s(g);

    auto in0 = s.resolve("in:0");
    auto out0 = s.resolve("out:0");

    tf_wrap::RunContext ctx;
    ctx.add_fetch(out0);

    // run #1
    ctx.reset();
    ctx.add_feed(in0, tf_wrap::Tensor::FromScalar<float>(1.0f));
    ctx.add_fetch(out0);
    auto r1 = s.Run(ctx);
    REQUIRE(r1.size() == 1);
    CHECK(r1[0].ToScalar<float>() == doctest::Approx(1.0f));

    // run #2 (reuse ctx)
    ctx.reset();
    ctx.add_feed(in0, tf_wrap::Tensor::FromScalar<float>(2.0f));
    ctx.add_fetch(out0);
    auto r2 = s.Run(ctx);
    REQUIRE(r2.size() == 1);
    CHECK(r2[0].ToScalar<float>() == doctest::Approx(2.0f));
}
