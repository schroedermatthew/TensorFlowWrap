// tests/test_runner.cpp
//
// Runner ergonomics tests.
//
// NOTE: CI compiles/runs test_improvements.cpp for Runner coverage.
// This file exists for additional development-time coverage (no op-wrappers).

#include "doctest.h"

#include "tf_wrap/core.hpp"

#include <memory>

namespace {

struct Graph2In1Out {
    tf_wrap::Graph g;
    TF_Operation* x{nullptr};
    TF_Operation* y{nullptr};
    TF_Operation* sum{nullptr};
};

Graph2In1Out MakeGraph2In1Out() {
    Graph2In1Out t;

    t.x = t.g.NewOperation("Placeholder", "x")
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();

    t.y = t.g.NewOperation("Placeholder", "y")
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();

    t.sum = t.g.NewOperation("AddV2", "sum")
        .AddInput(t.x, 0)
        .AddInput(t.y, 0)
        .SetAttrType("T", TF_FLOAT)
        .Finish();

    return t;
}

} // namespace

TEST_CASE("RunnerBuilder compiles once; Runner runs with minimal call-site friction") {
    auto tg = MakeGraph2In1Out();
    auto sess = std::make_shared<tf_wrap::Session>(tg.g);

    auto run = tf_wrap::RunnerBuilder(sess)
        .feed("x:0")
        .feed("y:0")
        .fetch("sum:0")
        .compile();

    auto x = tf_wrap::Tensor::FromScalar<float>(2.0f);
    auto y = tf_wrap::Tensor::FromScalar<float>(3.0f);

    // simplest "wow" path
    auto out = run(x, y);
    CHECK(out.ToScalar<float>() == doctest::Approx(5.0f));

    // explicit single-output helper
    auto out2 = run.run_one(x, y);
    CHECK(out2.ToScalar<float>() == doctest::Approx(5.0f));
}

TEST_CASE("Runner::run_one throws when fetch count != 1") {
    auto tg = MakeGraph2In1Out();
    auto sess = std::make_shared<tf_wrap::Session>(tg.g);

    auto run = tf_wrap::RunnerBuilder(sess)
        .feed("x:0")
        .feed("y:0")
        .fetch("sum:0")
        .fetch("sum:0")
        .compile();

    auto x = tf_wrap::Tensor::FromScalar<float>(1.0f);
    auto y = tf_wrap::Tensor::FromScalar<float>(1.0f);

    CHECK_THROWS_AS(run.run_one(x, y), tf_wrap::Error);
}
