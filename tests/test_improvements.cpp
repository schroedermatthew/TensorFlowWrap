// tests/test_improvements.cpp
//
// Focus: production inference ergonomics + safety.
//
// CI runs in stub mode by default, so we build tiny graphs manually and validate:
// - Session::resolve name parsing + output index validation
// - Compiled Runner signature (RunnerBuilder -> Runner)
// - Variadic call-site ergonomics ("wow this is easy")
// - Structured bindings via run_tuple<K>
// - Context reuse for stable latency

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

// Suppress warnings from CHECK_THROWS_AS with [[nodiscard]] functions
#if defined(_MSC_VER)
#pragma warning(disable: 4834)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

#include "tf_wrap/core.hpp"

#include <memory>
#include <tuple>
#include <vector>

namespace {

struct TinyGraph {
    tf_wrap::Graph g;
    TF_Operation* x{nullptr};
    TF_Operation* y{nullptr};
    TF_Operation* xid{nullptr};
    TF_Operation* yid{nullptr};
    TF_Operation* sum{nullptr};
};

TinyGraph MakeTinyGraph() {
    TinyGraph t;

    t.x = t.g.NewOperation("Placeholder", "x")
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();

    t.y = t.g.NewOperation("Placeholder", "y")
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();

    t.xid = t.g.NewOperation("Identity", "x_id")
        .AddInput(t.x, 0)
        .SetAttrType("T", TF_FLOAT)
        .Finish();

    t.yid = t.g.NewOperation("Identity", "y_id")
        .AddInput(t.y, 0)
        .SetAttrType("T", TF_FLOAT)
        .Finish();

    // Depends on both feeds (good for testing multi-feed + single fetch)
    t.sum = t.g.NewOperation("AddV2", "sum")
        .AddInput(t.xid, 0)
        .AddInput(t.yid, 0)
        .SetAttrType("T", TF_FLOAT)
        .Finish();

    return t;
}

} // namespace

// ============================================================================
// Session::resolve
// ============================================================================

TEST_CASE("Session::resolve resolves op:index and validates output index") {
    auto tiny = MakeTinyGraph();
    tf_wrap::Session s(tiny.g);

    // explicit index
    {
        TF_Output out = s.resolve("x:0");
        CHECK(out.oper == tiny.x);
        CHECK(out.index == 0);
    }

    // no index => defaults to op name (index 0)
    {
        TF_Output out = s.resolve("y");
        CHECK(out.oper == tiny.y);
        CHECK(out.index == 0);
    }

    // out of range index
    CHECK_THROWS_AS(s.resolve("x:999"), tf_wrap::Error);
}

// ============================================================================
// Runner (compiled signature)
// ============================================================================

TEST_CASE("Runner: compile 1-in/1-out and call like a function") {
    auto tiny = MakeTinyGraph();

    // Session must outlive Runner (Runner holds shared_ptr<const Session>)
    auto sess = std::make_shared<tf_wrap::Session>(tiny.g);

    auto run = tf_wrap::RunnerBuilder(sess)
        .feed(TF_Output{tiny.x, 0})
        .fetch(TF_Output{tiny.xid, 0})
        .compile();

    tf_wrap::Tensor in = tf_wrap::Tensor::FromScalar<float>(123.0f);

    // "wow this is easy"
    tf_wrap::Tensor out = run(in);
    CHECK(out.ToScalar<float>() == doctest::Approx(123.0f));
}

TEST_CASE("Runner: variadic operator() for multiple feeds + single fetch") {
    auto tiny = MakeTinyGraph();
    auto sess = std::make_shared<tf_wrap::Session>(tiny.g);

    auto run = tf_wrap::RunnerBuilder(sess)
        .feed("x:0")
        .feed("y:0")
        .fetch("sum:0")
        .compile();

    tf_wrap::Tensor x = tf_wrap::Tensor::FromScalar<float>(1.5f);
    tf_wrap::Tensor y = tf_wrap::Tensor::FromScalar<float>(2.5f);

    // feeds are positional in the compiled signature
    tf_wrap::Tensor out = run(x, y);
    CHECK(out.ToScalar<float>() == doctest::Approx(4.0f));
}

TEST_CASE("Runner: run_tuple<K> supports structured bindings") {
    auto tiny = MakeTinyGraph();
    auto sess = std::make_shared<tf_wrap::Session>(tiny.g);

    auto run = tf_wrap::RunnerBuilder(sess)
        .feed("x:0")
        .feed("y:0")
        .fetch("x_id:0")
        .fetch("y_id:0")
        .compile();

    tf_wrap::Tensor x = tf_wrap::Tensor::FromScalar<float>(10.0f);
    tf_wrap::Tensor y = tf_wrap::Tensor::FromScalar<float>(20.0f);

    auto [ox, oy] = run.run_tuple<2>(x, y);
    CHECK(ox.ToScalar<float>() == doctest::Approx(10.0f));
    CHECK(oy.ToScalar<float>() == doctest::Approx(20.0f));
}

TEST_CASE("Runner: Context reuse (run(ctx, ...))") {
    auto tiny = MakeTinyGraph();
    auto sess = std::make_shared<tf_wrap::Session>(tiny.g);

    auto run = tf_wrap::RunnerBuilder(sess)
        .feed("x:0")
        .feed("y:0")
        .fetch("sum:0")
        .compile();

    auto ctx = run.make_context();

    tf_wrap::Tensor x = tf_wrap::Tensor::FromScalar<float>(3.0f);
    tf_wrap::Tensor y = tf_wrap::Tensor::FromScalar<float>(4.0f);

    auto outs = run.run(ctx, x, y);
    REQUIRE(outs.size() == 1);
    CHECK(outs[0].ToScalar<float>() == doctest::Approx(7.0f));

    // Run again with same context (no reallocation in steady state)
    tf_wrap::Tensor x2 = tf_wrap::Tensor::FromScalar<float>(5.0f);
    tf_wrap::Tensor y2 = tf_wrap::Tensor::FromScalar<float>(6.0f);
    auto outs2 = run.run(ctx, x2, y2);
    REQUIRE(outs2.size() == 1);
    CHECK(outs2[0].ToScalar<float>() == doctest::Approx(11.0f));
}

TEST_CASE("Runner: BorrowedTensor interop works (explicitly unsafe)") {
    auto tiny = MakeTinyGraph();
    auto sess = std::make_shared<tf_wrap::Session>(tiny.g);

    auto run = tf_wrap::RunnerBuilder(sess)
        .feed("x:0")
        .fetch("x_id:0")
        .compile();

    tf_wrap::Tensor in = tf_wrap::Tensor::FromScalar<float>(42.0f);

    // Explicitly borrowed pointer; keepalive makes it safe for the call.
    tf_wrap::Runner::BorrowedTensor bt{in.handle(), in.keepalive()};

    tf_wrap::Tensor out = run(bt);
    CHECK(out.ToScalar<float>() == doctest::Approx(42.0f));
}

TEST_CASE("Runner: throws on wrong number of inputs") {
    auto tiny = MakeTinyGraph();
    auto sess = std::make_shared<tf_wrap::Session>(tiny.g);

    auto run = tf_wrap::RunnerBuilder(sess)
        .feed("x:0")
        .feed("y:0")
        .fetch("sum:0")
        .compile();

    tf_wrap::Tensor x = tf_wrap::Tensor::FromScalar<float>(1.0f);

    CHECK_THROWS_AS(run(x), tf_wrap::Error);
}

// ============================================================================
// P0-1 Fix: reshape keepalive chain flattening
// ============================================================================

TEST_CASE("Tensor::reshape keepalive chain is flattened (P0-1 fix)") {
    // Create a tensor and reshape it many times.
    // Before the fix, this would create an O(n) keepalive chain.
    // After the fix, all reshaped tensors point directly to the root.
    
    auto root = tf_wrap::Tensor::FromVector<float>({6}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    
    // Chain of reshapes
    auto r1 = root.reshape({2, 3});
    auto r2 = r1.reshape({3, 2});
    auto r3 = r2.reshape({6});
    auto r4 = r3.reshape({1, 6});
    auto r5 = r4.reshape({6, 1});
    
    // All should be valid and contain the same data
    CHECK(r5.num_elements() == 6);
    auto view = r5.read<float>();
    CHECK(view[0] == 1.0f);
    CHECK(view[5] == 6.0f);
    
    // The fix ensures memory doesn't grow with reshape depth.
    // We can't directly test the internal keepalive structure, but we can
    // verify correctness after many reshapes.
    
    // Stress test: 1000 reshapes should not cause memory issues
    tf_wrap::Tensor current = root.reshape({6});
    for (int i = 0; i < 1000; ++i) {
        current = current.reshape({6});
    }
    CHECK(current.num_elements() == 6);
    CHECK(current.read<float>()[0] == 1.0f);
}

TEST_CASE("Tensor::reshape preserves data after root destroyed") {
    // Verify keepalive works: reshape should keep data alive even after
    // the original tensor goes out of scope.
    
    tf_wrap::Tensor reshaped;
    {
        auto root = tf_wrap::Tensor::FromVector<float>({4}, {10.0f, 20.0f, 30.0f, 40.0f});
        reshaped = root.reshape({2, 2});
        // root goes out of scope here
    }
    
    // reshaped should still be valid
    CHECK(reshaped.num_elements() == 4);
    auto view = reshaped.read<float>();
    CHECK(view[0] == 10.0f);
    CHECK(view[3] == 40.0f);
}

// ============================================================================
// P3: resolve() index overflow protection
// ============================================================================

TEST_CASE("Session::resolve handles large index gracefully") {
    auto tiny = MakeTinyGraph();
    tf_wrap::Session sess(tiny.g);
    
    // Normal case works
    auto out = sess.resolve("x:0");
    CHECK(out.oper != nullptr);
    CHECK(out.index == 0);
    
    // Out of range index throws (but doesn't overflow/crash)
    CHECK_THROWS_AS(sess.resolve("x:999"), tf_wrap::Error);
}

// ============================================================================
// P3-9: SmallVector::at() error message includes context
// ============================================================================

TEST_CASE("SmallVector::at() error message includes index and size") {
    tf_wrap::SmallVector<int, 4> vec{1, 2, 3};
    
    // Valid access works
    CHECK(vec.at(0) == 1);
    CHECK(vec.at(2) == 3);
    
    // Invalid access throws with context
    bool caught = false;
    try {
        (void)vec.at(10);
    } catch (const std::out_of_range& e) {
        caught = true;
        std::string msg = e.what();
        // Should contain index and size
        CHECK(msg.find("10") != std::string::npos);
        CHECK(msg.find("3") != std::string::npos);
    }
    CHECK(caught);
}
