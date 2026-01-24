// tests/test_runner.cpp
// Tests for fluent Runner API
//
// Purpose: Verify the easy::Runner provides ergonomic session execution

#include "doctest.h"

#include "tf_wrap/all.hpp"

using namespace tf_wrap;
using namespace tf_wrap::easy;

// ============================================================================
// Basic Runner Usage
// ============================================================================

TEST_CASE("Runner fetches single output by handle") {
    Graph g;
    
    auto t = Tensor::FromScalar<float>(42.0f);
    auto* c = g.NewOperation("Const", "c")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    Session s(g);
    
    auto result = s.runner()
        .fetch({c, 0})
        .run_one();
    
    CHECK(result.ToScalar<float>() == 42.0f);
}

TEST_CASE("Runner fetches by name string with index") {
    Graph g;
    
    auto t = Tensor::FromScalar<float>(42.0f);
    g.NewOperation("Const", "my_const")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    Session s(g);
    
    auto result = s.runner()
        .fetch("my_const:0")
        .run_one();
    
    CHECK(result.ToScalar<float>() == 42.0f);
}

TEST_CASE("Runner fetches by name without index (defaults to 0)") {
    Graph g;
    
    auto t = Tensor::FromScalar<float>(42.0f);
    g.NewOperation("Const", "my_const")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    Session s(g);
    
    auto result = s.runner()
        .fetch("my_const")
        .run_one();
    
    CHECK(result.ToScalar<float>() == 42.0f);
}

// ============================================================================
// Runner with Feeds (Placeholders)
// ============================================================================

TEST_CASE("Runner with placeholder feed by name") {
    Graph g;
    
    auto* ph = g.NewOperation("Placeholder", "x")
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* id = g.NewOperation("Identity", "y")
        .AddInput({ph, 0})
        .SetAttrType("T", TF_FLOAT)
        .Finish();
    
    Session s(g);
    
    auto input = Tensor::FromScalar<float>(123.0f);
    
    auto result = s.runner()
        .feed("x:0", input)
        .fetch("y:0")
        .run_one();
    
    CHECK(result.ToScalar<float>() == 123.0f);
}

TEST_CASE("Runner with placeholder feed by handle") {
    Graph g;
    
    auto* ph = g.NewOperation("Placeholder", "x")
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* id = g.NewOperation("Identity", "y")
        .AddInput({ph, 0})
        .SetAttrType("T", TF_FLOAT)
        .Finish();
    
    Session s(g);
    
    auto input = Tensor::FromScalar<float>(456.0f);
    
    auto result = s.runner()
        .feed({ph, 0}, input)
        .fetch({id, 0})
        .run_one();
    
    CHECK(result.ToScalar<float>() == 456.0f);
}

// ============================================================================
// Multiple Fetches
// ============================================================================

TEST_CASE("Runner with multiple fetches") {
    Graph g;
    
    auto t1 = Tensor::FromScalar<float>(1.0f);
    auto t2 = Tensor::FromScalar<float>(2.0f);
    
    auto* c1 = g.NewOperation("Const", "c1")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* c2 = g.NewOperation("Const", "c2")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    Session s(g);
    
    auto results = s.runner()
        .fetch("c1:0")
        .fetch("c2:0")
        .run();
    
    REQUIRE(results.size() == 2);
    CHECK(results[0].ToScalar<float>() == 1.0f);
    CHECK(results[1].ToScalar<float>() == 2.0f);
}

TEST_CASE("Runner with mixed handle and name fetches") {
    Graph g;
    
    auto t1 = Tensor::FromScalar<float>(10.0f);
    auto t2 = Tensor::FromScalar<float>(20.0f);
    
    auto* c1 = g.NewOperation("Const", "by_handle")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    g.NewOperation("Const", "by_name")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    Session s(g);
    
    auto results = s.runner()
        .fetch({c1, 0})
        .fetch("by_name:0")
        .run();
    
    REQUIRE(results.size() == 2);
    CHECK(results[0].ToScalar<float>() == 10.0f);
    CHECK(results[1].ToScalar<float>() == 20.0f);
}

// ============================================================================
// Fluent Chaining
// ============================================================================

TEST_CASE("Runner fluent chaining (rvalue refs)") {
    Graph g;
    
    auto* ph = g.NewOperation("Placeholder", "input")
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* id = g.NewOperation("Identity", "output")
        .AddInput({ph, 0})
        .SetAttrType("T", TF_FLOAT)
        .Finish();
    
    Session s(g);
    
    auto input = Tensor::FromScalar<float>(999.0f);
    
    // Single expression with rvalue chaining
    auto result = s.runner()
        .feed("input", input)
        .fetch("output")
        .run_one();
    
    CHECK(result.ToScalar<float>() == 999.0f);
}

// ============================================================================
// OpResult Integration
// ============================================================================

TEST_CASE("Runner fetches by OpResult") {
    Graph g;
    
    auto t = Tensor::FromScalar<float>(77.0f);
    auto result_op = ops::OpResult(
        g.NewOperation("Const", "c")
            .SetAttrTensor("value", t.handle())
            .SetAttrType("dtype", TF_FLOAT)
            .Finish()
    );
    
    Session s(g);
    
    auto result = s.runner()
        .fetch(result_op)
        .run_one();
    
    CHECK(result.ToScalar<float>() == 77.0f);
}

// ============================================================================
// Error Cases
// ============================================================================

TEST_CASE("Runner throws on unknown operation") {
    Graph g;
    Session s(g);
    
    CHECK_THROWS_WITH(
        s.runner().fetch("nonexistent:0").run(),
        doctest::Contains("not found"));
}

TEST_CASE("Runner throws on invalid output index") {
    Graph g;
    
    auto t = Tensor::FromScalar<float>(1.0f);
    g.NewOperation("Const", "c")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    Session s(g);
    
    // Const has 1 output (index 0), index 99 is invalid
    CHECK_THROWS_WITH(
        s.runner().fetch("c:99").run(),
        doctest::Contains("out of range"));
}

TEST_CASE("Runner run_one throws on zero fetches") {
    Graph g;
    
    auto t = Tensor::FromScalar<float>(1.0f);
    g.NewOperation("Const", "c")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    Session s(g);
    
    CHECK_THROWS_WITH(
        s.runner().run_one(),
        doctest::Contains("expected 1 fetch"));
}

TEST_CASE("Runner run_one throws on multiple fetches") {
    Graph g;
    
    auto t = Tensor::FromScalar<float>(1.0f);
    g.NewOperation("Const", "c1")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    g.NewOperation("Const", "c2")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    Session s(g);
    
    CHECK_THROWS_WITH(
        s.runner().fetch("c1").fetch("c2").run_one(),
        doctest::Contains("expected 1 fetch"));
}

// ============================================================================
// TensorName Edge Cases
// ============================================================================

TEST_CASE("TensorName parsing edge cases") {
    SUBCASE("Standard name:index") {
        auto tn = TensorName::parse("foo:0");
        CHECK(tn.op == "foo");
        CHECK(tn.index == 0);
        CHECK(tn.had_explicit_index == true);
    }
    
    SUBCASE("Large index") {
        auto tn = TensorName::parse("op:99");
        CHECK(tn.op == "op");
        CHECK(tn.index == 99);
    }
    
    SUBCASE("Name without index") {
        auto tn = TensorName::parse("just_name");
        CHECK(tn.op == "just_name");
        CHECK(tn.index == 0);
        CHECK(tn.had_explicit_index == false);
    }
    
    SUBCASE("Name with non-digit suffix") {
        auto tn = TensorName::parse("foo:bar");
        CHECK(tn.op == "foo:bar");
        CHECK(tn.index == 0);
    }
    
    SUBCASE("Trailing colon") {
        auto tn = TensorName::parse("foo:");
        CHECK(tn.op == "foo:");
        CHECK(tn.index == 0);
    }
    
    SUBCASE("Multiple colons") {
        auto tn = TensorName::parse("a:b:3");
        CHECK(tn.op == "a:b");
        CHECK(tn.index == 3);
    }
    
    SUBCASE("Whitespace trimming") {
        auto tn = TensorName::parse("  foo:0  ");
        CHECK(tn.op == "foo");
        CHECK(tn.index == 0);
    }
    
    SUBCASE("Empty string throws") {
        CHECK_THROWS(TensorName::parse(""));
    }
    
    SUBCASE("Whitespace only throws") {
        CHECK_THROWS(TensorName::parse("   "));
    }
    
    SUBCASE("Empty op name throws") {
        CHECK_THROWS(TensorName::parse(":0"));
    }
}

TEST_CASE("TensorName looks_like_tensor_name") {
    CHECK(TensorName::looks_like_tensor_name("foo:0") == true);
    CHECK(TensorName::looks_like_tensor_name("bar:123") == true);
    CHECK(TensorName::looks_like_tensor_name("foo") == false);
    CHECK(TensorName::looks_like_tensor_name("foo:") == false);
    CHECK(TensorName::looks_like_tensor_name("foo:bar") == false);
    CHECK(TensorName::looks_like_tensor_name("") == false);
}

TEST_CASE("TensorName to_string round-trip") {
    auto tn = TensorName::parse("my_op:5");
    CHECK(tn.to_string() == "my_op:5");
    
    auto tn2 = TensorName::parse("op_without_index");
    CHECK(tn2.to_string() == "op_without_index:0");
}
