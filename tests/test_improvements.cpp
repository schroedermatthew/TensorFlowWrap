// tests/test_improvements.cpp
// Comprehensive tests for TensorFlowWrap improvements
//
// Tests:
// - P0: Stub dtype tracking
// - P0: Bool tensor safety
// - P0: Output index bounds checking
// - P1: TensorName parsing
// - P1: Dtype inference
// - P1: Runner API
// - P1: Easy helpers

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "tf_wrap/all.hpp"
#include <cstdint>
#include <cstring>
#include <string>
#include <stdexcept>

using namespace tf_wrap;
using namespace tf_wrap::ops;

// ============================================================================
// P0: Stub Dtype Tracking Tests
// ============================================================================

TEST_SUITE("P0: Stub Dtype Tracking") {

TEST_CASE("Const operation tracks dtype from tensor attribute") {
    Graph graph;
    
    // Create a float tensor
    auto t_float = Tensor::FromScalar<float>(1.5f);
    auto const_float = Const(graph, "const_float", t_float.handle(), TF_FLOAT);
    
    // Verify dtype is tracked
    TF_Output out = const_float.output(0);
    CHECK(TF_OperationOutputType(out) == TF_FLOAT);
    
    // Create an int32 tensor
    auto t_int = Tensor::FromScalar<std::int32_t>(42);
    auto const_int = Const(graph, "const_int", t_int.handle(), TF_INT32);
    
    out = const_int.output(0);
    CHECK(TF_OperationOutputType(out) == TF_INT32);
}

TEST_CASE("Placeholder tracks dtype") {
    Graph graph;
    
    SUBCASE("float placeholder") {
        auto ph = Placeholder(graph, "ph_float", TF_FLOAT);
        CHECK(TF_OperationOutputType(ph.output(0)) == TF_FLOAT);
    }
    
    SUBCASE("double placeholder") {
        auto ph = Placeholder(graph, "ph_double", TF_DOUBLE);
        CHECK(TF_OperationOutputType(ph.output(0)) == TF_DOUBLE);
    }
    
    SUBCASE("int32 placeholder") {
        auto ph = Placeholder(graph, "ph_int32", TF_INT32);
        CHECK(TF_OperationOutputType(ph.output(0)) == TF_INT32);
    }
    
    SUBCASE("int64 placeholder") {
        auto ph = Placeholder(graph, "ph_int64", TF_INT64);
        CHECK(TF_OperationOutputType(ph.output(0)) == TF_INT64);
    }
    
    SUBCASE("bool placeholder") {
        auto ph = Placeholder(graph, "ph_bool", TF_BOOL);
        CHECK(TF_OperationOutputType(ph.output(0)) == TF_BOOL);
    }
}

TEST_CASE("Identity preserves dtype through T attribute") {
    Graph graph;
    
    auto t = Tensor::FromScalar<double>(3.14159);
    auto c = Const(graph, "c", t.handle(), TF_DOUBLE);
    auto id = Identity(graph, "id", c.output(0), TF_DOUBLE);
    
    CHECK(TF_OperationOutputType(id.output(0)) == TF_DOUBLE);
}

TEST_CASE("Cast uses DstT for output dtype (not SrcT or T)") {
    Graph graph;
    
    // Cast from float to int32
    auto t = Tensor::FromScalar<float>(3.7f);
    auto c = Const(graph, "src", t.handle(), TF_FLOAT);
    auto cast = Cast(graph, "cast", c.output(0), TF_FLOAT, TF_INT32);
    
    // Output should be int32 (DstT), not float (SrcT)
    CHECK(TF_OperationOutputType(cast.output(0)) == TF_INT32);
}

TEST_CASE("Binary ops track dtype through T attribute") {
    Graph graph;
    
    auto a = Tensor::FromScalar<float>(1.0f);
    auto b = Tensor::FromScalar<float>(2.0f);
    
    auto ca = Const(graph, "a", a.handle(), TF_FLOAT);
    auto cb = Const(graph, "b", b.handle(), TF_FLOAT);
    
    SUBCASE("Add") {
        auto sum = Add(graph, "sum", ca, cb, TF_FLOAT);
        CHECK(TF_OperationOutputType(sum.output(0)) == TF_FLOAT);
    }
    
    SUBCASE("Sub") {
        auto diff = Sub(graph, "diff", ca, cb, TF_FLOAT);
        CHECK(TF_OperationOutputType(diff.output(0)) == TF_FLOAT);
    }
    
    SUBCASE("Mul") {
        auto prod = Mul(graph, "prod", ca, cb, TF_FLOAT);
        CHECK(TF_OperationOutputType(prod.output(0)) == TF_FLOAT);
    }
    
    SUBCASE("Div") {
        auto quot = Div(graph, "quot", ca, cb, TF_FLOAT);
        CHECK(TF_OperationOutputType(quot.output(0)) == TF_FLOAT);
    }
}

TEST_CASE("Dtype chains through multiple operations") {
    Graph graph;
    
    auto t = Tensor::FromScalar<std::int64_t>(100);
    auto c = Const(graph, "c", t.handle(), TF_INT64);
    auto id1 = Identity(graph, "id1", c.output(0), TF_INT64);
    auto id2 = Identity(graph, "id2", id1.output(0), TF_INT64);
    auto id3 = Identity(graph, "id3", id2.output(0), TF_INT64);
    
    // All should maintain int64 dtype
    CHECK(TF_OperationOutputType(c.output(0)) == TF_INT64);
    CHECK(TF_OperationOutputType(id1.output(0)) == TF_INT64);
    CHECK(TF_OperationOutputType(id2.output(0)) == TF_INT64);
    CHECK(TF_OperationOutputType(id3.output(0)) == TF_INT64);
}

} // TEST_SUITE P0: Stub Dtype Tracking

// ============================================================================
// P0: Bool Tensor Safety Tests
// ============================================================================

TEST_SUITE("P0: Bool Tensor Safety") {

TEST_CASE("sizeof(bool) == 1 (static_assert in tensor.hpp)") {
    // This test documents the requirement; the static_assert ensures it at compile time
    CHECK(sizeof(bool) == 1);
}

TEST_CASE("Bool tensor round-trip preserves values") {
    std::vector<bool> input = {true, false, true, true, false};
    auto tensor = Tensor::FromVector<bool>({5}, input);
    
    CHECK(tensor.dtype() == TF_BOOL);
    CHECK(tensor.num_elements() == 5);
    
    auto output = tensor.ToVector<bool>();
    REQUIRE(output.size() == input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        CHECK(output[i] == input[i]);
    }
}

TEST_CASE("Bool tensor raw bytes are 0x00 and 0x01") {
    auto tensor = Tensor::FromVector<bool>({3}, {false, true, false});
    
    auto view = tensor.read<bool>();
    const std::uint8_t* bytes = reinterpret_cast<const std::uint8_t*>(view.data());
    
    CHECK(bytes[0] == 0x00);  // false
    CHECK(bytes[1] == 0x01);  // true
    CHECK(bytes[2] == 0x00);  // false
}

TEST_CASE("Bool scalar tensor") {
    auto t_true = Tensor::FromScalar<bool>(true);
    auto t_false = Tensor::FromScalar<bool>(false);
    
    CHECK(t_true.ToScalar<bool>() == true);
    CHECK(t_false.ToScalar<bool>() == false);
}

TEST_CASE("Bool tensor view iteration") {
    auto tensor = Tensor::FromVector<bool>({4}, {true, false, false, true});
    
    auto view = tensor.read<bool>();
    std::vector<bool> collected;
    for (bool b : view) {
        collected.push_back(b);
    }
    
    REQUIRE(collected.size() == 4);
    CHECK(collected[0] == true);
    CHECK(collected[1] == false);
    CHECK(collected[2] == false);
    CHECK(collected[3] == true);
}

} // TEST_SUITE P0: Bool Tensor Safety

// ============================================================================
// P0: Output Index Bounds Checking Tests
// ============================================================================

TEST_SUITE("P0: Output Index Bounds Checking") {

TEST_CASE("resolve_output succeeds for valid index") {
    Graph graph;
    auto t = Tensor::FromScalar<float>(1.0f);
    auto c = Const(graph, "const", t.handle(), TF_FLOAT);
    
    Session session(graph);
    
    // Index 0 is valid
    TF_Output out = session.resolve_output("const", 0);
    CHECK(out.oper != nullptr);
    CHECK(out.index == 0);
}

TEST_CASE("resolve_output throws for operation not found") {
    Graph graph;
    Session session(graph);
    
    CHECK_THROWS_AS(session.resolve_output("nonexistent", 0), std::runtime_error);
}

TEST_CASE("resolve_output throws for negative index") {
    Graph graph;
    auto t = Tensor::FromScalar<float>(1.0f);
    Const(graph, "const", t.handle(), TF_FLOAT);
    
    Session session(graph);
    
    CHECK_THROWS_AS(session.resolve_output("const", -1), std::out_of_range);
}

TEST_CASE("resolve_output throws for index >= num_outputs") {
    Graph graph;
    auto t = Tensor::FromScalar<float>(1.0f);
    Const(graph, "const", t.handle(), TF_FLOAT);
    
    Session session(graph);
    
    // Const has 1 output, so index 1 is out of range
    CHECK_THROWS_AS(session.resolve_output("const", 1), std::out_of_range);
    CHECK_THROWS_AS(session.resolve_output("const", 100), std::out_of_range);
}

TEST_CASE("Run validates fetch indices") {
    Graph graph;
    auto t = Tensor::FromScalar<float>(1.0f);
    Const(graph, "const", t.handle(), TF_FLOAT);
    
    Session session(graph);
    
    // Valid fetch
    CHECK_NOTHROW(session.Run({Fetch{"const", 0}}));
    
    // Invalid fetch index
    CHECK_THROWS_AS(session.Run({Fetch{"const", 1}}), std::out_of_range);
}

TEST_CASE("Run validates feed indices") {
    Graph graph;
    auto ph = Placeholder(graph, "input", TF_FLOAT);
    auto id = Identity(graph, "output", ph.output(0), TF_FLOAT);
    
    Session session(graph);
    
    auto input = Tensor::FromScalar<float>(42.0f);
    
    // Valid feed
    CHECK_NOTHROW(session.Run({Feed{"input", 0, input}}, {Fetch{"output", 0}}));
    
    // Invalid feed index
    CHECK_THROWS_AS(
        session.Run({Feed{"input", 1, input}}, {Fetch{"output", 0}}),
        std::out_of_range);
}

} // TEST_SUITE P0: Output Index Bounds Checking

// ============================================================================
// P1: TensorName Parsing Tests
// ============================================================================

TEST_SUITE("P1: TensorName Parsing") {

TEST_CASE("Parse simple name without index") {
    auto tn = easy::TensorName::parse("my_op");
    CHECK(tn.op == "my_op");
    CHECK(tn.index == 0);
    CHECK(tn.had_explicit_index == false);
}

TEST_CASE("Parse name with index 0") {
    auto tn = easy::TensorName::parse("my_op:0");
    CHECK(tn.op == "my_op");
    CHECK(tn.index == 0);
    CHECK(tn.had_explicit_index == true);
}

TEST_CASE("Parse name with non-zero index") {
    auto tn = easy::TensorName::parse("my_op:3");
    CHECK(tn.op == "my_op");
    CHECK(tn.index == 3);
    CHECK(tn.had_explicit_index == true);
}

TEST_CASE("Parse name with large index") {
    auto tn = easy::TensorName::parse("op:42");
    CHECK(tn.op == "op");
    CHECK(tn.index == 42);
}

TEST_CASE("Parse name with scope (slash)") {
    auto tn = easy::TensorName::parse("scope/subscope/op:0");
    CHECK(tn.op == "scope/subscope/op");
    CHECK(tn.index == 0);
}

TEST_CASE("Whitespace is trimmed") {
    SUBCASE("leading whitespace") {
        auto tn = easy::TensorName::parse("  op:0");
        CHECK(tn.op == "op");
        CHECK(tn.index == 0);
    }
    
    SUBCASE("trailing whitespace") {
        auto tn = easy::TensorName::parse("op:0  ");
        CHECK(tn.op == "op");
        CHECK(tn.index == 0);
    }
    
    SUBCASE("both") {
        auto tn = easy::TensorName::parse("  op:0  ");
        CHECK(tn.op == "op");
        CHECK(tn.index == 0);
    }
}

TEST_CASE("Empty string throws") {
    CHECK_THROWS_AS(easy::TensorName::parse(""), std::invalid_argument);
    CHECK_THROWS_AS(easy::TensorName::parse("   "), std::invalid_argument);
}

TEST_CASE("Empty op name throws") {
    CHECK_THROWS_AS(easy::TensorName::parse(":0"), std::invalid_argument);
}

TEST_CASE("Missing index after colon throws") {
    CHECK_THROWS_AS(easy::TensorName::parse("op:"), std::invalid_argument);
}

TEST_CASE("Non-numeric after colon is treated as part of name") {
    // "op:name" - the colon is part of the name, not a separator
    auto tn = easy::TensorName::parse("op:name");
    CHECK(tn.op == "op:name");
    CHECK(tn.index == 0);
    CHECK(tn.had_explicit_index == false);
}

TEST_CASE("to_string round-trip") {
    SUBCASE("with explicit index") {
        auto tn = easy::TensorName::parse("op:5");
        CHECK(tn.to_string() == "op:5");
    }
    
    SUBCASE("without explicit index") {
        auto tn = easy::TensorName::parse("op");
        CHECK(tn.to_string() == "op");
    }
    
    SUBCASE("explicit index 0") {
        auto tn = easy::TensorName::parse("op:0");
        CHECK(tn.to_string() == "op:0");
    }
}

TEST_CASE("looks_like_tensor_name") {
    CHECK(easy::TensorName::looks_like_tensor_name("op"));
    CHECK(easy::TensorName::looks_like_tensor_name("op:0"));
    CHECK(easy::TensorName::looks_like_tensor_name("scope/op:0"));
    CHECK(easy::TensorName::looks_like_tensor_name("my_op_123"));
    
    CHECK_FALSE(easy::TensorName::looks_like_tensor_name(""));
    CHECK_FALSE(easy::TensorName::looks_like_tensor_name("   "));
}

} // TEST_SUITE P1: TensorName Parsing

// ============================================================================
// P1: Dtype Inference Tests
// ============================================================================

TEST_SUITE("P1: Dtype Inference") {

TEST_CASE("Add infers dtype from inputs") {
    Graph graph;
    
    auto a = Tensor::FromScalar<float>(1.0f);
    auto b = Tensor::FromScalar<float>(2.0f);
    auto ca = Const(graph, "a", a.handle(), TF_FLOAT);
    auto cb = Const(graph, "b", b.handle(), TF_FLOAT);
    
    // Use dtype-inferred Add
    auto sum = easy::Add(graph, "sum", ca, cb);
    CHECK(TF_OperationOutputType(sum.output(0)) == TF_FLOAT);
}

TEST_CASE("Dtype inference throws on mismatch") {
    Graph graph;
    
    auto a = Tensor::FromScalar<float>(1.0f);
    auto b = Tensor::FromScalar<std::int32_t>(2);
    auto ca = Const(graph, "a", a.handle(), TF_FLOAT);
    auto cb = Const(graph, "b", b.handle(), TF_INT32);
    
    // Different dtypes should throw
    CHECK_THROWS_AS(easy::Add(graph, "sum", ca, cb), std::invalid_argument);
}

TEST_CASE("All inferred binary ops work") {
    Graph graph;
    
    auto a = Tensor::FromScalar<double>(3.0);
    auto b = Tensor::FromScalar<double>(2.0);
    auto ca = Const(graph, "a", a.handle(), TF_DOUBLE);
    auto cb = Const(graph, "b", b.handle(), TF_DOUBLE);
    
    SUBCASE("AddV2") {
        auto r = easy::AddV2(graph, "r", ca, cb);
        CHECK(TF_OperationOutputType(r.output(0)) == TF_DOUBLE);
    }
    
    SUBCASE("Sub") {
        auto r = easy::Sub(graph, "r", ca, cb);
        CHECK(TF_OperationOutputType(r.output(0)) == TF_DOUBLE);
    }
    
    SUBCASE("Mul") {
        auto r = easy::Mul(graph, "r", ca, cb);
        CHECK(TF_OperationOutputType(r.output(0)) == TF_DOUBLE);
    }
    
    SUBCASE("Div") {
        auto r = easy::Div(graph, "r", ca, cb);
        CHECK(TF_OperationOutputType(r.output(0)) == TF_DOUBLE);
    }
}

TEST_CASE("MatMul infers dtype") {
    Graph graph;
    
    auto a = Tensor::FromVector<float>({2, 3}, {1, 2, 3, 4, 5, 6});
    auto b = Tensor::FromVector<float>({3, 2}, {1, 2, 3, 4, 5, 6});
    auto ca = Const(graph, "a", a.handle(), TF_FLOAT);
    auto cb = Const(graph, "b", b.handle(), TF_FLOAT);
    
    auto mm = easy::MatMul(graph, "mm", ca, cb);
    CHECK(TF_OperationOutputType(mm.output(0)) == TF_FLOAT);
}

} // TEST_SUITE P1: Dtype Inference

// ============================================================================
// P1: Runner API Tests
// ============================================================================

TEST_SUITE("P1: Runner API") {

TEST_CASE("Runner basic feed and fetch") {
    Graph graph;
    auto ph = Placeholder(graph, "input", TF_FLOAT);
    auto id = Identity(graph, "output", ph.output(0), TF_FLOAT);
    
    Session session(graph);
    
    auto input = Tensor::FromScalar<float>(42.0f);
    auto result = Runner(session)
        .feed("input:0", input)
        .fetch("output:0")
        .run_one();
    
    CHECK(result.ToScalar<float>() == doctest::Approx(42.0f));
}

TEST_CASE("Runner with TF_Output endpoints") {
    Graph graph;
    auto ph = Placeholder(graph, "input", TF_FLOAT);
    auto id = Identity(graph, "output", ph.output(0), TF_FLOAT);
    
    Session session(graph);
    
    auto input = Tensor::FromScalar<float>(7.0f);
    auto result = Runner(session)
        .feed(ph.output(0), input)
        .fetch(id.output(0))
        .run_one();
    
    CHECK(result.ToScalar<float>() == doctest::Approx(7.0f));
}

TEST_CASE("Runner with multiple fetches") {
    Graph graph;
    
    auto t1 = Tensor::FromScalar<float>(10.0f);
    auto t2 = Tensor::FromScalar<float>(20.0f);
    auto c1 = Const(graph, "c1", t1.handle(), TF_FLOAT);
    auto c2 = Const(graph, "c2", t2.handle(), TF_FLOAT);
    
    Session session(graph);
    
    auto results = Runner(session)
        .fetch("c1:0")
        .fetch("c2:0")
        .run();
    
    REQUIRE(results.size() == 2);
    CHECK(results[0].ToScalar<float>() == doctest::Approx(10.0f));
    CHECK(results[1].ToScalar<float>() == doctest::Approx(20.0f));
}

TEST_CASE("Runner run_one throws for wrong number of fetches") {
    Graph graph;
    
    auto t1 = Tensor::FromScalar<float>(1.0f);
    auto t2 = Tensor::FromScalar<float>(2.0f);
    Const(graph, "c1", t1.handle(), TF_FLOAT);
    Const(graph, "c2", t2.handle(), TF_FLOAT);
    
    Session session(graph);
    
    // No fetches
    CHECK_THROWS_AS(Runner(session).run_one(), std::runtime_error);
    
    // Two fetches
    CHECK_THROWS_AS(
        Runner(session).fetch("c1:0").fetch("c2:0").run_one(),
        std::runtime_error);
}

TEST_CASE("Runner validates endpoints") {
    Graph graph;
    auto t = Tensor::FromScalar<float>(1.0f);
    Const(graph, "c", t.handle(), TF_FLOAT);
    
    Session session(graph);
    auto input = Tensor::FromScalar<float>(1.0f);
    
    // Invalid operation name
    CHECK_THROWS(Runner(session).feed("nonexistent:0", input).fetch("c:0").run());
    
    // Invalid index
    CHECK_THROWS(Runner(session).fetch("c:5").run());
}

TEST_CASE("Runner fluent chaining") {
    Graph graph;
    
    auto ph1 = Placeholder(graph, "in1", TF_FLOAT);
    auto ph2 = Placeholder(graph, "in2", TF_FLOAT);
    auto sum = Add(graph, "sum", ph1, ph2, TF_FLOAT);
    
    Session session(graph);
    
    auto a = Tensor::FromScalar<float>(3.0f);
    auto b = Tensor::FromScalar<float>(4.0f);
    
    // All chained in one expression
    auto result = Runner(session)
        .feed("in1:0", a)
        .feed("in2:0", b)
        .fetch("sum:0")
        .run_one();
    
    CHECK(result.ToScalar<float>() == doctest::Approx(7.0f));
}

} // TEST_SUITE P1: Runner API

// ============================================================================
// P1: Easy Helpers Tests
// ============================================================================

TEST_SUITE("P1: Easy Helpers") {

TEST_CASE("Scalar helper creates constant") {
    Graph graph;
    auto c = easy::Scalar<float>(graph, "c", 3.14f);
    
    CHECK(TF_OperationOutputType(c.output(0)) == TF_FLOAT);
    
    Session session(graph);
    auto result = session.Run("c");
    CHECK(result.ToScalar<float>() == doctest::Approx(3.14f));
}

TEST_CASE("Scalar with different types") {
    Graph graph;
    
    easy::Scalar<std::int32_t>(graph, "i32", 42);
    easy::Scalar<std::int64_t>(graph, "i64", 100);
    easy::Scalar<double>(graph, "f64", 2.718);
    
    Session session(graph);
    
    CHECK(session.Run("i32").ToScalar<std::int32_t>() == 42);
    CHECK(session.Run("i64").ToScalar<std::int64_t>() == 100);
    CHECK(session.Run("f64").ToScalar<double>() == doctest::Approx(2.718));
}

TEST_CASE("Placeholder helper") {
    Graph graph;
    auto ph = easy::Placeholder<float>(graph, "ph");
    
    CHECK(TF_OperationOutputType(ph.output(0)) == TF_FLOAT);
}

TEST_CASE("Identity helper") {
    Graph graph;
    auto c = easy::Scalar<float>(graph, "c", 5.0f);
    auto id = easy::Identity<float>(graph, "id", c.output(0));
    
    CHECK(TF_OperationOutputType(id.output(0)) == TF_FLOAT);
    
    Session session(graph);
    CHECK(session.Run("id").ToScalar<float>() == doctest::Approx(5.0f));
}

} // TEST_SUITE P1: Easy Helpers

// ============================================================================
// Integration Tests
// ============================================================================

TEST_SUITE("Integration Tests") {

TEST_CASE("Full pipeline: build graph, run with Runner") {
    Graph graph;
    
    // Build a simple computation graph: (a + b) * c
    auto ph_a = easy::Placeholder<float>(graph, "a");
    auto ph_b = easy::Placeholder<float>(graph, "b");
    auto ph_c = easy::Placeholder<float>(graph, "c");
    
    auto sum = easy::Add(graph, "sum", ph_a, ph_b);  // dtype inferred
    auto product = easy::Mul(graph, "result", sum, ph_c);  // dtype inferred
    
    Session session(graph);
    
    auto a = Tensor::FromScalar<float>(2.0f);
    auto b = Tensor::FromScalar<float>(3.0f);
    auto c = Tensor::FromScalar<float>(4.0f);
    
    auto result = Runner(session)
        .feed("a:0", a)
        .feed("b:0", b)
        .feed("c:0", c)
        .fetch("result:0")
        .run_one();
    
    // (2 + 3) * 4 = 20
    CHECK(result.ToScalar<float>() == doctest::Approx(20.0f));
}

TEST_CASE("TensorName resolution through Runner") {
    Graph graph;
    
    auto c = easy::Scalar<float>(graph, "const", 99.0f);
    auto id = easy::Identity<float>(graph, "output", c.output(0));
    
    Session session(graph);
    
    // Various ways to refer to the same output
    auto r1 = Runner(session).fetch("output:0").run_one();
    auto r2 = Runner(session).fetch("output").run_one();  // implicit :0
    auto r3 = Runner(session).fetch(easy::TensorName::parse("output:0")).run_one();
    
    CHECK(r1.ToScalar<float>() == doctest::Approx(99.0f));
    CHECK(r2.ToScalar<float>() == doctest::Approx(99.0f));
    CHECK(r3.ToScalar<float>() == doctest::Approx(99.0f));
}

} // TEST_SUITE Integration Tests
