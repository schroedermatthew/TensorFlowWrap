// tests/test_improvements.cpp
// Tests for P0/P1 improvements: dtype tracking, bool safety, bounds checking, easy layer

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

// Suppress warnings from CHECK_THROWS_AS with [[nodiscard]] functions
#if defined(_MSC_VER)
#pragma warning(disable: 4834)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

#include "tf_wrap/all.hpp"

#include <cstdint>
#include <string>
#include <vector>

using namespace tf_wrap;
using namespace tf_wrap::ops;

// ============================================================================
// P0: Stub dtype tracking
// ============================================================================

TEST_SUITE("Stub dtype tracking") {

TEST_CASE("Const tracks dtype") {
    Graph graph;
    auto t = Tensor::FromScalar<float>(3.14f);
    auto c = Const(graph, "const_f32", t.handle(), TF_FLOAT);
    CHECK(TF_OperationOutputType(c.output(0)) == TF_FLOAT);
}

TEST_CASE("Const tracks different dtypes") {
    Graph graph;
    
    auto t_f64 = Tensor::FromScalar<double>(2.718);
    auto c_f64 = Const(graph, "const_f64", t_f64.handle(), TF_DOUBLE);
    CHECK(TF_OperationOutputType(c_f64.output(0)) == TF_DOUBLE);
    
    auto t_i32 = Tensor::FromScalar<std::int32_t>(42);
    auto c_i32 = Const(graph, "const_i32", t_i32.handle(), TF_INT32);
    CHECK(TF_OperationOutputType(c_i32.output(0)) == TF_INT32);
    
    auto t_i64 = Tensor::FromScalar<std::int64_t>(100);
    auto c_i64 = Const(graph, "const_i64", t_i64.handle(), TF_INT64);
    CHECK(TF_OperationOutputType(c_i64.output(0)) == TF_INT64);
}

TEST_CASE("Placeholder tracks dtype") {
    Graph graph;
    std::vector<std::int64_t> shape = {};
    
    auto ph_f32 = Placeholder(graph, "ph_float", TF_FLOAT, shape);
    CHECK(TF_OperationOutputType(ph_f32.output(0)) == TF_FLOAT);
    
    auto ph_f64 = Placeholder(graph, "ph_double", TF_DOUBLE, shape);
    CHECK(TF_OperationOutputType(ph_f64.output(0)) == TF_DOUBLE);
    
    auto ph_i32 = Placeholder(graph, "ph_int32", TF_INT32, shape);
    CHECK(TF_OperationOutputType(ph_i32.output(0)) == TF_INT32);
}

TEST_CASE("Identity tracks dtype via T attr") {
    Graph graph;
    auto t = Tensor::FromScalar<float>(1.0f);
    auto c = Const(graph, "c", t.handle(), TF_FLOAT);
    auto id = Identity(graph, "id", c.output(0), TF_FLOAT);
    CHECK(TF_OperationOutputType(id.output(0)) == TF_FLOAT);
}

TEST_CASE("Cast uses DstT for output dtype") {
    Graph graph;
    auto t = Tensor::FromScalar<float>(1.0f);
    auto c = Const(graph, "c", t.handle(), TF_FLOAT);
    auto cast = Cast(graph, "cast", c.output(0), TF_FLOAT, TF_INT32);
    CHECK(TF_OperationOutputType(cast.output(0)) == TF_INT32);
}

TEST_CASE("Binary ops track dtype") {
    Graph graph;
    auto t1 = Tensor::FromScalar<float>(1.0f);
    auto t2 = Tensor::FromScalar<float>(2.0f);
    auto c1 = Const(graph, "c1", t1.handle(), TF_FLOAT);
    auto c2 = Const(graph, "c2", t2.handle(), TF_FLOAT);
    
    auto sum = Add(graph, "sum", c1.output(0), c2.output(0), TF_FLOAT);
    CHECK(TF_OperationOutputType(sum.output(0)) == TF_FLOAT);
    
    auto prod = Mul(graph, "prod", c1.output(0), c2.output(0), TF_FLOAT);
    CHECK(TF_OperationOutputType(prod.output(0)) == TF_FLOAT);
}

TEST_CASE("Dtype chains correctly") {
    Graph graph;
    auto t = Tensor::FromScalar<float>(1.0f);
    auto c = Const(graph, "c", t.handle(), TF_FLOAT);
    auto id1 = Identity(graph, "id1", c.output(0), TF_FLOAT);
    auto id2 = Identity(graph, "id2", id1.output(0), TF_FLOAT);
    auto cast = Cast(graph, "cast", id2.output(0), TF_FLOAT, TF_DOUBLE);
    auto id3 = Identity(graph, "id3", cast.output(0), TF_DOUBLE);
    
    CHECK(TF_OperationOutputType(c.output(0)) == TF_FLOAT);
    CHECK(TF_OperationOutputType(id1.output(0)) == TF_FLOAT);
    CHECK(TF_OperationOutputType(id2.output(0)) == TF_FLOAT);
    CHECK(TF_OperationOutputType(cast.output(0)) == TF_DOUBLE);
    CHECK(TF_OperationOutputType(id3.output(0)) == TF_DOUBLE);
}

} // TEST_SUITE

// ============================================================================
// P0: Bool tensor safety
// ============================================================================

TEST_SUITE("Bool tensor safety") {

TEST_CASE("Bool tensor round-trip") {
    std::vector<bool> input = {true, false, true, false, true};
    auto t = Tensor::FromVector<bool>({5}, input);
    
    CHECK(t.dtype() == TF_BOOL);
    CHECK(t.num_elements() == 5);
    
    auto output = t.ToVector<bool>();
    CHECK(output == input);
}

TEST_CASE("Bool tensor raw bytes") {
    std::vector<bool> input = {true, false, true};
    auto t = Tensor::FromVector<bool>({3}, input);
    
    // Check raw bytes are 0x00 or 0x01
    auto view = t.read<bool>();
    CHECK(view[0] == true);
    CHECK(view[1] == false);
    CHECK(view[2] == true);
}

TEST_CASE("Bool scalar") {
    auto t_true = Tensor::FromScalar<bool>(true);
    auto t_false = Tensor::FromScalar<bool>(false);
    
    CHECK(t_true.ToScalar<bool>() == true);
    CHECK(t_false.ToScalar<bool>() == false);
}

TEST_CASE("Bool tensor read iteration") {
    std::vector<bool> input = {true, true, false, false, true};
    auto t = Tensor::FromVector<bool>({5}, input);
    
    auto view = t.read<bool>();
    std::vector<bool> output(view.begin(), view.end());
    CHECK(output == input);
}

TEST_CASE("Empty bool vector") {
    std::vector<bool> empty;
    auto t = Tensor::FromVector<bool>({0}, empty);
    CHECK(t.num_elements() == 0);
    CHECK(t.dtype() == TF_BOOL);
}

} // TEST_SUITE

// ============================================================================
// P0: Output index bounds checking
// ============================================================================

TEST_SUITE("Output bounds checking") {

TEST_CASE("resolve_output finds existing op") {
    Graph graph;
    auto t = Tensor::FromScalar<float>(1.0f);
    Const(graph, "my_const", t.handle(), TF_FLOAT);
    
    Session session(graph);
    auto output = session.resolve_output("my_const", 0);
    CHECK(output.oper != nullptr);
    CHECK(output.index == 0);
}

TEST_CASE("resolve_output throws for nonexistent op") {
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

TEST_CASE("resolve_output throws for out of range index") {
    Graph graph;
    auto t = Tensor::FromScalar<float>(1.0f);
    Const(graph, "const", t.handle(), TF_FLOAT);
    
    Session session(graph);
    // Const has 1 output, so index 1 is out of range
    CHECK_THROWS_AS(session.resolve_output("const", 1), std::out_of_range);
    CHECK_THROWS_AS(session.resolve_output("const", 100), std::out_of_range);
}

TEST_CASE("Session::Run validates fetch indices") {
    Graph graph;
    auto t = Tensor::FromScalar<float>(1.0f);
    Const(graph, "const", t.handle(), TF_FLOAT);
    
    Session session(graph);
    
    // Valid fetch
    CHECK_NOTHROW(session.Run({Fetch{"const", 0}}));
    
    // Invalid fetch index
    CHECK_THROWS_AS(session.Run({Fetch{"const", 1}}), std::out_of_range);
}

TEST_CASE("Session::Run validates feed indices") {
    Graph graph;
    std::vector<std::int64_t> shape = {};
    auto ph = Placeholder(graph, "input", TF_FLOAT, shape);
    Identity(graph, "output", ph.output(0), TF_FLOAT);
    
    Session session(graph);
    auto input = Tensor::FromScalar<float>(5.0f);
    
    // Valid feed
    CHECK_NOTHROW(session.Run({Feed{"input", 0, input}}, {Fetch{"output", 0}}));
    
    // Invalid feed index
    CHECK_THROWS_AS(
        session.Run({Feed{"input", 1, input}}, {Fetch{"output", 0}}),
        std::out_of_range);
}

} // TEST_SUITE

// ============================================================================
// P1: TensorName parsing
// ============================================================================

TEST_SUITE("TensorName parsing") {

TEST_CASE("Simple op name without index") {
    auto tn = easy::TensorName::parse("my_op");
    CHECK(tn.op == "my_op");
    CHECK(tn.index == 0);
    CHECK(tn.had_explicit_index == false);
}

TEST_CASE("Op name with index 0") {
    auto tn = easy::TensorName::parse("my_op:0");
    CHECK(tn.op == "my_op");
    CHECK(tn.index == 0);
    CHECK(tn.had_explicit_index == true);
}

TEST_CASE("Op name with index > 0") {
    auto tn = easy::TensorName::parse("my_op:5");
    CHECK(tn.op == "my_op");
    CHECK(tn.index == 5);
    CHECK(tn.had_explicit_index == true);
}

TEST_CASE("Scoped op name") {
    auto tn = easy::TensorName::parse("scope/inner/op:2");
    CHECK(tn.op == "scope/inner/op");
    CHECK(tn.index == 2);
}

TEST_CASE("Whitespace trimming") {
    auto tn = easy::TensorName::parse("  my_op:1  ");
    CHECK(tn.op == "my_op");
    CHECK(tn.index == 1);
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

TEST_CASE("Round-trip to_string") {
    auto tn1 = easy::TensorName::parse("op:0");
    CHECK(tn1.to_string() == "op:0");
    
    auto tn2 = easy::TensorName::parse("op");
    CHECK(tn2.to_string() == "op");
    
    auto tn3 = easy::TensorName::parse("scope/op:3");
    CHECK(tn3.to_string() == "scope/op:3");
}

} // TEST_SUITE

// ============================================================================
// P1: Dtype inference helpers
// ============================================================================

TEST_SUITE("Dtype inference") {

TEST_CASE("easy::Add infers dtype") {
    Graph graph;
    auto t1 = Tensor::FromScalar<float>(1.0f);
    auto t2 = Tensor::FromScalar<float>(2.0f);
    auto c1 = Const(graph, "c1", t1.handle(), TF_FLOAT);
    auto c2 = Const(graph, "c2", t2.handle(), TF_FLOAT);
    
    auto sum = easy::Add(graph, "sum", c1.output(0), c2.output(0));
    CHECK(TF_OperationOutputType(sum.output(0)) == TF_FLOAT);
}

TEST_CASE("easy::Mul infers dtype") {
    Graph graph;
    auto t1 = Tensor::FromScalar<double>(1.0);
    auto t2 = Tensor::FromScalar<double>(2.0);
    auto c1 = Const(graph, "c1", t1.handle(), TF_DOUBLE);
    auto c2 = Const(graph, "c2", t2.handle(), TF_DOUBLE);
    
    auto prod = easy::Mul(graph, "prod", c1.output(0), c2.output(0));
    CHECK(TF_OperationOutputType(prod.output(0)) == TF_DOUBLE);
}

TEST_CASE("easy::Add throws on dtype mismatch") {
    Graph graph;
    auto t1 = Tensor::FromScalar<float>(1.0f);
    auto t2 = Tensor::FromScalar<std::int32_t>(2);
    auto c1 = Const(graph, "c1", t1.handle(), TF_FLOAT);
    auto c2 = Const(graph, "c2", t2.handle(), TF_INT32);
    
    CHECK_THROWS_AS(easy::Add(graph, "sum", c1.output(0), c2.output(0)), std::invalid_argument);
}

TEST_CASE("easy::MatMul infers dtype") {
    Graph graph;
    auto t1 = Tensor::FromScalar<float>(1.0f);
    auto t2 = Tensor::FromScalar<float>(2.0f);
    auto c1 = Const(graph, "c1", t1.handle(), TF_FLOAT);
    auto c2 = Const(graph, "c2", t2.handle(), TF_FLOAT);
    
    auto mm = easy::MatMul(graph, "mm", c1.output(0), c2.output(0));
    CHECK(TF_OperationOutputType(mm.output(0)) == TF_FLOAT);
}

} // TEST_SUITE

// ============================================================================
// P1: Runner API
// ============================================================================

TEST_SUITE("Runner API") {

TEST_CASE("Runner basic feed/fetch") {
    Graph graph;
    std::vector<std::int64_t> shape = {};
    auto ph = Placeholder(graph, "input", TF_FLOAT, shape);
    Identity(graph, "output", ph.output(0), TF_FLOAT);
    
    Session session(graph);
    auto input = Tensor::FromScalar<float>(42.0f);
    
    auto result = easy::Runner(session)
        .feed("input:0", input)
        .fetch("output:0")
        .run_one();
    
    CHECK(result.ToScalar<float>() == doctest::Approx(42.0f));
}

TEST_CASE("Runner multiple fetches") {
    Graph graph;
    auto t1 = Tensor::FromScalar<float>(1.0f);
    auto t2 = Tensor::FromScalar<float>(2.0f);
    Const(graph, "c1", t1.handle(), TF_FLOAT);
    Const(graph, "c2", t2.handle(), TF_FLOAT);
    
    Session session(graph);
    
    auto results = easy::Runner(session)
        .fetch("c1:0")
        .fetch("c2:0")
        .run();
    
    CHECK(results.size() == 2);
    CHECK(results[0].ToScalar<float>() == doctest::Approx(1.0f));
    CHECK(results[1].ToScalar<float>() == doctest::Approx(2.0f));
}

TEST_CASE("Runner validates endpoints") {
    Graph graph;
    auto t = Tensor::FromScalar<float>(1.0f);
    Const(graph, "c", t.handle(), TF_FLOAT);
    
    Session session(graph);
    auto input = Tensor::FromScalar<float>(1.0f);
    
    // Nonexistent op
    CHECK_THROWS(easy::Runner(session).feed("nonexistent:0", input).fetch("c:0").run());
    
    // Out of range index
    CHECK_THROWS(easy::Runner(session).fetch("c:5").run());
}

TEST_CASE("Runner run_one throws on wrong fetch count") {
    Graph graph;
    auto t1 = Tensor::FromScalar<float>(1.0f);
    auto t2 = Tensor::FromScalar<float>(2.0f);
    Const(graph, "c1", t1.handle(), TF_FLOAT);
    Const(graph, "c2", t2.handle(), TF_FLOAT);
    
    Session session(graph);
    
    // No fetches
    CHECK_THROWS_AS(easy::Runner(session).run_one(), std::runtime_error);
    
    // Multiple fetches
    CHECK_THROWS_AS(
        easy::Runner(session).fetch("c1:0").fetch("c2:0").run_one(),
        std::runtime_error);
}

TEST_CASE("Runner fluent chaining") {
    Graph graph;
    std::vector<std::int64_t> shape = {};
    auto ph1 = Placeholder(graph, "in1", TF_FLOAT, shape);
    auto ph2 = Placeholder(graph, "in2", TF_FLOAT, shape);
    Add(graph, "sum", ph1.output(0), ph2.output(0), TF_FLOAT);
    
    Session session(graph);
    auto t1 = Tensor::FromScalar<float>(3.0f);
    auto t2 = Tensor::FromScalar<float>(4.0f);
    
    auto result = easy::Runner(session)
        .feed("in1:0", t1)
        .feed("in2:0", t2)
        .fetch("sum:0")
        .run_one();
    
    CHECK(result.ToScalar<float>() == doctest::Approx(7.0f));
}

} // TEST_SUITE

// ============================================================================
// P1: Easy helpers
// ============================================================================

TEST_SUITE("Easy helpers") {

TEST_CASE("easy::Scalar creates typed constants") {
    Graph graph;
    
    auto sf = easy::Scalar<float>(graph, "sf", 3.14f);
    CHECK(TF_OperationOutputType(sf.output(0)) == TF_FLOAT);
    
    auto sd = easy::Scalar<double>(graph, "sd", 2.718);
    CHECK(TF_OperationOutputType(sd.output(0)) == TF_DOUBLE);
    
    auto si = easy::Scalar<std::int32_t>(graph, "si", 42);
    CHECK(TF_OperationOutputType(si.output(0)) == TF_INT32);
}

TEST_CASE("easy::Placeholder creates typed placeholders") {
    Graph graph;
    
    auto ph_f = easy::Placeholder<float>(graph, "ph_f");
    CHECK(TF_OperationOutputType(ph_f.output(0)) == TF_FLOAT);
    
    auto ph_d = easy::Placeholder<double>(graph, "ph_d");
    CHECK(TF_OperationOutputType(ph_d.output(0)) == TF_DOUBLE);
    
    auto ph_i = easy::Placeholder<std::int32_t>(graph, "ph_i");
    CHECK(TF_OperationOutputType(ph_i.output(0)) == TF_INT32);
}

TEST_CASE("easy::Identity creates typed identity") {
    Graph graph;
    auto c = easy::Scalar<float>(graph, "c", 1.0f);
    auto id = easy::Identity<float>(graph, "id", c.output(0));
    CHECK(TF_OperationOutputType(id.output(0)) == TF_FLOAT);
}

} // TEST_SUITE

// ============================================================================
// Integration
// ============================================================================

TEST_SUITE("Integration") {

TEST_CASE("Full pipeline: placeholder -> identity -> run") {
    Graph graph;
    auto ph = easy::Placeholder<float>(graph, "input");
    easy::Identity<float>(graph, "output", ph.output(0));
    
    Session session(graph);
    auto input = Tensor::FromScalar<float>(99.0f);
    
    auto result = easy::Runner(session)
        .feed("input:0", input)
        .fetch("output:0")
        .run_one();
    
    CHECK(result.ToScalar<float>() == doctest::Approx(99.0f));
}

TEST_CASE("TensorName resolution in Runner") {
    Graph graph;
    auto c = easy::Scalar<float>(graph, "my_const", 42.0f);
    
    Session session(graph);
    
    // Using string
    auto r1 = easy::Runner(session).fetch("my_const:0").run_one();
    CHECK(r1.ToScalar<float>() == doctest::Approx(42.0f));
    
    // Using Endpoint with TF_Output directly
    auto r2 = easy::Runner(session).fetch(c.output(0)).run_one();
    CHECK(r2.ToScalar<float>() == doctest::Approx(42.0f));
}

} // TEST_SUITE
