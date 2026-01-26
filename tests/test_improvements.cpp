// tests/test_improvements.cpp
// Tests for P0/P1 improvements: dtype tracking, bool safety, bounds checking, facade layer

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

// Suppress warnings from CHECK_THROWS_AS with [[nodiscard]] functions
#if defined(_MSC_VER)
#pragma warning(disable: 4834)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

#include "tf_wrap/core.hpp"
#include "tf_wrap/facade_ops.hpp"
#include "tf_wrap/ops/cast.hpp"

#include <cstdint>
#include <cstdlib>
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

TEST_CASE("resolve_output parses explicit index suffix") {
    Graph graph;
    auto t = Tensor::FromScalar<float>(1.0f);
    Const(graph, "const", t.handle(), TF_FLOAT);

    Session session(graph);

    // If ":1" were treated as part of the op name, this would be NOT_FOUND.
    // With tensor-name parsing, this resolves op "const" and then fails
    // bounds-checking with OUT_OF_RANGE.
    try {
        (void)session.resolve_output("const:1");
        FAIL("Expected resolve_output(const:1) to throw");
    } catch (const tf_wrap::Error& e) {
        CHECK(e.code() == TF_OUT_OF_RANGE);
        CHECK(e.op_name() == "const");
        CHECK(e.index() == 1);
    }
}

TEST_CASE("resolve_output detects conflicting indices") {
    Graph graph;
    auto t = Tensor::FromScalar<float>(1.0f);
    Const(graph, "const", t.handle(), TF_FLOAT);

    Session session(graph);

    try {
        (void)session.resolve_output("const:0", 1);
        FAIL("Expected resolve_output(const:0, 1) to throw");
    } catch (const tf_wrap::Error& e) {
        CHECK(e.code() == TF_INVALID_ARGUMENT);
    }
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
    CHECK_THROWS_AS(session.resolve_output("const", -1), std::runtime_error);
}

TEST_CASE("resolve_output throws for out of range index") {
    Graph graph;
    auto t = Tensor::FromScalar<float>(1.0f);
    Const(graph, "const", t.handle(), TF_FLOAT);
    
    Session session(graph);
    // Const has 1 output, so index 1 is out of range
    CHECK_THROWS_AS(session.resolve_output("const", 1), std::runtime_error);
    CHECK_THROWS_AS(session.resolve_output("const", 100), std::runtime_error);
}

TEST_CASE("Session::Run validates fetch indices") {
    Graph graph;
    auto t = Tensor::FromScalar<float>(1.0f);
    Const(graph, "const", t.handle(), TF_FLOAT);
    
    Session session(graph);
    
    // Valid fetch
    CHECK_NOTHROW(session.Run({Fetch{"const", 0}}));
    
    // Invalid fetch index
    CHECK_THROWS_AS(session.Run({Fetch{"const", 1}}), std::runtime_error);
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
        std::runtime_error);
}

TEST_CASE("Feed and Fetch parse 'op:index' names") {
    auto t = Tensor::FromScalar<float>(1.0f);

    Feed f1{"input:3", t};
    CHECK(f1.op_name == "input");
    CHECK(f1.index == 3);

    // Colon not followed by digits is treated as part of the op name.
    Feed f2{"scope/op:name", t};
    CHECK(f2.op_name == "scope/op:name");
    CHECK(f2.index == 0);

    Fetch k1{"output:2"};
    CHECK(k1.op_name == "output");
    CHECK(k1.index == 2);

    Fetch k2{"scope/op:name", 7};
    CHECK(k2.op_name == "scope/op:name");
    CHECK(k2.index == 7);

    // Conflicting explicit indices are rejected (when the caller passes a non-zero idx).
    CHECK_THROWS_AS((Fetch{"output:0", 1}), std::invalid_argument);
    CHECK_THROWS_AS((Feed{"input:0", 1, t}), std::invalid_argument);
}

TEST_CASE("OpResult::output throws for invalid index") {
    Graph graph;
    auto t = Tensor::FromScalar<float>(1.0f);
    auto c = Const(graph, "c", t.handle(), TF_FLOAT);

    // Const has exactly 1 output
    CHECK_NOTHROW(c.output(0));
    CHECK_THROWS_AS(c.output(1), std::out_of_range);
    CHECK_THROWS_AS(c.output(-1), std::out_of_range);

    // Unchecked variant preserves old behavior
    CHECK(c.output_unchecked(1).oper == c.op());
    CHECK(c.output_unchecked(1).index == 1);
}

TEST_CASE("Operation::output throws for invalid index") {
    Graph graph;
    auto t = Tensor::FromScalar<float>(1.0f);
    auto c = Const(graph, "c", t.handle(), TF_FLOAT);

    Operation op(c.op());

    CHECK_NOTHROW(op.output(0));
    CHECK_THROWS_AS(op.output(1), std::out_of_range);
    CHECK_THROWS_AS(op.output(-1), std::out_of_range);

    // Unchecked variant preserves old behavior
    CHECK(op.output_unchecked(1).oper == op.handle());
    CHECK(op.output_unchecked(1).index == 1);
}

} // TEST_SUITE


// ============================================================================
// P0: BatchRunStacked dtype allocation
// ============================================================================

TEST_SUITE("BatchRunStacked") {

TEST_CASE("BatchRunStacked preserves TF_HALF dtype and raw bytes") {
    Graph graph;

    // A simple identity graph: input -> output
    std::vector<std::int64_t> shape = {2};
    auto ph = Placeholder(graph, "input", TF_HALF, shape);
    Identity(graph, "output", ph.output(0), TF_HALF);

    Session session(graph);

    auto make_half = [](std::uint16_t a, std::uint16_t b) {
        std::int64_t dims[1] = {2};
        const std::size_t bytes = 2 * sizeof(std::uint16_t);
        TF_Tensor* raw = TF_AllocateTensor(TF_HALF, dims, 1, bytes);
        REQUIRE(raw != nullptr);
        auto* p = static_cast<std::uint16_t*>(TF_TensorData(raw));
        REQUIRE(p != nullptr);
        p[0] = a;
        p[1] = b;
        return Tensor::FromRaw(raw);
    };

    std::vector<Tensor> inputs;
    inputs.emplace_back(make_half(0x3C00u, 0x4000u)); // 1.0, 2.0 (IEEE 754 half bits)
    inputs.emplace_back(make_half(0x0000u, 0xBC00u)); // 0.0, -1.0
    inputs.emplace_back(make_half(0x3555u, 0x3E00u)); // ~0.333, 1.5

    auto outputs = session.BatchRunStacked("input", inputs, "output");
    REQUIRE(outputs.size() == inputs.size());

    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const Tensor& in = inputs[i];
        const Tensor& out = outputs[i];

        CHECK(out.dtype() == TF_HALF);
        CHECK(out.shape() == in.shape());

        const std::size_t in_bytes = TF_TensorByteSize(in.handle());
        const std::size_t out_bytes = TF_TensorByteSize(out.handle());
        CHECK(out_bytes == in_bytes);

        const void* in_data = TF_TensorData(in.handle());
        const void* out_data = TF_TensorData(out.handle());
        REQUIRE(in_data != nullptr);
        REQUIRE(out_data != nullptr);
        CHECK(std::memcmp(in_data, out_data, in_bytes) == 0);
    }
}

} // TEST_SUITE

// ============================================================================
// P1: TensorName parsing
// ============================================================================

TEST_SUITE("TensorName parsing") {

TEST_CASE("Simple op name without index") {
    auto tn = facade::TensorName::parse("my_op");
    CHECK(tn.op == "my_op");
    CHECK(tn.index == 0);
    CHECK(tn.had_explicit_index == false);
}

TEST_CASE("Op name with index 0") {
    auto tn = facade::TensorName::parse("my_op:0");
    CHECK(tn.op == "my_op");
    CHECK(tn.index == 0);
    CHECK(tn.had_explicit_index == true);
}

TEST_CASE("Op name with index > 0") {
    auto tn = facade::TensorName::parse("my_op:5");
    CHECK(tn.op == "my_op");
    CHECK(tn.index == 5);
    CHECK(tn.had_explicit_index == true);
}

TEST_CASE("Scoped op name") {
    auto tn = facade::TensorName::parse("scope/inner/op:2");
    CHECK(tn.op == "scope/inner/op");
    CHECK(tn.index == 2);
}

TEST_CASE("Whitespace trimming") {
    auto tn = facade::TensorName::parse("  my_op:1  ");
    CHECK(tn.op == "my_op");
    CHECK(tn.index == 1);
}

TEST_CASE("Empty string throws") {
    CHECK_THROWS_AS(facade::TensorName::parse(""), std::invalid_argument);
    CHECK_THROWS_AS(facade::TensorName::parse("   "), std::invalid_argument);
}

TEST_CASE("Empty op name throws") {
    CHECK_THROWS_AS(facade::TensorName::parse(":0"), std::invalid_argument);
}

TEST_CASE("Missing index after colon throws") {
    CHECK_THROWS_AS(facade::TensorName::parse("op:"), std::invalid_argument);
}

TEST_CASE("Round-trip to_string") {
    auto tn1 = facade::TensorName::parse("op:0");
    CHECK(tn1.to_string() == "op:0");
    
    auto tn2 = facade::TensorName::parse("op");
    CHECK(tn2.to_string() == "op");
    
    auto tn3 = facade::TensorName::parse("scope/op:3");
    CHECK(tn3.to_string() == "scope/op:3");
}

} // TEST_SUITE

// ============================================================================
// P1: Dtype inference helpers
// ============================================================================

TEST_SUITE("Dtype inference") {

TEST_CASE("facade::Add infers dtype") {
    Graph graph;
    auto t1 = Tensor::FromScalar<float>(1.0f);
    auto t2 = Tensor::FromScalar<float>(2.0f);
    auto c1 = Const(graph, "c1", t1.handle(), TF_FLOAT);
    auto c2 = Const(graph, "c2", t2.handle(), TF_FLOAT);
    
    auto sum = facade::Add(graph, "sum", c1.output(0), c2.output(0));
    CHECK(TF_OperationOutputType(sum.output(0)) == TF_FLOAT);
}

TEST_CASE("facade::Mul infers dtype") {
    Graph graph;
    auto t1 = Tensor::FromScalar<double>(1.0);
    auto t2 = Tensor::FromScalar<double>(2.0);
    auto c1 = Const(graph, "c1", t1.handle(), TF_DOUBLE);
    auto c2 = Const(graph, "c2", t2.handle(), TF_DOUBLE);
    
    auto prod = facade::Mul(graph, "prod", c1.output(0), c2.output(0));
    CHECK(TF_OperationOutputType(prod.output(0)) == TF_DOUBLE);
}

TEST_CASE("facade::Add throws on dtype mismatch") {
    Graph graph;
    auto t1 = Tensor::FromScalar<float>(1.0f);
    auto t2 = Tensor::FromScalar<std::int32_t>(2);
    auto c1 = Const(graph, "c1", t1.handle(), TF_FLOAT);
    auto c2 = Const(graph, "c2", t2.handle(), TF_INT32);
    
    CHECK_THROWS_AS(facade::Add(graph, "sum", c1.output(0), c2.output(0)), std::invalid_argument);
}

TEST_CASE("facade::MatMul infers dtype") {
    Graph graph;
    auto t1 = Tensor::FromScalar<float>(1.0f);
    auto t2 = Tensor::FromScalar<float>(2.0f);
    auto c1 = Const(graph, "c1", t1.handle(), TF_FLOAT);
    auto c2 = Const(graph, "c2", t2.handle(), TF_FLOAT);
    
    auto mm = facade::MatMul(graph, "mm", c1.output(0), c2.output(0));
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
    
    auto result = facade::Runner(session)
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
    
    auto results = facade::Runner(session)
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
    CHECK_THROWS(facade::Runner(session).feed("nonexistent:0", input).fetch("c:0").run());
    
    // Out of range index
    CHECK_THROWS(facade::Runner(session).fetch("c:5").run());
}


TEST_CASE("Runner validates resolved TF_Output endpoints") {
    Graph graph;
    auto t = Tensor::FromScalar<float>(1.0f);
    auto c = Const(graph, "c", t.handle(), TF_FLOAT);

    Session session(graph);

    TF_Output c0 = session.resolve_output("c", 0);
    TF_Output bad = TF_Output{c0.oper, 1}; // Const has exactly 1 output

    CHECK_THROWS_AS(facade::Runner(session).fetch(Endpoint(bad)).run(), std::out_of_range);

    // Null operation pointer should throw invalid_argument
    TF_Output nullop{nullptr, 0};
    CHECK_THROWS_AS(facade::Runner(session).fetch(Endpoint(nullop)).run(), std::invalid_argument);
}

TEST_CASE("Runner run_one throws on wrong fetch count") {
    Graph graph;
    auto t1 = Tensor::FromScalar<float>(1.0f);
    auto t2 = Tensor::FromScalar<float>(2.0f);
    Const(graph, "c1", t1.handle(), TF_FLOAT);
    Const(graph, "c2", t2.handle(), TF_FLOAT);
    
    Session session(graph);
    
    // No fetches
    CHECK_THROWS_AS(facade::Runner(session).run_one(), std::runtime_error);
    
    // Multiple fetches
    CHECK_THROWS_AS(
        facade::Runner(session).fetch("c1:0").fetch("c2:0").run_one(),
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
    
    auto result = facade::Runner(session)
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

TEST_CASE("facade::Scalar creates typed constants") {
    Graph graph;
    
    auto sf = facade::Scalar<float>(graph, "sf", 3.14f);
    CHECK(TF_OperationOutputType(sf.output(0)) == TF_FLOAT);
    
    auto sd = facade::Scalar<double>(graph, "sd", 2.718);
    CHECK(TF_OperationOutputType(sd.output(0)) == TF_DOUBLE);
    
    auto si = facade::Scalar<std::int32_t>(graph, "si", 42);
    CHECK(TF_OperationOutputType(si.output(0)) == TF_INT32);
}

TEST_CASE("facade::Placeholder creates typed placeholders") {
    Graph graph;
    
    auto ph_f = facade::Placeholder<float>(graph, "ph_f");
    CHECK(TF_OperationOutputType(ph_f.output(0)) == TF_FLOAT);
    
    auto ph_d = facade::Placeholder<double>(graph, "ph_d");
    CHECK(TF_OperationOutputType(ph_d.output(0)) == TF_DOUBLE);
    
    auto ph_i = facade::Placeholder<std::int32_t>(graph, "ph_i");
    CHECK(TF_OperationOutputType(ph_i.output(0)) == TF_INT32);
}

TEST_CASE("facade::Identity creates typed identity") {
    Graph graph;
    auto c = facade::Scalar<float>(graph, "c", 1.0f);
    auto id = facade::Identity<float>(graph, "id", c.output(0));
    CHECK(TF_OperationOutputType(id.output(0)) == TF_FLOAT);
}

} // TEST_SUITE

// ============================================================================
// Integration
// ============================================================================

TEST_SUITE("Integration") {

TEST_CASE("Full pipeline: placeholder -> identity -> run") {
    Graph graph;
    auto ph = facade::Placeholder<float>(graph, "input");
    facade::Identity<float>(graph, "output", ph.output(0));
    
    Session session(graph);
    auto input = Tensor::FromScalar<float>(99.0f);
    
    auto result = facade::Runner(session)
        .feed("input:0", input)
        .fetch("output:0")
        .run_one();
    
    CHECK(result.ToScalar<float>() == doctest::Approx(99.0f));
}

TEST_CASE("TensorName resolution in Runner") {
    Graph graph;
    auto c = facade::Scalar<float>(graph, "my_const", 42.0f);
    
    Session session(graph);
    
    // Using string
    auto r1 = facade::Runner(session).fetch("my_const:0").run_one();
    CHECK(r1.ToScalar<float>() == doctest::Approx(42.0f));
    
    // Using Endpoint with TF_Output directly
    auto r2 = facade::Runner(session).fetch(c.output(0)).run_one();
    CHECK(r2.ToScalar<float>() == doctest::Approx(42.0f));
}

} // TEST_SUITE


// ============================================================================
// P2: Zero-copy reshape + batch inference helper
// ============================================================================

TEST_SUITE("P2 enhancements") {

TEST_CASE("Tensor::reshape shares buffer and changes shape") {
    auto t = Tensor::FromVector<float>({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    auto r = t.reshape({3, 2});

    CHECK(r.shape() == std::vector<std::int64_t>{3, 2});
    CHECK(r.dtype() == TF_FLOAT);
    CHECK(r.num_elements() == 6);

    // Same underlying buffer
    CHECK(r.data<float>() == t.data<float>());

    // Mutate via reshaped view and observe in original tensor
    r.write<float>()[0] = 99.0f;
    CHECK(t.read<float>()[0] == doctest::Approx(99.0f));
}

TEST_CASE("Tensor::reshape throws on mismatched element count") {
    auto t = Tensor::FromVector<float>({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    CHECK_THROWS_AS(t.reshape({2, 2}), std::invalid_argument);
}

TEST_CASE("Tensor::matches_shape works") {
    auto t = Tensor::Allocate<float>({2, 3});
    CHECK(t.matches_shape({2, 3}));
    CHECK_FALSE(t.matches_shape({3, 2}));
    CHECK_FALSE(t.matches_shape({2}));
}

TEST_CASE("Session::BatchRun runs many inputs") {
    Graph graph;
    auto ph = facade::Placeholder<float>(graph, "input");
    facade::Identity<float>(graph, "output", ph.output(0));

    Session session(graph);

    std::vector<Tensor> inputs;
    inputs.push_back(Tensor::FromScalar<float>(1.0f));
    inputs.push_back(Tensor::FromScalar<float>(2.5f));
    inputs.push_back(Tensor::FromScalar<float>(-3.0f));

    auto outputs = session.BatchRun("input", inputs, "output");
    REQUIRE(outputs.size() == inputs.size());

    CHECK(outputs[0].ToScalar<float>() == doctest::Approx(1.0f));
    CHECK(outputs[1].ToScalar<float>() == doctest::Approx(2.5f));
    CHECK(outputs[2].ToScalar<float>() == doctest::Approx(-3.0f));
}

} // TEST_SUITE


// ============================================================================
// P1: Structured errors + handle-based feeds/fetches
// ============================================================================

TEST_SUITE("P1: Structured errors + handle-based feeds/fetches") {

TEST_CASE("Status throws tf_wrap::Error with code/context") {
    Status st;
    st.set(TF_INVALID_ARGUMENT, "bad arg");

    try {
        st.throw_if_error("unit_test");
        FAIL("Expected throw");
    } catch (const tf_wrap::Error& e) {
        CHECK(e.source() == ErrorSource::TensorFlow);
        CHECK(e.code() == TF_INVALID_ARGUMENT);
        CHECK(std::string(e.context()) == "unit_test");
        CHECK(std::string(e.code_name()) == "INVALID_ARGUMENT");
        CHECK(std::string(e.what()).find("bad arg") != std::string::npos);
    }
}

TEST_CASE("Session::resolve_output throws structured wrapper error") {
    Graph graph;
    // Create an empty session/graph; resolve_output should fail cleanly
    Session session(graph);

    try {
        (void)session.resolve_output("does_not_exist", 0);
        FAIL("Expected throw");
    } catch (const tf_wrap::Error& e) {
        CHECK(e.source() == ErrorSource::Wrapper);
        CHECK(e.code() == TF_NOT_FOUND);
        CHECK(std::string(e.op_name()) == "does_not_exist");
        CHECK(e.index() == 0);
    }
}

TEST_CASE("Session::Run accepts TF_Output-based feeds/fetches") {
    Graph graph;

    std::vector<std::int64_t> shape{2};
    auto x = Placeholder(graph, "X", TF_FLOAT, shape);
    auto y = Placeholder(graph, "Y", TF_FLOAT, shape);
    auto sum = Add(graph, "Sum", x.output(0), y.output(0), TF_FLOAT);
    (void)sum;

    Session session(graph);

    auto tx = Tensor::FromVector<float>({2}, std::vector<float>{1.0f, 2.0f});
    auto ty = Tensor::FromVector<float>({2}, std::vector<float>{3.0f, 4.0f});

    TF_Output x_out = session.resolve_output("X", 0);
    TF_Output y_out = session.resolve_output("Y", 0);
    TF_Output sum_out = session.resolve_output("Sum", 0);

    auto results = session.Run(
        {Feed{x_out, tx}, Feed{y_out, ty}},
        {Fetch{sum_out}},
        {} // targets
    );

    REQUIRE(results.size() == 1);

    auto v = results[0].ToVector<float>();
    REQUIRE(v.size() == 2);
    CHECK(v[0] == doctest::Approx(4.0f));
    CHECK(v[1] == doctest::Approx(6.0f));
}

TEST_CASE("Handle-based fetch bounds checking throws") {
    Graph graph;

    std::vector<std::int64_t> shape{1};
    auto x = Placeholder(graph, "X", TF_FLOAT, shape);
    auto id = Identity(graph, "Id", x.output(0), TF_FLOAT);
    (void)id;

    Session session(graph);

    TF_Output id0 = session.resolve_output("Id", 0);
    TF_Output bad = TF_Output{id0.oper, 1}; // invalid index

    try {
        (void)session.Run(
            {Feed{"X", Tensor::FromVector<float>({1}, std::vector<float>{1.0f})}},
            {Fetch{bad}},
            {}
        );
        FAIL("Expected throw");
    } catch (const tf_wrap::Error& e) {
        CHECK(e.source() == ErrorSource::Wrapper);
        CHECK(e.code() == TF_OUT_OF_RANGE);
        CHECK(e.index() == 1);
    }
}



TEST_CASE("Feed constructed from temporary Tensor keeps tensor alive") {
    struct Flag {
        bool freed{false};
    } flag;

    auto dealloc = [](void* data, std::size_t, void* arg) noexcept {
        static_cast<Flag*>(arg)->freed = true;
        std::free(data);
    };

    void* data = std::malloc(sizeof(float));
    REQUIRE(data != nullptr);
    *static_cast<float*>(data) = 1.0f;

    {
        Feed f("X:0", Tensor::Adopt(TF_FLOAT, std::vector<std::int64_t>{1}, data, sizeof(float), dealloc, &flag));
        CHECK_FALSE(flag.freed);
        (void)f;
    }

    CHECK(flag.freed);
}

TEST_CASE("Runner::feed constructed from temporary Tensor keeps tensor alive") {
    Graph graph;
    std::vector<std::int64_t> shape = {};
    auto ph = Placeholder(graph, "input", TF_FLOAT, shape);
    Identity(graph, "output", ph.output(0), TF_FLOAT);

    Session session(graph);

    struct Flag {
        bool freed{false};
    } flag;

    auto dealloc = [](void* data, std::size_t, void* arg) noexcept {
        static_cast<Flag*>(arg)->freed = true;
        std::free(data);
    };

    void* data = std::malloc(sizeof(float));
    REQUIRE(data != nullptr);
    *static_cast<float*>(data) = 2.0f;

    {
        facade::Runner r(session);
        r.feed("input:0", Tensor::Adopt(TF_FLOAT, std::vector<std::int64_t>{}, data, sizeof(float), dealloc, &flag));
        CHECK_FALSE(flag.freed);
    }

    CHECK(flag.freed);
}
} // TEST_SUITE

