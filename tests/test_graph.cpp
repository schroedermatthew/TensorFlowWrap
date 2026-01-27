// test_graph.cpp
// Comprehensive tests for tf_wrap::Graph
//
// Framework: doctest
// Runs with: Stub TensorFlow (all platforms)
//
// These tests cover:
// - Graph: construction, move semantics, validity
// - Operation lookup: GetOperation, GetOperationOrThrow, HasOperation
// - Introspection: GetAllOperations, num_operations (empty graph)
// - Freeze state: freeze, is_frozen
// - State sharing: share_state
//
// Note: Tests with populated graphs (from SavedModel) are in test_graph_tf.cpp

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "tf_wrap/graph.hpp"

using namespace tf_wrap;

// ============================================================================
// Construction Tests
// ============================================================================

TEST_CASE("Graph - default construction") {
    Graph graph;
    CHECK(graph.valid());
    CHECK(graph.handle() != nullptr);
}

TEST_CASE("Graph - move constructor") {
    Graph g1;
    auto* handle1 = g1.handle();
    CHECK(handle1 != nullptr);
    
    Graph g2(std::move(g1));
    CHECK(g2.handle() == handle1);
    CHECK(g2.valid());
    CHECK_FALSE(g1.valid());
    CHECK(g1.handle() == nullptr);
}

TEST_CASE("Graph - move assignment") {
    Graph g1;
    Graph g2;
    auto* handle1 = g1.handle();
    
    g2 = std::move(g1);
    CHECK(g2.handle() == handle1);
    CHECK(g2.valid());
    CHECK_FALSE(g1.valid());
}

TEST_CASE("Graph - self move assignment") {
    Graph g1;
    auto* handle1 = g1.handle();
    
    // Use a reference to avoid -Werror=self-move
    Graph& ref = g1;
    g1 = std::move(ref);
    CHECK(g1.handle() == handle1);
    CHECK(g1.valid());
}

// ============================================================================
// Operation Lookup Tests (empty graph)
// ============================================================================

TEST_CASE("Graph - GetOperation not found returns nullopt") {
    Graph graph;
    auto op = graph.GetOperation("nonexistent");
    CHECK_FALSE(op.has_value());
}

TEST_CASE("Graph - GetOperationOrThrow throws on not found") {
    Graph graph;
    
    bool threw = false;
    try {
        auto op = graph.GetOperationOrThrow("nonexistent");
        (void)op;
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_NOT_FOUND);
    }
    CHECK(threw);
}

TEST_CASE("Graph - HasOperation returns false for nonexistent") {
    Graph graph;
    CHECK_FALSE(graph.HasOperation("nonexistent"));
}

// ============================================================================
// Introspection Tests (empty graph)
// ============================================================================

TEST_CASE("Graph - GetAllOperations empty") {
    Graph graph;
    auto ops = graph.GetAllOperations();
    CHECK(ops.empty());
}

TEST_CASE("Graph - num_operations empty") {
    Graph graph;
    CHECK(graph.num_operations() == 0);
}

TEST_CASE("Graph - GetOperationsByType empty") {
    Graph graph;
    auto ops = graph.GetOperationsByType("Placeholder");
    CHECK(ops.empty());
}

TEST_CASE("Graph - GetOperationInfoByType empty") {
    Graph graph;
    auto infos = graph.GetOperationInfoByType("Placeholder");
    CHECK(infos.empty());
}

TEST_CASE("Graph - GetPlaceholders empty") {
    Graph graph;
    auto placeholders = graph.GetPlaceholders();
    CHECK(placeholders.empty());
}

// ============================================================================
// Serialization Tests (empty graph)
// ============================================================================

TEST_CASE("Graph - ToGraphDef empty graph") {
    Graph graph;
    auto def = graph.ToGraphDef();
    // Empty graph still produces a valid (but small) GraphDef
    // The exact size depends on TF version, just check it doesn't throw
    CHECK(def.size() >= 0);
}

TEST_CASE("Graph - DebugString empty graph") {
    Graph graph;
    auto debug = graph.DebugString();
    CHECK_FALSE(debug.empty());
    CHECK(debug.find("0 operations") != std::string::npos);
}

// ============================================================================
// Freeze State Tests
// ============================================================================

TEST_CASE("Graph - initial state is not frozen") {
    Graph graph;
    CHECK_FALSE(graph.is_frozen());
}

TEST_CASE("Graph - freeze sets frozen state") {
    Graph graph;
    graph.freeze();
    CHECK(graph.is_frozen());
}

TEST_CASE("Graph - freeze is idempotent") {
    Graph graph;
    graph.freeze();
    graph.freeze();
    CHECK(graph.is_frozen());
}

// ============================================================================
// State Sharing Tests
// ============================================================================

TEST_CASE("Graph - share_state returns valid shared_ptr") {
    Graph graph;
    auto state = graph.share_state();
    CHECK(state != nullptr);
}

TEST_CASE("Graph - share_state references same underlying graph") {
    Graph graph;
    auto state = graph.share_state();
    CHECK(state->graph == graph.handle());
}

TEST_CASE("Graph - shared state keeps graph alive after move") {
    auto state = []{
        Graph graph;
        return graph.share_state();
    }();
    
    // State should still be valid even though Graph was destroyed
    CHECK(state != nullptr);
    CHECK(state->graph != nullptr);
}

TEST_CASE("Graph - freeze state is shared") {
    Graph g1;
    auto state = g1.share_state();
    
    CHECK_FALSE(g1.is_frozen());
    CHECK_FALSE(state->frozen);
    
    g1.freeze();
    
    CHECK(g1.is_frozen());
    CHECK(state->frozen);
}

// ============================================================================
// Error Cases (moved-from graph)
// ============================================================================

TEST_CASE("Graph - GetOperation on moved-from throws") {
    Graph g1;
    Graph g2(std::move(g1));
    
    bool threw = false;
    try {
        auto op = g1.GetOperation("x");
        (void)op;
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_FAILED_PRECONDITION);
    }
    CHECK(threw);
}

TEST_CASE("Graph - GetOperationOrThrow on moved-from throws") {
    Graph g1;
    Graph g2(std::move(g1));
    
    bool threw = false;
    try {
        auto op = g1.GetOperationOrThrow("x");
        (void)op;
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_FAILED_PRECONDITION);
    }
    CHECK(threw);
}

TEST_CASE("Graph - GetAllOperations on moved-from throws") {
    Graph g1;
    Graph g2(std::move(g1));
    
    bool threw = false;
    try {
        auto ops = g1.GetAllOperations();
        (void)ops;
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_FAILED_PRECONDITION);
    }
    CHECK(threw);
}

TEST_CASE("Graph - num_operations on moved-from throws") {
    Graph g1;
    Graph g2(std::move(g1));
    
    bool threw = false;
    try {
        auto n = g1.num_operations();
        (void)n;
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_FAILED_PRECONDITION);
    }
    CHECK(threw);
}

TEST_CASE("Graph - ToGraphDef on moved-from throws") {
    Graph g1;
    Graph g2(std::move(g1));
    
    bool threw = false;
    try {
        auto def = g1.ToGraphDef();
        (void)def;
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_FAILED_PRECONDITION);
    }
    CHECK(threw);
}

TEST_CASE("Graph - DebugString on moved-from throws") {
    Graph g1;
    Graph g2(std::move(g1));
    
    bool threw = false;
    try {
        auto s = g1.DebugString();
        (void)s;
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_FAILED_PRECONDITION);
    }
    CHECK(threw);
}

// ============================================================================
// OperationInfo Struct Tests
// ============================================================================

TEST_CASE("OperationInfo - initialized values") {
    OperationInfo info{"my_op", "MatMul", 2, 1};
    CHECK(info.op_name == "my_op");
    CHECK(info.op_type == "MatMul");
    CHECK(info.num_inputs == 2);
    CHECK(info.num_outputs == 1);
}

TEST_CASE("OperationInfo - empty strings") {
    OperationInfo info{"", "", 0, 0};
    CHECK(info.op_name.empty());
    CHECK(info.op_type.empty());
    CHECK(info.num_inputs == 0);
    CHECK(info.num_outputs == 0);
}
