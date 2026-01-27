// test_graph_tf.cpp
// Graph tests with real TensorFlow C library
//
// Framework: Custom (see below)
// Runs with: Real TensorFlow (Linux CI only)
//
// These tests cover Graph operations with a populated graph from SavedModel:
// - Operation lookup in real graph
// - Graph introspection with actual operations
// - ToGraphDef serialization
// - DebugString with real operations

#include "tf_wrap/graph.hpp"
#include "tf_wrap/session.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace tf_wrap;

// ============================================================================
// Test Framework
// ============================================================================

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    void test_##name(); \
    struct TestRunner_##name { \
        TestRunner_##name() { \
            std::cout << "Testing " #name "... " << std::flush; \
            tests_run++; \
            try { \
                test_##name(); \
                std::cout << "PASSED\n"; \
                tests_passed++; \
            } catch (const std::exception& e) { \
                std::cout << "FAILED: " << e.what() << "\n"; \
            } catch (...) { \
                std::cout << "FAILED: unknown exception\n"; \
            } \
        } \
    } test_runner_##name; \
    void test_##name()

#define REQUIRE(cond) \
    do { if (!(cond)) throw std::runtime_error("REQUIRE failed: " #cond); } while (0)

#define REQUIRE_THROWS(expr) \
    do { \
        bool threw = false; \
        try { (void)(expr); } catch (...) { threw = true; } \
        if (!threw) throw std::runtime_error("REQUIRE_THROWS failed: " #expr); \
    } while (0)

// ============================================================================
// Test Model Path
// ============================================================================

static const char* TEST_MODEL_PATH = "test_savedmodel";

// ============================================================================
// Helper: Load graph from SavedModel
// ============================================================================

static Graph load_test_graph() {
    auto [session, graph] = Session::LoadSavedModel(TEST_MODEL_PATH);
    return std::move(graph);
}

// ============================================================================
// Construction Tests
// ============================================================================

TEST(graph_from_savedmodel_valid) {
    auto graph = load_test_graph();
    REQUIRE(graph.valid());
    REQUIRE(graph.handle() != nullptr);
}

TEST(graph_from_savedmodel_is_frozen) {
    auto graph = load_test_graph();
    // Graph from SavedModel should be frozen (session uses it)
    REQUIRE(graph.is_frozen());
}

// ============================================================================
// Operation Lookup Tests
// ============================================================================

TEST(graph_get_operation_found) {
    auto graph = load_test_graph();
    
    // The test model has input "serving_default_x"
    auto op = graph.GetOperation("serving_default_x");
    REQUIRE(op.has_value());
    REQUIRE(*op != nullptr);
}

TEST(graph_get_operation_not_found) {
    auto graph = load_test_graph();
    
    auto op = graph.GetOperation("nonexistent_operation_xyz");
    REQUIRE(!op.has_value());
}

TEST(graph_get_operation_or_throw_found) {
    auto graph = load_test_graph();
    
    auto* op = graph.GetOperationOrThrow("serving_default_x");
    REQUIRE(op != nullptr);
}

TEST(graph_get_operation_or_throw_not_found) {
    auto graph = load_test_graph();
    
    REQUIRE_THROWS(graph.GetOperationOrThrow("nonexistent_operation_xyz"));
}

TEST(graph_has_operation_true) {
    auto graph = load_test_graph();
    
    REQUIRE(graph.HasOperation("serving_default_x"));
}

TEST(graph_has_operation_false) {
    auto graph = load_test_graph();
    
    REQUIRE(!graph.HasOperation("nonexistent_operation_xyz"));
}

// ============================================================================
// Introspection Tests
// ============================================================================

TEST(graph_get_all_operations_nonempty) {
    auto graph = load_test_graph();
    
    auto ops = graph.GetAllOperations();
    REQUIRE(!ops.empty());
    
    // All returned pointers should be valid
    for (auto* op : ops) {
        REQUIRE(op != nullptr);
    }
}

TEST(graph_num_operations_positive) {
    auto graph = load_test_graph();
    
    REQUIRE(graph.num_operations() > 0);
}

TEST(graph_num_operations_matches_get_all) {
    auto graph = load_test_graph();
    
    auto ops = graph.GetAllOperations();
    REQUIRE(ops.size() == graph.num_operations());
}

TEST(graph_get_operations_by_type_placeholder) {
    auto graph = load_test_graph();
    
    auto placeholders = graph.GetOperationsByType("Placeholder");
    // Test model should have at least one placeholder (the input)
    REQUIRE(!placeholders.empty());
}

TEST(graph_get_placeholders) {
    auto graph = load_test_graph();
    
    auto placeholders = graph.GetPlaceholders();
    REQUIRE(!placeholders.empty());
    
    // Check that info is populated
    for (const auto& info : placeholders) {
        REQUIRE(!info.op_name.empty());
        REQUIRE(info.op_type == "Placeholder");
        REQUIRE(info.num_outputs >= 1);
    }
}

TEST(graph_find_input_placeholder) {
    auto graph = load_test_graph();
    
    auto placeholders = graph.GetPlaceholders();
    
    // Find serving_default_x among placeholders
    bool found = false;
    for (const auto& info : placeholders) {
        if (info.op_name == "serving_default_x") {
            found = true;
            break;
        }
    }
    REQUIRE(found);
}

// ============================================================================
// Serialization Tests
// ============================================================================

TEST(graph_to_graphdef_nonempty) {
    auto graph = load_test_graph();
    
    auto def = graph.ToGraphDef();
    REQUIRE(!def.empty());
    // GraphDef should be at least a few hundred bytes for our test model
    REQUIRE(def.size() > 100);
}

TEST(graph_debug_string_contains_operations) {
    auto graph = load_test_graph();
    
    auto debug = graph.DebugString();
    REQUIRE(!debug.empty());
    
    // Should contain operation count
    REQUIRE(debug.find("operations") != std::string::npos);
    
    // Should contain at least one operation name
    REQUIRE(debug.find("serving_default_x") != std::string::npos);
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST(graph_move_preserves_operations) {
    auto g1 = load_test_graph();
    auto count1 = g1.num_operations();
    
    Graph g2(std::move(g1));
    
    REQUIRE(g2.valid());
    REQUIRE(g2.num_operations() == count1);
    REQUIRE(!g1.valid());
}

TEST(graph_move_assignment_preserves_operations) {
    auto g1 = load_test_graph();
    auto count1 = g1.num_operations();
    
    Graph g2;
    g2 = std::move(g1);
    
    REQUIRE(g2.valid());
    REQUIRE(g2.num_operations() == count1);
    REQUIRE(!g1.valid());
}

// ============================================================================
// State Sharing Tests
// ============================================================================

TEST(graph_share_state_preserves_freeze) {
    auto [session, graph] = Session::LoadSavedModel(TEST_MODEL_PATH);
    
    auto state = graph.share_state();
    
    REQUIRE(state != nullptr);
    REQUIRE(state->frozen == graph.is_frozen());
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== TensorFlowWrap Graph Tests (Real TF) ===\n\n";
    
    // Tests run automatically via static initialization
    
    std::cout << "\n=== Results: " << tests_passed << "/" << tests_run << " passed ===\n";
    
    return (tests_passed == tests_run) ? 0 : 1;
}
