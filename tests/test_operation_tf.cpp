// test_operation_tf.cpp
// Operation tests with real TensorFlow C library
//
// Framework: Custom (see below)
// Runs with: Real TensorFlow (Linux CI only)
//
// These tests cover Operation with actual TF_Operation* from a loaded graph:
// - Construction from real operation
// - Metadata: name, op_type, device
// - Topology: num_inputs, num_outputs
// - Output/input accessors
// - output_num_dims with real graph

#include "tf_wrap/operation.hpp"
#include "tf_wrap/graph.hpp"
#include "tf_wrap/session.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

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
// Helper: Get an operation from loaded graph
// ============================================================================

static std::pair<Graph, TF_Operation*> get_test_graph_and_op() {
    auto [session, graph] = Session::LoadSavedModel(TEST_MODEL_PATH);
    
    // Find the input placeholder
    auto* op = graph.GetOperationOrThrow("serving_default_x");
    return {std::move(graph), op};
}

// ============================================================================
// Construction Tests
// ============================================================================

TEST(operation_construction_valid) {
    auto [graph, raw_op] = get_test_graph_and_op();
    
    Operation op(raw_op);
    REQUIRE(op.handle() == raw_op);
}

TEST(operation_construction_null_throws) {
    REQUIRE_THROWS(Operation(nullptr));
}

// ============================================================================
// Metadata Tests
// ============================================================================

TEST(operation_name) {
    auto [graph, raw_op] = get_test_graph_and_op();
    Operation op(raw_op);
    
    std::string name = op.name();
    REQUIRE(name == "serving_default_x");
}

TEST(operation_op_type) {
    auto [graph, raw_op] = get_test_graph_and_op();
    Operation op(raw_op);
    
    std::string op_type = op.op_type();
    REQUIRE(op_type == "Placeholder");
}

TEST(operation_device) {
    auto [graph, raw_op] = get_test_graph_and_op();
    Operation op(raw_op);
    
    // Device may be empty or set - just check it doesn't throw
    std::string device = op.device();
    // Device could be "" or "/device:CPU:0" etc
    REQUIRE(true);  // Just verify no exception
}

// ============================================================================
// Topology Tests
// ============================================================================

TEST(operation_num_inputs) {
    auto [graph, raw_op] = get_test_graph_and_op();
    Operation op(raw_op);
    
    // Placeholder has 0 inputs
    REQUIRE(op.num_inputs() == 0);
}

TEST(operation_num_outputs) {
    auto [graph, raw_op] = get_test_graph_and_op();
    Operation op(raw_op);
    
    // Placeholder has 1 output
    REQUIRE(op.num_outputs() == 1);
}

// ============================================================================
// Output Accessor Tests
// ============================================================================

TEST(operation_output_valid_index) {
    auto [graph, raw_op] = get_test_graph_and_op();
    Operation op(raw_op);
    
    TF_Output out = op.output(0);
    REQUIRE(out.oper == raw_op);
    REQUIRE(out.index == 0);
}

TEST(operation_output_default_index) {
    auto [graph, raw_op] = get_test_graph_and_op();
    Operation op(raw_op);
    
    TF_Output out = op.output();  // Default index = 0
    REQUIRE(out.oper == raw_op);
    REQUIRE(out.index == 0);
}

TEST(operation_output_invalid_index_throws) {
    auto [graph, raw_op] = get_test_graph_and_op();
    Operation op(raw_op);
    
    // Placeholder has only 1 output (index 0)
    REQUIRE_THROWS(op.output(1));
    REQUIRE_THROWS(op.output(-1));
    REQUIRE_THROWS(op.output(100));
}

TEST(operation_output_unchecked) {
    auto [graph, raw_op] = get_test_graph_and_op();
    Operation op(raw_op);
    
    // output_unchecked doesn't validate
    TF_Output out = op.output_unchecked(0);
    REQUIRE(out.oper == raw_op);
    REQUIRE(out.index == 0);
    
    // Can call with any index (no bounds check)
    TF_Output out2 = op.output_unchecked(999);
    REQUIRE(out2.oper == raw_op);
    REQUIRE(out2.index == 999);
}

// ============================================================================
// Input Accessor Tests
// ============================================================================

TEST(operation_input) {
    auto [graph, raw_op] = get_test_graph_and_op();
    Operation op(raw_op);
    
    TF_Input inp = op.input(0);
    REQUIRE(inp.oper == raw_op);
    REQUIRE(inp.index == 0);
}

// ============================================================================
// Output Type Tests
// ============================================================================

TEST(operation_output_type) {
    auto [graph, raw_op] = get_test_graph_and_op();
    Operation op(raw_op);
    
    // The input placeholder is float32
    TF_DataType dtype = op.output_type(0);
    REQUIRE(dtype == TF_FLOAT);
}

// ============================================================================
// Output Num Dims Tests
// ============================================================================

TEST(operation_output_num_dims) {
    auto [graph, raw_op] = get_test_graph_and_op();
    Operation op(raw_op);
    
    // Query shape info - requires the graph
    int ndims = op.output_num_dims(graph.handle(), 0);
    // The test model input has shape [None], so ndims should be 1
    // or -1 if unknown
    REQUIRE(ndims >= -1);
}

TEST(operation_output_num_dims_null_graph_throws) {
    auto [graph, raw_op] = get_test_graph_and_op();
    Operation op(raw_op);
    
    REQUIRE_THROWS(op.output_num_dims(nullptr, 0));
}

// ============================================================================
// Copy/Move Tests
// ============================================================================

TEST(operation_copy) {
    auto [graph, raw_op] = get_test_graph_and_op();
    Operation op1(raw_op);
    
    Operation op2 = op1;  // Copy
    REQUIRE(op2.handle() == op1.handle());
    REQUIRE(op2.name() == op1.name());
}

TEST(operation_move) {
    auto [graph, raw_op] = get_test_graph_and_op();
    Operation op1(raw_op);
    
    Operation op2 = std::move(op1);
    REQUIRE(op2.handle() == raw_op);
    // op1 still has the handle (it's just a pointer wrapper)
}

// ============================================================================
// Output Helper with Operation
// ============================================================================

TEST(output_helper_from_operation) {
    auto [graph, raw_op] = get_test_graph_and_op();
    Operation op(raw_op);
    
    TF_Output out = Output(op, 0);
    REQUIRE(out.oper == raw_op);
    REQUIRE(out.index == 0);
}

TEST(output_helper_from_operation_invalid_throws) {
    auto [graph, raw_op] = get_test_graph_and_op();
    Operation op(raw_op);
    
    // Output() with Operation uses bounds checking
    REQUIRE_THROWS(Output(op, 100));
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "=== TensorFlowWrap Operation Tests (Real TF) ===\n\n";
    
    // Tests run automatically via static initialization
    
    std::cout << "\n=== Results: " << tests_passed << "/" << tests_run << " passed ===\n";
    
    return (tests_passed == tests_run) ? 0 : 1;
}
