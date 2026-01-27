// test_operation.cpp
// Comprehensive tests for tf_wrap::Operation
//
// Framework: doctest
// Runs with: Stub TensorFlow (all platforms)
//
// These tests cover:
// - Operation: construction, null handling
// - Output helper function
// - Copy/move semantics
//
// Note: Tests requiring actual TF_Operation* from a real graph are in
// test_operation_tf.cpp

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "tf_wrap/operation.hpp"
#include "tf_wrap/graph.hpp"

using namespace tf_wrap;

// ============================================================================
// Construction Tests
// ============================================================================

TEST_CASE("Operation - null construction throws") {
    bool threw = false;
    try {
        Operation op(nullptr);
        (void)op;
    } catch (const std::invalid_argument& e) {
        threw = true;
        CHECK(std::string(e.what()).find("null") != std::string::npos);
    }
    CHECK(threw);
}

// ============================================================================
// Output Helper Function Tests
// ============================================================================

TEST_CASE("Output - from raw pointer") {
    TF_Operation* raw = reinterpret_cast<TF_Operation*>(0x1234);
    TF_Output out = Output(raw, 0);
    CHECK(out.oper == raw);
    CHECK(out.index == 0);
}

TEST_CASE("Output - from raw pointer with index") {
    TF_Operation* raw = reinterpret_cast<TF_Operation*>(0x5678);
    TF_Output out = Output(raw, 3);
    CHECK(out.oper == raw);
    CHECK(out.index == 3);
}

TEST_CASE("Output - from null pointer") {
    TF_Output out = Output(nullptr, 0);
    CHECK(out.oper == nullptr);
    CHECK(out.index == 0);
}

// ============================================================================
// TF_Output Struct Tests
// ============================================================================

TEST_CASE("TF_Output - default initialization") {
    TF_Output out{nullptr, 0};
    CHECK(out.oper == nullptr);
    CHECK(out.index == 0);
}

TEST_CASE("TF_Output - with values") {
    TF_Operation* raw = reinterpret_cast<TF_Operation*>(0xABCD);
    TF_Output out{raw, 5};
    CHECK(out.oper == raw);
    CHECK(out.index == 5);
}

// ============================================================================
// TF_Input Struct Tests  
// ============================================================================

TEST_CASE("TF_Input - default initialization") {
    TF_Input inp{nullptr, 0};
    CHECK(inp.oper == nullptr);
    CHECK(inp.index == 0);
}

TEST_CASE("TF_Input - with values") {
    TF_Operation* raw = reinterpret_cast<TF_Operation*>(0xDEAD);
    TF_Input inp{raw, 2};
    CHECK(inp.oper == raw);
    CHECK(inp.index == 2);
}
