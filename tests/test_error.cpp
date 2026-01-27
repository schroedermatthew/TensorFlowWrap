// test_error.cpp
// Comprehensive tests for tf_wrap::Error
//
// Framework: doctest
// Runs with: Stub TensorFlow (all platforms)
//
// These tests cover:
// - Error: construction, accessors
// - Factory methods: TensorFlow, Wrapper
// - Inheritance: std::runtime_error compatibility
// - what() message formatting

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "tf_wrap/error.hpp"

using namespace tf_wrap;

// ============================================================================
// Construction Tests
// ============================================================================

TEST_CASE("Error - full constructor") {
    Error err(
        ErrorSource::Wrapper,
        TF_INVALID_ARGUMENT,
        "TestContext",
        "test message",
        "op_name",
        42
    );
    
    CHECK(err.source() == ErrorSource::Wrapper);
    CHECK(err.code() == TF_INVALID_ARGUMENT);
    CHECK(err.context() == "TestContext");
    CHECK(err.op_name() == "op_name");
    CHECK(err.index() == 42);
}

TEST_CASE("Error - constructor with defaults") {
    Error err(
        ErrorSource::TensorFlow,
        TF_NOT_FOUND,
        "Context",
        "message"
    );
    
    CHECK(err.source() == ErrorSource::TensorFlow);
    CHECK(err.code() == TF_NOT_FOUND);
    CHECK(err.op_name().empty());
    CHECK(err.index() == -1);
}

// ============================================================================
// Factory Method Tests
// ============================================================================

TEST_CASE("Error::TensorFlow factory") {
    auto err = Error::TensorFlow(TF_INTERNAL, "TF_Context", "TF error message");
    
    CHECK(err.source() == ErrorSource::TensorFlow);
    CHECK(err.code() == TF_INTERNAL);
    CHECK(err.context() == "TF_Context");
    CHECK(err.op_name().empty());
    CHECK(err.index() == -1);
}

TEST_CASE("Error::Wrapper factory - minimal") {
    auto err = Error::Wrapper(TF_ABORTED, "WrapContext", "wrapper error");
    
    CHECK(err.source() == ErrorSource::Wrapper);
    CHECK(err.code() == TF_ABORTED);
    CHECK(err.context() == "WrapContext");
    CHECK(err.op_name().empty());
    CHECK(err.index() == -1);
}

TEST_CASE("Error::Wrapper factory - with op_name") {
    auto err = Error::Wrapper(TF_CANCELLED, "Context", "message", "my_op");
    
    CHECK(err.op_name() == "my_op");
    CHECK(err.index() == -1);
}

TEST_CASE("Error::Wrapper factory - with op_name and index") {
    auto err = Error::Wrapper(TF_UNKNOWN, "Context", "message", "op", 5);
    
    CHECK(err.op_name() == "op");
    CHECK(err.index() == 5);
}

// ============================================================================
// Accessor Tests
// ============================================================================

TEST_CASE("Error - code accessor") {
    auto err = Error::Wrapper(TF_RESOURCE_EXHAUSTED, "ctx", "msg");
    CHECK(err.code() == TF_RESOURCE_EXHAUSTED);
}

TEST_CASE("Error - code_name accessor") {
    auto err = Error::Wrapper(TF_DEADLINE_EXCEEDED, "ctx", "msg");
    CHECK(std::string(err.code_name()) == "DEADLINE_EXCEEDED");
}

TEST_CASE("Error - context accessor") {
    auto err = Error::Wrapper(TF_OK, "MyContext", "msg");
    CHECK(err.context() == "MyContext");
}

TEST_CASE("Error - op_name accessor") {
    auto err = Error::Wrapper(TF_OK, "ctx", "msg", "operation_name");
    CHECK(err.op_name() == "operation_name");
}

TEST_CASE("Error - index accessor") {
    auto err = Error::Wrapper(TF_OK, "ctx", "msg", "op", 123);
    CHECK(err.index() == 123);
}

TEST_CASE("Error - location accessor") {
    auto err = Error::Wrapper(TF_OK, "ctx", "msg");
    auto loc = err.location();
    
    // Location should be populated
    CHECK(loc.line() > 0);
    CHECK(loc.file_name() != nullptr);
    CHECK(loc.function_name() != nullptr);
}

// ============================================================================
// what() Message Tests
// ============================================================================

TEST_CASE("Error - what contains code name") {
    auto err = Error::Wrapper(TF_INVALID_ARGUMENT, "ctx", "message");
    std::string what = err.what();
    
    CHECK(what.find("INVALID_ARGUMENT") != std::string::npos);
}

TEST_CASE("Error - what contains context") {
    auto err = Error::Wrapper(TF_OK, "ImportantContext", "message");
    std::string what = err.what();
    
    CHECK(what.find("ImportantContext") != std::string::npos);
}

TEST_CASE("Error - what contains message") {
    auto err = Error::Wrapper(TF_OK, "ctx", "the actual error message");
    std::string what = err.what();
    
    CHECK(what.find("the actual error message") != std::string::npos);
}

TEST_CASE("Error - what contains op_name when present") {
    auto err = Error::Wrapper(TF_OK, "ctx", "msg", "my_operation");
    std::string what = err.what();
    
    CHECK(what.find("my_operation") != std::string::npos);
}

TEST_CASE("Error - what contains index when present") {
    auto err = Error::Wrapper(TF_OK, "ctx", "msg", "op", 7);
    std::string what = err.what();
    
    CHECK(what.find("7") != std::string::npos);
}

TEST_CASE("Error - what contains file location") {
    auto err = Error::Wrapper(TF_OK, "ctx", "msg");
    std::string what = err.what();
    
    // Should contain filename (test_error.cpp or similar)
    CHECK(what.find(".cpp") != std::string::npos);
}

TEST_CASE("Error - TensorFlow source prefix") {
    auto err = Error::TensorFlow(TF_OK, "ctx", "msg");
    std::string what = err.what();
    
    CHECK(what.find("TF_") != std::string::npos);
}

TEST_CASE("Error - Wrapper source prefix") {
    auto err = Error::Wrapper(TF_OK, "ctx", "msg");
    std::string what = err.what();
    
    CHECK(what.find("WRAP_") != std::string::npos);
}

// ============================================================================
// Inheritance Tests
// ============================================================================

TEST_CASE("Error - inherits from std::runtime_error") {
    auto err = Error::Wrapper(TF_INTERNAL, "ctx", "runtime error test");
    
    // Should be catchable as std::runtime_error
    bool caught_as_runtime = false;
    try {
        throw err;
    } catch (const std::runtime_error& e) {
        caught_as_runtime = true;
        CHECK(std::string(e.what()).find("runtime error test") != std::string::npos);
    }
    CHECK(caught_as_runtime);
}

TEST_CASE("Error - inherits from std::exception") {
    auto err = Error::Wrapper(TF_INTERNAL, "ctx", "exception test");
    
    // Should be catchable as std::exception
    bool caught_as_exception = false;
    try {
        throw err;
    } catch (const std::exception& e) {
        caught_as_exception = true;
        CHECK(std::string(e.what()).find("exception test") != std::string::npos);
    }
    CHECK(caught_as_exception);
}

TEST_CASE("Error - catch as Error preserves code") {
    try {
        throw Error::Wrapper(TF_PERMISSION_DENIED, "ctx", "denied");
    } catch (const Error& e) {
        CHECK(e.code() == TF_PERMISSION_DENIED);
    }
}

// ============================================================================
// code_to_string Static Function Tests
// ============================================================================

TEST_CASE("Error::code_to_string - all standard codes") {
    CHECK(std::string(Error::code_to_string(TF_OK)) == "OK");
    CHECK(std::string(Error::code_to_string(TF_CANCELLED)) == "CANCELLED");
    CHECK(std::string(Error::code_to_string(TF_UNKNOWN)) == "UNKNOWN");
    CHECK(std::string(Error::code_to_string(TF_INVALID_ARGUMENT)) == "INVALID_ARGUMENT");
    CHECK(std::string(Error::code_to_string(TF_DEADLINE_EXCEEDED)) == "DEADLINE_EXCEEDED");
    CHECK(std::string(Error::code_to_string(TF_NOT_FOUND)) == "NOT_FOUND");
    CHECK(std::string(Error::code_to_string(TF_ALREADY_EXISTS)) == "ALREADY_EXISTS");
    CHECK(std::string(Error::code_to_string(TF_PERMISSION_DENIED)) == "PERMISSION_DENIED");
    CHECK(std::string(Error::code_to_string(TF_RESOURCE_EXHAUSTED)) == "RESOURCE_EXHAUSTED");
    CHECK(std::string(Error::code_to_string(TF_FAILED_PRECONDITION)) == "FAILED_PRECONDITION");
    CHECK(std::string(Error::code_to_string(TF_ABORTED)) == "ABORTED");
    CHECK(std::string(Error::code_to_string(TF_OUT_OF_RANGE)) == "OUT_OF_RANGE");
    CHECK(std::string(Error::code_to_string(TF_UNIMPLEMENTED)) == "UNIMPLEMENTED");
    CHECK(std::string(Error::code_to_string(TF_INTERNAL)) == "INTERNAL");
    CHECK(std::string(Error::code_to_string(TF_UNAVAILABLE)) == "UNAVAILABLE");
    CHECK(std::string(Error::code_to_string(TF_DATA_LOSS)) == "DATA_LOSS");
    CHECK(std::string(Error::code_to_string(TF_UNAUTHENTICATED)) == "UNAUTHENTICATED");
}

// ============================================================================
// ErrorSource Enum Tests
// ============================================================================

TEST_CASE("ErrorSource - TensorFlow value") {
    CHECK(ErrorSource::TensorFlow != ErrorSource::Wrapper);
}

TEST_CASE("ErrorSource - Wrapper value") {
    CHECK(ErrorSource::Wrapper != ErrorSource::TensorFlow);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_CASE("Error - empty context") {
    auto err = Error::Wrapper(TF_OK, "", "message");
    CHECK(err.context().empty());
    // what() should still be valid
    CHECK(std::string(err.what()).length() > 0);
}

TEST_CASE("Error - empty message") {
    auto err = Error::Wrapper(TF_OK, "ctx", "");
    std::string what = err.what();
    CHECK(what.length() > 0);
}

TEST_CASE("Error - empty op_name with index") {
    auto err = Error::Wrapper(TF_OK, "ctx", "msg", "", 5);
    CHECK(err.op_name().empty());
    CHECK(err.index() == 5);
}

TEST_CASE("Error - negative index") {
    auto err = Error::Wrapper(TF_OK, "ctx", "msg", "op", -1);
    CHECK(err.index() == -1);
}
