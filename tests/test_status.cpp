// test_status.cpp
// Comprehensive tests for tf_wrap::Status
//
// Framework: doctest
// Runs with: Stub TensorFlow (all platforms)
//
// These tests cover:
// - Status: construction, move semantics
// - Accessors: code, message, ok, code_name
// - Mutation: reset, set
// - Error handling: throw_if_error, ThrowIfNotOK
// - Free functions: throw_if_error, consume_status

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "tf_wrap/status.hpp"

using namespace tf_wrap;

// ============================================================================
// Construction Tests
// ============================================================================

TEST_CASE("Status - default construction is OK") {
    Status st;
    CHECK(st.ok());
    CHECK(st.code() == TF_OK);
}

TEST_CASE("Status - get returns non-null handle") {
    Status st;
    CHECK(st.get() != nullptr);
}

TEST_CASE("Status - handle returns same as get") {
    Status st;
    CHECK(st.handle() == st.get());
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_CASE("Status - move constructor") {
    Status s1;
    auto* handle1 = s1.get();
    
    Status s2(std::move(s1));
    CHECK(s2.get() == handle1);
    CHECK(s1.get() == nullptr);
}

TEST_CASE("Status - move assignment") {
    Status s1;
    Status s2;
    auto* handle1 = s1.get();
    
    s2 = std::move(s1);
    CHECK(s2.get() == handle1);
    CHECK(s1.get() == nullptr);
}

TEST_CASE("Status - self move assignment") {
    Status s1;
    auto* handle1 = s1.get();
    
    // Use a reference to avoid -Werror=self-move
    Status& ref = s1;
    s1 = std::move(ref);
    CHECK(s1.get() == handle1);
}

// ============================================================================
// Accessor Tests
// ============================================================================

TEST_CASE("Status - code returns TF_OK initially") {
    Status st;
    CHECK(st.code() == TF_OK);
}

TEST_CASE("Status - message returns empty string initially") {
    Status st;
    CHECK(std::string(st.message()).empty());
}

TEST_CASE("Status - ok returns true initially") {
    Status st;
    CHECK(st.ok());
}

TEST_CASE("Status - operator bool returns true when ok") {
    Status st;
    CHECK(static_cast<bool>(st));
}

TEST_CASE("Status - operator! returns false when ok") {
    Status st;
    CHECK_FALSE(!st);
}

TEST_CASE("Status - code_name returns OK for TF_OK") {
    Status st;
    CHECK(std::string(st.code_name()) == "OK");
}

// ============================================================================
// Mutation Tests
// ============================================================================

TEST_CASE("Status - set with TF_Code and const char*") {
    Status st;
    st.set(TF_INVALID_ARGUMENT, "test error");
    
    CHECK(st.code() == TF_INVALID_ARGUMENT);
    CHECK(std::string(st.message()) == "test error");
    CHECK_FALSE(st.ok());
}

TEST_CASE("Status - set with TF_Code and std::string") {
    Status st;
    std::string msg = "string error";
    st.set(TF_NOT_FOUND, msg);
    
    CHECK(st.code() == TF_NOT_FOUND);
    CHECK(std::string(st.message()) == "string error");
}

TEST_CASE("Status - set with TF_Code and string_view") {
    Status st;
    std::string_view msg = "view error";
    st.set(TF_ABORTED, msg);
    
    CHECK(st.code() == TF_ABORTED);
    CHECK(std::string(st.message()) == "view error");
}

TEST_CASE("Status - set with empty string_view") {
    Status st;
    st.set(TF_CANCELLED, std::string_view{});
    
    CHECK(st.code() == TF_CANCELLED);
    CHECK(std::string(st.message()).empty());
}

TEST_CASE("Status - set with null message") {
    Status st;
    st.set(TF_UNKNOWN, static_cast<const char*>(nullptr));
    
    CHECK(st.code() == TF_UNKNOWN);
}

TEST_CASE("Status - reset restores OK state") {
    Status st;
    st.set(TF_INTERNAL, "error");
    CHECK_FALSE(st.ok());
    
    st.reset();
    CHECK(st.ok());
    CHECK(st.code() == TF_OK);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_CASE("Status - throw_if_error does nothing when OK") {
    Status st;
    st.throw_if_error("context");
    // Should not throw
    CHECK(true);
}

TEST_CASE("Status - throw_if_error throws when not OK") {
    Status st;
    st.set(TF_INVALID_ARGUMENT, "bad input");
    
    bool threw = false;
    try {
        st.throw_if_error("test context");
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_INVALID_ARGUMENT);
    }
    CHECK(threw);
}

TEST_CASE("Status - throw_if_error includes context in message") {
    Status st;
    st.set(TF_NOT_FOUND, "resource missing");
    
    bool threw = false;
    try {
        st.throw_if_error("MyOperation");
    } catch (const Error& e) {
        threw = true;
        std::string what = e.what();
        CHECK(what.find("MyOperation") != std::string::npos);
    }
    CHECK(threw);
}

TEST_CASE("Status - ThrowIfNotOK is alias for throw_if_error") {
    Status st;
    st.set(TF_ABORTED, "aborted");
    
    bool threw = false;
    try {
        st.ThrowIfNotOK("context");
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_ABORTED);
    }
    CHECK(threw);
}

// ============================================================================
// Operator Tests
// ============================================================================

TEST_CASE("Status - operator bool false when error") {
    Status st;
    st.set(TF_INTERNAL, "error");
    CHECK_FALSE(static_cast<bool>(st));
}

TEST_CASE("Status - operator! true when error") {
    Status st;
    st.set(TF_INTERNAL, "error");
    CHECK(!st);
}

// ============================================================================
// Code Name Tests
// ============================================================================

TEST_CASE("Status - code_name for various codes") {
    Status st;
    
    st.set(TF_OK, "");
    CHECK(std::string(st.code_name()) == "OK");
    
    st.set(TF_CANCELLED, "");
    CHECK(std::string(st.code_name()) == "CANCELLED");
    
    st.set(TF_INVALID_ARGUMENT, "");
    CHECK(std::string(st.code_name()) == "INVALID_ARGUMENT");
    
    st.set(TF_NOT_FOUND, "");
    CHECK(std::string(st.code_name()) == "NOT_FOUND");
    
    st.set(TF_INTERNAL, "");
    CHECK(std::string(st.code_name()) == "INTERNAL");
}

TEST_CASE("Status - code_to_string static function") {
    CHECK(std::string(Status::code_to_string(TF_OK)) == "OK");
    CHECK(std::string(Status::code_to_string(TF_UNKNOWN)) == "UNKNOWN");
    CHECK(std::string(Status::code_to_string(TF_FAILED_PRECONDITION)) == "FAILED_PRECONDITION");
}

// ============================================================================
// Const Accessor Tests
// ============================================================================

TEST_CASE("Status - const get returns same handle") {
    Status st;
    const Status& cst = st;
    CHECK(cst.get() == st.get());
}

TEST_CASE("Status - const handle returns same as const get") {
    const Status st;
    CHECK(st.handle() == st.get());
}

// ============================================================================
// Free Function Tests
// ============================================================================

TEST_CASE("throw_if_error free function - OK status") {
    Status st;
    throw_if_error(st, "context");
    CHECK(true);  // Should not throw
}

TEST_CASE("throw_if_error free function - error status") {
    Status st;
    st.set(TF_UNAVAILABLE, "service down");
    
    bool threw = false;
    try {
        throw_if_error(st, "service call");
    } catch (const Error& e) {
        threw = true;
        CHECK(e.code() == TF_UNAVAILABLE);
    }
    CHECK(threw);
}

// Note: consume_status is harder to test because it takes ownership
// of a raw TF_Status* - tested indirectly through other components
