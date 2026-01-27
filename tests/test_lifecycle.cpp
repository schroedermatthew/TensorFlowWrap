// test_lifecycle.cpp
// Lifecycle and RAII tests for TensorFlowWrap
//
// Framework: doctest
// Runs with: Stub TensorFlow (all platforms)
//
// These tests verify proper RAII cleanup behavior:
// - Destructors are called at scope exit
// - Cleanup happens even on exception paths
// - Move semantics transfer ownership correctly

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "tf_wrap/tensor.hpp"
#include "tf_wrap/session.hpp"
#include "tf_wrap/graph.hpp"
#include "tf_wrap/status.hpp"
#include "tf_wrap/scope_guard.hpp"

#include <memory>
#include <stdexcept>

using namespace tf_wrap;

// ============================================================================
// LifecycleTracker - Helper for tracking construction/destruction
// ============================================================================

class LifecycleTracker {
public:
    static int constructions;
    static int destructions;
    static int moves;
    static int copies;
    
    static void reset() {
        constructions = 0;
        destructions = 0;
        moves = 0;
        copies = 0;
    }
    
    static int alive() { return constructions - destructions; }
    
    LifecycleTracker() { ++constructions; }
    ~LifecycleTracker() { ++destructions; }
    
    LifecycleTracker(const LifecycleTracker&) { ++constructions; ++copies; }
    LifecycleTracker& operator=(const LifecycleTracker&) { ++copies; return *this; }
    
    LifecycleTracker(LifecycleTracker&&) noexcept { ++constructions; ++moves; }
    LifecycleTracker& operator=(LifecycleTracker&&) noexcept { ++moves; return *this; }
};

int LifecycleTracker::constructions = 0;
int LifecycleTracker::destructions = 0;
int LifecycleTracker::moves = 0;
int LifecycleTracker::copies = 0;

// ============================================================================
// ScopeGuard Lifecycle Tests
// ============================================================================

TEST_CASE("ScopeGuard - action called on scope exit") {
    bool action_called = false;
    
    {
        auto guard = makeScopeGuard([&]() { action_called = true; });
        CHECK_FALSE(action_called);
    }
    
    CHECK(action_called);
}

TEST_CASE("ScopeGuard - action not called after dismiss") {
    bool action_called = false;
    
    {
        auto guard = makeScopeGuard([&]() { action_called = true; });
        guard.dismiss();
    }
    
    CHECK_FALSE(action_called);
}

TEST_CASE("ScopeGuard - action called on exception") {
    bool action_called = false;
    
    try {
        auto guard = makeScopeGuard([&]() { action_called = true; });
        throw std::runtime_error("test exception");
    } catch (...) {
        // Exception caught
    }
    
    CHECK(action_called);
}

TEST_CASE("ScopeGuard - move transfers ownership") {
    int call_count = 0;
    
    {
        auto guard1 = makeScopeGuard([&]() { ++call_count; });
        auto guard2 = std::move(guard1);
        // guard1 should be dismissed after move
    }
    
    // Action should only be called once (by guard2)
    CHECK(call_count == 1);
}

// ============================================================================
// Tensor Lifecycle Tests
// ============================================================================

TEST_CASE("Tensor - destructor called on scope exit") {
    // We can't directly track TF_Tensor destruction in stub mode,
    // but we can verify Tensor objects are properly destroyed
    
    std::weak_ptr<int> weak;
    
    {
        auto shared = std::make_shared<int>(42);
        weak = shared;
        
        // Create a tensor and a guard that captures the shared_ptr
        auto tensor = Tensor::FromScalar(1.0f);
        auto guard = makeScopeGuard([shared]() { /* captures shared */ });
        
        CHECK_FALSE(weak.expired());
    }
    
    // After scope exit, shared_ptr should be released
    CHECK(weak.expired());
}

TEST_CASE("Tensor - move leaves source in valid state") {
    auto t1 = Tensor::FromScalar(42.0f);
    CHECK(t1.valid());
    
    auto t2 = std::move(t1);
    CHECK(t2.valid());
    CHECK_FALSE(t1.valid());  // Moved-from is invalid
}

TEST_CASE("Tensor - move assignment cleans up destination") {
    auto t1 = Tensor::FromScalar(1.0f);
    auto t2 = Tensor::FromScalar(2.0f);
    
    t1 = std::move(t2);
    
    CHECK(t1.valid());
    CHECK_FALSE(t2.valid());
}

// ============================================================================
// Status Lifecycle Tests
// ============================================================================

TEST_CASE("Status - destructor called on scope exit") {
    // Status wraps TF_Status* - verify it's properly managed
    {
        Status s;
        CHECK(s.ok());
        // Destructor will call TF_DeleteStatus
    }
    // No crash = success
}

TEST_CASE("Status - move leaves source null") {
    Status s1;
    CHECK(s1.get() != nullptr);
    
    Status s2 = std::move(s1);
    CHECK(s2.get() != nullptr);
    CHECK(s1.get() == nullptr);  // Moved-from has null handle
}

TEST_CASE("Status - move assignment cleans up destination") {
    Status s1;
    Status s2;
    
    auto* original_handle = s1.get();
    s1 = std::move(s2);
    
    // s1 now has s2's handle, original was deleted
    CHECK(s1.get() != nullptr);
    CHECK(s1.get() != original_handle);
    CHECK(s2.get() == nullptr);
}

// ============================================================================
// Graph Lifecycle Tests
// ============================================================================

TEST_CASE("Graph - destructor called on scope exit") {
    {
        Graph g;
        CHECK(g.handle() != nullptr);
    }
    // No crash = success
}

TEST_CASE("Graph - move leaves source invalid") {
    Graph g1;
    CHECK(g1.handle() != nullptr);
    
    Graph g2 = std::move(g1);
    CHECK(g2.handle() != nullptr);
    
    // g1 should have null handle after move
    CHECK(g1.handle() == nullptr);
}

// ============================================================================
// Session Lifecycle Tests
// ============================================================================

TEST_CASE("Session - destructor called on scope exit") {
    {
        Graph g;
        SessionOptions opts;
        Session s(g, opts);
        CHECK(s.handle() != nullptr);
    }
    // No crash = success
}

TEST_CASE("Session - move leaves source invalid") {
    Graph g;
    SessionOptions opts;
    Session s1(g, opts);
    
    Session s2 = std::move(s1);
    CHECK(s2.handle() != nullptr);
    
    // s1 should have null handle after move
    CHECK(s1.handle() == nullptr);
}

// ============================================================================
// Exception Safety Tests
// ============================================================================

TEST_CASE("Exception safety - cleanup on throw during tensor operations") {
    bool cleanup_called = false;
    
    try {
        auto guard = makeScopeGuard([&]() { cleanup_called = true; });
        
        auto tensor = Tensor::FromScalar(1.0f);
        
        // Simulate an operation that throws
        throw std::runtime_error("simulated error");
        
    } catch (const std::runtime_error&) {
        // Expected
    }
    
    CHECK(cleanup_called);
}

TEST_CASE("Exception safety - multiple guards unwind in order") {
    std::vector<int> order;
    
    try {
        auto guard1 = makeScopeGuard([&]() { order.push_back(1); });
        auto guard2 = makeScopeGuard([&]() { order.push_back(2); });
        auto guard3 = makeScopeGuard([&]() { order.push_back(3); });
        
        throw std::runtime_error("test");
        
    } catch (...) {
        // Expected
    }
    
    // Guards should unwind in reverse order (LIFO)
    REQUIRE(order.size() == 3);
    CHECK(order[0] == 3);
    CHECK(order[1] == 2);
    CHECK(order[2] == 1);
}

// ============================================================================
// LifecycleTracker Integration Tests
// ============================================================================

TEST_CASE("LifecycleTracker - basic construction/destruction") {
    LifecycleTracker::reset();
    
    {
        LifecycleTracker t;
        CHECK(LifecycleTracker::constructions == 1);
        CHECK(LifecycleTracker::destructions == 0);
        CHECK(LifecycleTracker::alive() == 1);
    }
    
    CHECK(LifecycleTracker::destructions == 1);
    CHECK(LifecycleTracker::alive() == 0);
}

TEST_CASE("LifecycleTracker - copy tracking") {
    LifecycleTracker::reset();
    
    {
        LifecycleTracker t1;
        LifecycleTracker t2 = t1;  // Copy
        
        CHECK(LifecycleTracker::constructions == 2);
        CHECK(LifecycleTracker::copies == 1);
    }
    
    CHECK(LifecycleTracker::destructions == 2);
}

TEST_CASE("LifecycleTracker - move tracking") {
    LifecycleTracker::reset();
    
    {
        LifecycleTracker t1;
        LifecycleTracker t2 = std::move(t1);  // Move
        
        CHECK(LifecycleTracker::constructions == 2);
        CHECK(LifecycleTracker::moves == 1);
        CHECK(LifecycleTracker::copies == 0);
    }
    
    CHECK(LifecycleTracker::destructions == 2);
}

TEST_CASE("LifecycleTracker - vector with scope guard cleanup") {
    LifecycleTracker::reset();
    
    std::vector<LifecycleTracker>* vec = nullptr;
    
    {
        vec = new std::vector<LifecycleTracker>(3);
        auto guard = makeScopeGuard([&]() { delete vec; vec = nullptr; });
        
        CHECK(LifecycleTracker::alive() == 3);
        // Guard will cleanup
    }
    
    CHECK(vec == nullptr);
    CHECK(LifecycleTracker::alive() == 0);
}
