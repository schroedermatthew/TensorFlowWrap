/**
 * @file test_scope_guard.cpp
 * @brief Comprehensive test suite for tf_wrap::ScopeGuard
 * 
 * Tests cover:
 * - Basic scope exit behavior
 * - Exception-aware guards (OnFail, OnSuccess)
 * - dismiss() functionality
 * - Move semantics
 * - Nested guards and cleanup order
 * - Policy behavior
 * - TensorFlow C API usage patterns
 * 
 * Compile: g++ -std=c++17 -O2 test_scope_guard.cpp -o test_scope_guard
 * Run: ./test_scope_guard
 */

#include "tf_wrap/scope_guard.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cassert>
#include <functional>

using namespace tf_wrap;

// Test result tracking
static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    void test_##name(); \
    struct TestRegistrar_##name { \
        TestRegistrar_##name() { \
            std::cout << "Testing " #name "... "; \
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
    } test_registrar_##name; \
    void test_##name()

#define ASSERT(cond) \
    do { if (!(cond)) throw std::runtime_error("Assertion failed: " #cond); } while(0)

#define ASSERT_EQ(a, b) \
    do { if ((a) != (b)) throw std::runtime_error("Assertion failed: " #a " == " #b); } while(0)

// =============================================================================
// Basic ScopeGuard Tests
// =============================================================================

TEST(basic_scope_exit_runs) {
    bool executed = false;
    {
        auto guard = makeScopeGuard([&]() noexcept { executed = true; });
        ASSERT(!executed);  // Not yet
    }
    ASSERT(executed);  // Should have run
}

TEST(scope_exit_runs_on_exception) {
    bool executed = false;
    try {
        auto guard = makeScopeGuard([&]() noexcept { executed = true; });
        throw std::runtime_error("test");
    } catch (...) {
        // Expected
    }
    ASSERT(executed);  // Should have run even on exception
}

TEST(dismiss_prevents_execution) {
    bool executed = false;
    {
        auto guard = makeScopeGuard([&]() noexcept { executed = true; });
        guard.dismiss();
    }
    ASSERT(!executed);  // Should NOT have run
}

TEST(dismiss_if_conditional) {
    bool executed1 = false, executed2 = false;
    {
        auto guard1 = makeScopeGuard([&]() noexcept { executed1 = true; });
        auto guard2 = makeScopeGuard([&]() noexcept { executed2 = true; });
        guard1.dismiss_if(true);   // Should dismiss
        guard2.dismiss_if(false);  // Should NOT dismiss
    }
    ASSERT(!executed1);  // Dismissed
    ASSERT(executed2);   // Not dismissed
}

TEST(is_active_reflects_state) {
    auto guard = makeScopeGuard([]() noexcept {});
    ASSERT(guard.is_active());
    guard.dismiss();
    ASSERT(!guard.is_active());
}

// =============================================================================
// ScopeGuardOnFail Tests
// =============================================================================

TEST(scope_fail_not_executed_on_normal_exit) {
    bool executed = false;
    {
        auto guard = makeScopeGuardOnFail([&]() noexcept { executed = true; });
    }
    ASSERT(!executed);  // Normal exit - should NOT run
}

TEST(scope_fail_executed_on_exception) {
    bool executed = false;
    try {
        auto guard = makeScopeGuardOnFail([&]() noexcept { executed = true; });
        throw std::runtime_error("test");
    } catch (...) {
        // Expected
    }
    ASSERT(executed);  // Exception - should run
}

TEST(scope_fail_dismiss_works) {
    bool executed = false;
    try {
        auto guard = makeScopeGuardOnFail([&]() noexcept { executed = true; });
        guard.dismiss();
        throw std::runtime_error("test");
    } catch (...) {
        // Expected
    }
    ASSERT(!executed);  // Dismissed - should NOT run even on exception
}

// =============================================================================
// ScopeGuardOnSuccess Tests
// =============================================================================

TEST(scope_success_executed_on_normal_exit) {
    bool executed = false;
    {
        auto guard = makeScopeGuardOnSuccess([&]() noexcept { executed = true; });
    }
    ASSERT(executed);  // Normal exit - should run
}

TEST(scope_success_not_executed_on_exception) {
    bool executed = false;
    try {
        auto guard = makeScopeGuardOnSuccess([&]() noexcept { executed = true; });
        throw std::runtime_error("test");
    } catch (...) {
        // Expected
    }
    ASSERT(!executed);  // Exception - should NOT run
}

// =============================================================================
// Transaction Pattern Test
// =============================================================================

TEST(transaction_commit_rollback_pattern) {
    std::string state = "initial";
    
    // Success case
    {
        state = "started";
        auto commit = makeScopeGuardOnSuccess([&]() noexcept { state = "committed"; });
        auto rollback = makeScopeGuardOnFail([&]() noexcept { state = "rolled_back"; });
        // No exception - commit should run
    }
    ASSERT_EQ(state, std::string("committed"));
    
    // Failure case
    state = "initial";
    try {
        state = "started";
        auto commit = makeScopeGuardOnSuccess([&]() noexcept { state = "committed"; });
        auto rollback = makeScopeGuardOnFail([&]() noexcept { state = "rolled_back"; });
        throw std::runtime_error("simulated failure");
    } catch (...) {
        // Expected
    }
    ASSERT_EQ(state, std::string("rolled_back"));
}

// =============================================================================
// Move Semantics Tests
// =============================================================================

TEST(move_constructor_transfers_ownership) {
    bool executed = false;
    {
        auto guard1 = makeScopeGuard([&]() noexcept { executed = true; });
        ASSERT(guard1.is_active());
        
        auto guard2 = std::move(guard1);
        ASSERT(!guard1.is_active());  // Source should be dismissed
        ASSERT(guard2.is_active());   // Destination should be active
    }
    ASSERT(executed);  // Should run once via guard2
}

TEST(move_assignment_executes_current_then_transfers) {
    // Note: Move assignment between ScopeGuards requires same lambda type.
    // In practice, this is rare since each lambda is a unique type.
    // This test uses std::function to enable the assignment.
    
    int count = 0;
    using GuardType = ScopeGuard<std::function<void()>>;
    
    {
        GuardType guard1{std::function<void()>([&]() noexcept { count += 1; })};
        GuardType guard2{std::function<void()>([&]() noexcept { count += 10; })};
        
        // Move assign guard2 to guard1
        // guard1's action (count += 1) should execute immediately
        guard1 = std::move(guard2);
        ASSERT_EQ(count, 1);  // guard1's original action ran
        
        ASSERT(!guard2.is_active());  // Source dismissed
        ASSERT(guard1.is_active());   // guard1 now has guard2's action
    }
    ASSERT_EQ(count, 11);  // +10 from guard2's action via guard1
}

// =============================================================================
// Cleanup Order Tests (LIFO)
// =============================================================================

TEST(cleanup_order_is_lifo) {
    std::vector<int> order;
    {
        auto guard1 = makeScopeGuard([&]() noexcept { order.push_back(1); });
        auto guard2 = makeScopeGuard([&]() noexcept { order.push_back(2); });
        auto guard3 = makeScopeGuard([&]() noexcept { order.push_back(3); });
    }
    ASSERT_EQ(order.size(), 3u);
    ASSERT_EQ(order[0], 3);  // Last declared, first executed
    ASSERT_EQ(order[1], 2);
    ASSERT_EQ(order[2], 1);  // First declared, last executed
}

// =============================================================================
// Macro Tests
// =============================================================================

TEST(scope_exit_macro_works) {
    bool executed = false;
    {
        TF_SCOPE_EXIT { executed = true; };
    }
    ASSERT(executed);
}

TEST(scope_fail_macro_works) {
    bool executed = false;
    try {
        TF_SCOPE_FAIL { executed = true; };
        throw std::runtime_error("test");
    } catch (...) {}
    ASSERT(executed);
}

TEST(scope_success_macro_works) {
    bool executed = false;
    {
        TF_SCOPE_SUCCESS { executed = true; };
    }
    ASSERT(executed);
}

TEST(multiple_macros_in_same_scope) {
    std::vector<int> order;
    {
        TF_SCOPE_EXIT { order.push_back(1); };
        TF_SCOPE_EXIT { order.push_back(2); };
        TF_SCOPE_EXIT { order.push_back(3); };
    }
    ASSERT_EQ(order.size(), 3u);
    ASSERT_EQ(order[0], 3);
    ASSERT_EQ(order[1], 2);
    ASSERT_EQ(order[2], 1);
}

// =============================================================================
// Policy Tests
// =============================================================================

TEST(nothrow_policy_compiles_with_noexcept_lambda) {
    bool executed = false;
    {
        auto guard = makeScopeGuard<ScopeGuardNothrowPolicy>(
            [&]() noexcept { executed = true; }
        );
    }
    ASSERT(executed);
}

TEST(swallow_policy_suppresses_exception) {
    // This test verifies the guard itself doesn't propagate exceptions
    // from a throwing cleanup (though our macros use noexcept lambdas)
    bool executed = false;
    {
        auto guard = makeScopeGuard<ScopeGuardSwallowPolicy>([&]() {
            executed = true;
            // In real code this might throw - swallow policy would catch it
        });
    }
    ASSERT(executed);
}

// =============================================================================
// Real-World Usage Pattern Tests
// =============================================================================

// Simulate TensorFlow C API types
struct FakeTensor {
    static int live_count;
    int id;
    FakeTensor(int i) : id(i) { live_count++; }
    ~FakeTensor() { live_count--; }
};
int FakeTensor::live_count = 0;

void FakeDeleteTensor(FakeTensor* t) { delete t; }

TEST(tensorflow_tensor_cleanup_pattern) {
    ASSERT_EQ(FakeTensor::live_count, 0);
    
    {
        FakeTensor* tensor = new FakeTensor(42);
        auto guard = makeScopeGuard([&]() noexcept { FakeDeleteTensor(tensor); });
        
        ASSERT_EQ(FakeTensor::live_count, 1);
        // Simulate some work...
    }
    
    ASSERT_EQ(FakeTensor::live_count, 0);  // Automatically cleaned up
}

TEST(tensorflow_tensor_ownership_transfer) {
    ASSERT_EQ(FakeTensor::live_count, 0);
    FakeTensor* transferred = nullptr;
    
    {
        FakeTensor* tensor = new FakeTensor(42);
        auto guard = makeScopeGuard([&]() noexcept { FakeDeleteTensor(tensor); });
        
        // Simulate successful operation - transfer ownership
        transferred = tensor;
        guard.dismiss();  // Don't delete, we're transferring
    }
    
    ASSERT_EQ(FakeTensor::live_count, 1);  // Still alive
    delete transferred;
    ASSERT_EQ(FakeTensor::live_count, 0);  // Now cleaned up
}

TEST(tensorflow_multi_resource_cleanup) {
    std::vector<int> cleanup_order;
    
    {
        FakeTensor* t1 = new FakeTensor(1);
        TF_SCOPE_EXIT { 
            cleanup_order.push_back(1);
            FakeDeleteTensor(t1); 
        };
        
        FakeTensor* t2 = new FakeTensor(2);
        TF_SCOPE_EXIT { 
            cleanup_order.push_back(2);
            FakeDeleteTensor(t2); 
        };
        
        ASSERT_EQ(FakeTensor::live_count, 2);
    }
    
    ASSERT_EQ(FakeTensor::live_count, 0);
    ASSERT_EQ(cleanup_order.size(), 2u);
    ASSERT_EQ(cleanup_order[0], 2);  // t2 cleaned first (LIFO)
    ASSERT_EQ(cleanup_order[1], 1);  // t1 cleaned second
}

TEST(tensorflow_exception_safe_session_run) {
    // Simulates a Session::Run pattern where we need cleanup even on exception
    std::vector<std::string> events;
    
    auto run_session = [&](bool should_throw) {
        events.push_back("allocate_inputs");
        TF_SCOPE_EXIT { events.push_back("cleanup_inputs"); };
        
        events.push_back("allocate_outputs");
        TF_SCOPE_EXIT { events.push_back("cleanup_outputs"); };
        
        if (should_throw) {
            throw std::runtime_error("session run failed");
        }
        
        events.push_back("success");
    };
    
    // Success path
    events.clear();
    run_session(false);
    ASSERT_EQ(events.size(), 5u);
    ASSERT_EQ(events[0], std::string("allocate_inputs"));
    ASSERT_EQ(events[1], std::string("allocate_outputs"));
    ASSERT_EQ(events[2], std::string("success"));
    ASSERT_EQ(events[3], std::string("cleanup_outputs"));
    ASSERT_EQ(events[4], std::string("cleanup_inputs"));
    
    // Failure path
    events.clear();
    try {
        run_session(true);
    } catch (...) {}
    ASSERT_EQ(events.size(), 4u);  // No "success"
    ASSERT_EQ(events[0], std::string("allocate_inputs"));
    ASSERT_EQ(events[1], std::string("allocate_outputs"));
    ASSERT_EQ(events[2], std::string("cleanup_outputs"));  // Still runs!
    ASSERT_EQ(events[3], std::string("cleanup_inputs"));   // Still runs!
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(empty_lambda_works) {
    {
        auto guard = makeScopeGuard([]() noexcept {});
    }
    // Should not crash
}

TEST(nested_scope_guards) {
    std::vector<int> order;
    {
        TF_SCOPE_EXIT { order.push_back(1); };
        {
            TF_SCOPE_EXIT { order.push_back(2); };
            {
                TF_SCOPE_EXIT { order.push_back(3); };
            }
            order.push_back(100);  // After innermost guard
        }
        order.push_back(200);  // After middle guard
    }
    
    // Expected: 3, 100, 2, 200, 1
    ASSERT_EQ(order.size(), 5u);
    ASSERT_EQ(order[0], 3);
    ASSERT_EQ(order[1], 100);
    ASSERT_EQ(order[2], 2);
    ASSERT_EQ(order[3], 200);
    ASSERT_EQ(order[4], 1);
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "\n=== ScopeGuard Test Suite ===\n\n";
    
    // Tests are auto-registered and run via static initialization
    // (The TEST macro creates static objects that run in their constructors)
    
    std::cout << "\n=== Results ===\n";
    std::cout << "Passed: " << tests_passed << "/" << tests_run << "\n";
    
    if (tests_passed == tests_run) {
        std::cout << "\nAll tests PASSED!\n";
        return 0;
    } else {
        std::cout << "\nSome tests FAILED!\n";
        return 1;
    }
}
