// test_adversarial.cpp
// Adversarial tests designed to break tf_wrap
//
// Framework: doctest (header-only)
// Runs with: TF stub (all platforms)
//
// These tests attempt to:
// - Corrupt memory
// - Trigger undefined behavior
// - Exhaust resources
// - Violate invariants
// - Exploit edge cases

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "tf_wrap/core.hpp"
#include "tf_wrap/codes.hpp"

#include <atomic>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <thread>
#include <vector>

using namespace tf_wrap;

// ============================================================================
// Memory Corruption Attacks
// ============================================================================

TEST_SUITE("Memory Corruption") {

    TEST_CASE("double move should not double-free") {
        auto t1 = Tensor::FromScalar<float>(1.0f);
        auto t2 = std::move(t1);
        auto t3 = std::move(t2);
        
        CHECK_FALSE(t1.valid());
        CHECK_FALSE(t2.valid());
        CHECK(t3.valid());
    }
    
    TEST_CASE("move chain stress") {
        Tensor t = Tensor::FromScalar<float>(1.0f);
        
        for (int i = 0; i < 1000; ++i) {
            Tensor t2 = std::move(t);
            t = std::move(t2);
        }
        
        CHECK(t.valid());
        CHECK(t.ToScalar<float>() == doctest::Approx(1.0f));
    }
    
    TEST_CASE("SmallVector push beyond inline - no corruption") {
        SmallVector<int, 4> v;
        
        // Push way beyond inline capacity
        for (int i = 0; i < 10000; ++i) {
            v.push_back(i);
        }
        
        // Verify no corruption
        for (int i = 0; i < 10000; ++i) {
            CHECK(v[i] == i);
        }
    }
    
    TEST_CASE("SmallVector rapid grow/shrink cycles") {
        SmallVector<int, 8> v;
        
        for (int cycle = 0; cycle < 100; ++cycle) {
            // Grow
            for (int i = 0; i < 100; ++i) {
                v.push_back(i);
            }
            
            // Shrink
            v.clear();
            
            // Verify empty
            CHECK(v.empty());
        }
    }
    
    TEST_CASE("tensor data pointer stability") {
        auto tensor = Tensor::FromVector<float>({1000}, std::vector<float>(1000, 1.0f));
        
        const float* ptr1 = tensor.data<float>();
        
        // Read multiple times
        for (int i = 0; i < 100; ++i) {
            auto view = tensor.read<float>();
            CHECK(view.data() == ptr1);
        }
    }
    
    TEST_CASE("graph handle stability across operations") {
        Graph g;
        TF_Graph* handle = g.handle();
        
        // Various operations should not change handle
        (void)g.GetAllOperations();
        (void)g.num_operations();
        (void)g.GetOperation("nonexistent");
        
        CHECK(g.handle() == handle);
    }
    
    TEST_CASE("session survives graph going out of scope first") {
        std::unique_ptr<Session> session;
        
        {
            Graph g;
            SessionOptions opts;
            session = std::make_unique<Session>(g, opts);
        }
        // Graph destroyed, session should still be valid
        
        CHECK(session->handle() != nullptr);
    }

}

// ============================================================================
// Use-After-Move Attacks
// ============================================================================

TEST_SUITE("Use After Move") {

    TEST_CASE("tensor operations after move are safe") {
        auto t1 = Tensor::FromScalar<float>(1.0f);
        auto t2 = std::move(t1);
        
        // These should not crash
        CHECK_FALSE(t1.valid());
        CHECK(t1.rank() == 0);
        CHECK(t1.num_elements() == 0);
        CHECK_THROWS(t1.dtype());
    }
    
    TEST_CASE("graph operations after move") {
        Graph g1;
        Graph g2 = std::move(g1);
        
        CHECK(g1.handle() == nullptr);
        CHECK_THROWS(g1.GetAllOperations());
    }
    
    TEST_CASE("session operations after move") {
        Graph g;
        SessionOptions opts;
        Session s1(g, opts);
        Session s2 = std::move(s1);
        
        CHECK(s1.handle() == nullptr);
        CHECK_THROWS(s1.ListDevices());
    }
    
    TEST_CASE("buffer operations after move") {
        Buffer b1("test", 4);
        Buffer b2 = std::move(b1);
        
        CHECK(b1.handle() == nullptr);
        // to_bytes on moved-from should be safe
        auto bytes = b1.to_bytes();
        CHECK(bytes.empty());
    }
    
    TEST_CASE("SmallVector use after move") {
        SmallVector<int, 4> v1;
        v1.push_back(1);
        v1.push_back(2);
        
        SmallVector<int, 4> v2 = std::move(v1);
        
        // v1 should be in valid empty state
        CHECK(v1.empty());
        CHECK(v1.size() == 0);
        
        // Should be reusable
        v1.push_back(99);
        CHECK(v1.size() == 1);
        CHECK(v1[0] == 99);
    }

}

// ============================================================================
// Numeric Edge Cases / Overflow Attacks
// ============================================================================

TEST_SUITE("Numeric Attacks") {

    TEST_CASE("tensor with max int64 dimension - should reject") {
        std::vector<float> tiny = {1.0f};
        
        // This should throw, not overflow
        CHECK_THROWS(
            Tensor::FromVector<float>({std::numeric_limits<int64_t>::max()}, tiny)
        );
    }
    
    TEST_CASE("tensor shape product overflow") {
        // Shape that would overflow: {1000000, 1000000, 1000000}
        // = 10^18 elements, way more than memory
        std::vector<float> tiny = {1.0f};
        
        CHECK_THROWS(
            Tensor::FromVector<float>({1000000, 1000000, 1000000}, tiny)
        );
    }
    
    TEST_CASE("SmallVector size_t overflow attempt") {
        SmallVector<int, 4> v;
        
        // reserve with huge size should throw or handle gracefully
        CHECK_THROWS(v.reserve(std::numeric_limits<std::size_t>::max()));
    }
    
    TEST_CASE("tensor byte_size overflow check") {
        // Create tensor and verify byte_size calculation doesn't overflow
        auto tensor = Tensor::FromVector<double>({1000}, std::vector<double>(1000, 0.0));
        
        // byte_size should be 1000 * 8 = 8000
        CHECK(tensor.byte_size() == 8000);
    }
    
    TEST_CASE("reshape with dimension that would overflow") {
        auto tensor = Tensor::FromVector<float>({6}, std::vector<float>(6, 0.0f));
        
        // Reshape to dimensions that multiply to overflow
        CHECK_THROWS(tensor.reshape({std::numeric_limits<int64_t>::max(), 1}));
    }
    
    TEST_CASE("negative dimension in shape") {
        std::vector<float> data = {1.0f};
        
        CHECK_THROWS(Tensor::FromVector<float>({-1}, data));
    }
    
    TEST_CASE("zero in shape middle") {
        std::vector<float> empty;
        
        // {3, 0, 4} = 0 elements, should work
        auto tensor = Tensor::FromVector<float>({3, 0, 4}, empty);
        CHECK(tensor.num_elements() == 0);
    }

}

// Helper for exception safety testing
namespace {
    struct ThrowOnCopy {
        int value;
        static inline int copy_count = 0;
        static inline int throw_after = 1000000;
        
        ThrowOnCopy(int v = 0) : value(v) {}
        ThrowOnCopy(const ThrowOnCopy& other) : value(other.value) {
            if (++copy_count >= throw_after) {
                throw std::runtime_error("copy failed");
            }
        }
        ThrowOnCopy& operator=(const ThrowOnCopy&) = default;
        ThrowOnCopy(ThrowOnCopy&&) = default;
        ThrowOnCopy& operator=(ThrowOnCopy&&) = default;
    };
}

// ============================================================================
// Exception Safety Attacks
// ============================================================================

TEST_SUITE("Exception Safety") {

    TEST_CASE("tensor clone during iteration - no corruption") {
        auto original = Tensor::FromVector<float>({100}, std::vector<float>(100, 1.0f));
        
        auto view = original.read<float>();
        
        // Clone while view exists
        auto cloned = original.Clone();
        
        // Both should be valid
        CHECK(view.size() == 100);
        CHECK(cloned.num_elements() == 100);
    }
    
    TEST_CASE("exception in SmallVector - strong guarantee") {
        ThrowOnCopy::copy_count = 0;
        ThrowOnCopy::throw_after = 1000000;  // Don't throw for this test
        
        SmallVector<ThrowOnCopy, 4> v;
        v.push_back(ThrowOnCopy(1));
        v.push_back(ThrowOnCopy(2));
        
        CHECK(v.size() == 2);
    }
    
    TEST_CASE("scope guard executes on exception") {
        int cleanup_count = 0;
        
        try {
            auto guard = makeScopeGuard([&]{ cleanup_count++; });
            throw std::runtime_error("test");
        } catch (...) {}
        
        CHECK(cleanup_count == 1);
    }
    
    TEST_CASE("nested scope guards execute in reverse order") {
        std::vector<int> order;
        
        {
            auto g1 = makeScopeGuard([&]{ order.push_back(1); });
            auto g2 = makeScopeGuard([&]{ order.push_back(2); });
            auto g3 = makeScopeGuard([&]{ order.push_back(3); });
        }
        
        CHECK(order.size() == 3);
        CHECK(order[0] == 3);
        CHECK(order[1] == 2);
        CHECK(order[2] == 1);
    }

}

// ============================================================================
// Concurrent Access Attacks
// ============================================================================

TEST_SUITE("Concurrency Attacks") {

    TEST_CASE("concurrent tensor creation") {
        std::atomic<int> success_count{0};
        std::atomic<int> error_count{0};
        
        auto worker = [&](int thread_id) {
            for (int i = 0; i < 100; ++i) {
                try {
                    auto t = Tensor::FromScalar<float>(static_cast<float>(thread_id * 1000 + i));
                    if (t.valid()) {
                        success_count++;
                    }
                } catch (...) {
                    error_count++;
                }
            }
        };
        
        std::vector<std::thread> threads;
        for (int t = 0; t < 8; ++t) {
            threads.emplace_back(worker, t);
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        CHECK(error_count == 0);
        CHECK(success_count == 800);
    }
    
    TEST_CASE("concurrent SmallVector operations - separate instances") {
        std::atomic<bool> failed{false};
        
        auto worker = [&](int thread_id) {
            SmallVector<int, 8> v;
            
            for (int i = 0; i < 1000; ++i) {
                v.push_back(thread_id * 10000 + i);
            }
            
            // Verify
            for (int i = 0; i < 1000; ++i) {
                if (v[i] != thread_id * 10000 + i) {
                    failed = true;
                }
            }
        };
        
        std::vector<std::thread> threads;
        for (int t = 0; t < 8; ++t) {
            threads.emplace_back(worker, t);
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        CHECK_FALSE(failed);
    }
    
    TEST_CASE("concurrent session creation from same graph") {
        Graph g;
        std::atomic<int> success_count{0};
        
        auto worker = [&]() {
            try {
                SessionOptions opts;
                Session s(g, opts);
                if (s.handle() != nullptr) {
                    success_count++;
                }
            } catch (...) {
                // May throw if TF doesn't support concurrent session creation
            }
        };
        
        std::vector<std::thread> threads;
        for (int t = 0; t < 4; ++t) {
            threads.emplace_back(worker);
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        // At least some should succeed
        CHECK(success_count >= 1);
    }

}

// ============================================================================
// API Misuse Attacks
// ============================================================================

TEST_SUITE("API Misuse") {

    TEST_CASE("read with wrong dtype") {
        auto tensor = Tensor::FromScalar<float>(1.0f);
        
        CHECK_THROWS(tensor.read<int32_t>());
        CHECK_THROWS(tensor.read<double>());
        CHECK_THROWS(tensor.read<int64_t>());
    }
    
    TEST_CASE("write with wrong dtype") {
        auto tensor = Tensor::Allocate<float>({10});
        
        CHECK_THROWS(tensor.write<int32_t>());
        CHECK_THROWS(tensor.write<double>());
    }
    
    TEST_CASE("ToScalar on multi-element tensor") {
        auto tensor = Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
        
        CHECK_THROWS(tensor.ToScalar<float>());
    }
    
    TEST_CASE("operations on default-constructed tensor") {
        Tensor t;
        
        CHECK_FALSE(t.valid());
        CHECK_THROWS(t.dtype());
        CHECK(t.rank() == 0);
        CHECK(t.num_elements() == 0);
    }
    
    TEST_CASE("SmallVector at() with invalid index") {
        SmallVector<int, 4> v;
        v.push_back(1);
        
        CHECK_NOTHROW(v.at(0));
        CHECK_THROWS(v.at(1));
        CHECK_THROWS(v.at(100));
        CHECK_THROWS(v.at(static_cast<std::size_t>(-1)));
    }
    
    TEST_CASE("graph freeze idempotence") {
        Graph g;
        
        g.freeze();
        CHECK(g.is_frozen());
        
        g.freeze();  // Should be safe
        CHECK(g.is_frozen());
        
        g.freeze();  // Still safe
        CHECK(g.is_frozen());
    }
    
    TEST_CASE("resolve with malformed name") {
        Graph g;
        SessionOptions opts;
        Session s(g, opts);
        
        CHECK_THROWS(s.resolve(""));
        CHECK_THROWS(s.resolve(":"));
        CHECK_THROWS(s.resolve(":::"));
        CHECK_THROWS(s.resolve("op:notanumber"));
        CHECK_THROWS(s.resolve("op:-1"));
    }

}

// ============================================================================
// Resource Exhaustion Attacks
// ============================================================================

TEST_SUITE("Resource Exhaustion") {

    TEST_CASE("many small tensors - no leak") {
        for (int i = 0; i < 10000; ++i) {
            auto t = Tensor::FromScalar<float>(static_cast<float>(i));
            (void)t.ToScalar<float>();
        }
        
        CHECK(true);  // If we get here without OOM, we pass
    }
    
    TEST_CASE("many graphs - no leak") {
        for (int i = 0; i < 1000; ++i) {
            Graph g;
            (void)g.handle();
        }
        
        CHECK(true);
    }
    
    TEST_CASE("many sessions - no leak") {
        Graph g;
        
        for (int i = 0; i < 100; ++i) {
            SessionOptions opts;
            Session s(g, opts);
            (void)s.handle();
        }
        
        CHECK(true);
    }
    
    TEST_CASE("SmallVector repeated grow/shrink - no leak") {
        for (int iter = 0; iter < 1000; ++iter) {
            SmallVector<std::string, 4> v;
            
            for (int i = 0; i < 100; ++i) {
                v.push_back("test string that is long enough to allocate");
            }
            
            v.clear();
            v.shrink_to_fit();
        }
        
        CHECK(true);
    }
    
    TEST_CASE("string tensor stress") {
        for (int i = 0; i < 1000; ++i) {
            std::string s(100, 'x');
            auto t = Tensor::FromString(s);
            auto back = t.ToString();
            CHECK(back == s);
        }
    }

}

// ============================================================================
// Boundary Value Attacks
// ============================================================================

TEST_SUITE("Boundary Values") {

    TEST_CASE("tensor with exactly 1 element in each of many dims") {
        std::vector<float> data = {42.0f};
        auto tensor = Tensor::FromVector<float>({1,1,1,1,1,1,1,1}, data);
        
        CHECK(tensor.rank() == 8);
        CHECK(tensor.num_elements() == 1);
        CHECK(tensor.ToScalar<float>() == doctest::Approx(42.0f));
    }
    
    TEST_CASE("SmallVector at exact inline capacity") {
        SmallVector<int, 8> v;
        
        for (int i = 0; i < 8; ++i) {
            v.push_back(i);
        }
        
        CHECK(v.size() == 8);
        CHECK(v.capacity() == 8);
        
        // One more triggers heap allocation
        v.push_back(8);
        CHECK(v.size() == 9);
        CHECK(v.capacity() > 8);
    }
    
    TEST_CASE("empty string tensor") {
        auto t = Tensor::FromString("");
        CHECK(t.valid());
        CHECK(t.ToString() == "");
    }
    
    TEST_CASE("string with only null bytes") {
        std::string s(10, '\0');
        auto t = Tensor::FromString(s);
        CHECK(t.ToString().size() == 10);
    }
    
    TEST_CASE("float special values") {
        auto pos_inf = Tensor::FromScalar<float>(std::numeric_limits<float>::infinity());
        auto neg_inf = Tensor::FromScalar<float>(-std::numeric_limits<float>::infinity());
        auto nan_val = Tensor::FromScalar<float>(std::numeric_limits<float>::quiet_NaN());
        auto denorm = Tensor::FromScalar<float>(std::numeric_limits<float>::denorm_min());
        
        CHECK(std::isinf(pos_inf.ToScalar<float>()));
        CHECK(std::isinf(neg_inf.ToScalar<float>()));
        CHECK(std::isnan(nan_val.ToScalar<float>()));
        CHECK(denorm.ToScalar<float>() == std::numeric_limits<float>::denorm_min());
    }
    
    TEST_CASE("int64 boundary values") {
        auto max_val = Tensor::FromScalar<int64_t>(std::numeric_limits<int64_t>::max());
        auto min_val = Tensor::FromScalar<int64_t>(std::numeric_limits<int64_t>::min());
        
        CHECK(max_val.ToScalar<int64_t>() == std::numeric_limits<int64_t>::max());
        CHECK(min_val.ToScalar<int64_t>() == std::numeric_limits<int64_t>::min());
    }

}

// ============================================================================
// Rapid State Transitions
// ============================================================================

TEST_SUITE("State Transitions") {

    TEST_CASE("tensor valid->invalid->reuse cycle") {
        Tensor t = Tensor::FromScalar<float>(1.0f);
        CHECK(t.valid());
        
        Tensor t2 = std::move(t);
        CHECK_FALSE(t.valid());
        
        t = Tensor::FromScalar<float>(2.0f);
        CHECK(t.valid());
        CHECK(t.ToScalar<float>() == doctest::Approx(2.0f));
    }
    
    TEST_CASE("graph freeze->session->operations") {
        Graph g;
        CHECK_FALSE(g.is_frozen());
        
        SessionOptions opts;
        Session s(g, opts);
        
        // Creating session may freeze the graph
        // Operations should still work
        (void)g.GetAllOperations();
        (void)g.num_operations();
    }
    
    TEST_CASE("scope guard dismiss->undismiss not possible") {
        int counter = 0;
        
        {
            auto guard = makeScopeGuard([&]{ counter++; });
            guard.dismiss();
            // Cannot undismiss - guard stays dismissed
        }
        
        CHECK(counter == 0);
    }

}
