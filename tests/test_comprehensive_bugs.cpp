// tests/test_comprehensive_bugs.cpp
// Comprehensive test suite for all identified bugs and gaps
//
// This file tests:
// - H1: Clone() race condition
// - H2: DebugString() deadlock with SafeGraph
// - M1: LoadSavedModel lifetime issues
// - View lifetime safety
// - SafeGraph/SafeTensor/SafeSession coverage
//
// Compile: g++ -std=c++20 -pthread -fsanitize=thread -I../include test_comprehensive_bugs.cpp ../third_party/tf_stub/tf_c_stub.cpp -o test_bugs

#include "tf_wrap/all.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

// ============================================================================
// Test Framework
// ============================================================================

namespace tf_test {

struct TestCase {
    const char* name;
    void (*fn)();
    bool is_stress;
    bool expect_fail;  // For tests that demonstrate bugs
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

struct Registrar {
    Registrar(const char* name, void (*fn)(), bool is_stress = false, bool expect_fail = false) {
        registry().push_back({name, fn, is_stress, expect_fail});
    }
};

inline void require_impl(bool cond, const char* expr, const char* file, int line) {
    if (cond) return;
    throw std::runtime_error(std::string("REQUIRE failed: ") + expr + 
                             " (" + file + ":" + std::to_string(line) + ")");
}

template<class Ex, class Fn>
inline void require_throws_impl(Fn&& fn, const char* expr, const char* ex_name,
                                const char* file, int line) {
    try {
        fn();
        throw std::runtime_error(std::string("REQUIRE_THROWS failed: ") + expr + 
                                 " did not throw " + ex_name +
                                 " (" + file + ":" + std::to_string(line) + ")");
    } catch (const Ex&) {
        // Expected
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("REQUIRE_THROWS failed: ") + expr + 
                                 " threw wrong type: " + e.what() +
                                 " (" + file + ":" + std::to_string(line) + ")");
    }
}

} // namespace tf_test

#define TF_JOIN2(a, b) a##b
#define TF_JOIN(a, b) TF_JOIN2(a, b)

#define TEST_CASE(name) \
    static void TF_JOIN(test_fn_, __LINE__)(); \
    static tf_test::Registrar TF_JOIN(test_reg_, __LINE__)( \
        name, &TF_JOIN(test_fn_, __LINE__), false, false); \
    static void TF_JOIN(test_fn_, __LINE__)()

#define STRESS_TEST(name) \
    static void TF_JOIN(stress_fn_, __LINE__)(); \
    static tf_test::Registrar TF_JOIN(stress_reg_, __LINE__)( \
        name, &TF_JOIN(stress_fn_, __LINE__), true, false); \
    static void TF_JOIN(stress_fn_, __LINE__)()

// Test that demonstrates a bug (expected to fail or timeout before fix)
#define BUG_DEMO_TEST(name) \
    static void TF_JOIN(bug_fn_, __LINE__)(); \
    static tf_test::Registrar TF_JOIN(bug_reg_, __LINE__)( \
        name, &TF_JOIN(bug_fn_, __LINE__), true, true); \
    static void TF_JOIN(bug_fn_, __LINE__)()

#define REQUIRE(expr) tf_test::require_impl((expr), #expr, __FILE__, __LINE__)
#define REQUIRE_THROWS_AS(expr, ex) \
    tf_test::require_throws_impl<ex>([&]{ (void)(expr); }, #expr, #ex, __FILE__, __LINE__)

// ============================================================================
// H1: Clone() Race Condition Tests
// ============================================================================

BUG_DEMO_TEST("H1-BUG: Clone during concurrent write detects torn reads") {
    // This test demonstrates BUG H1: Clone() reads without lock
    // With SharedTensor, write() takes exclusive lock but Clone() doesn't lock
    
    constexpr int TENSOR_SIZE = 1000;
    
    std::vector<std::int64_t> shape = {TENSOR_SIZE};
    std::vector<float> init_data(TENSOR_SIZE, 0.0f);
    auto tensor = tf_wrap::SharedTensor::FromVector<float>(shape, init_data);
    
    std::atomic<bool> stop{false};
    std::atomic<int> corrupted{0};
    std::atomic<int> write_count{0};
    std::atomic<int> clone_count{0};
    
    // Writer thread: fills tensor with same value
    std::thread writer([&]() {
        float v = 1.0f;
        while (!stop) {
            {
                auto view = tensor.template write<float>();
                std::fill(view.begin(), view.end(), v);
            }
            v += 1.0f;
            ++write_count;
        }
    });
    
    // Clone checker threads: verify cloned data is consistent
    std::vector<std::thread> cloners;
    for (int t = 0; t < 4; ++t) {
        cloners.emplace_back([&]() {
            while (!stop) {
                auto cloned = tensor.Clone();  // BUG: No lock held during copy!
                auto data = cloned.template ToVector<float>();
                
                // All elements should be the same value
                float first = data[0];
                bool is_consistent = true;
                for (std::size_t i = 1; i < data.size(); ++i) {
                    if (data[i] != first) {
                        is_consistent = false;
                        break;
                    }
                }
                
                if (!is_consistent) {
                    ++corrupted;
                }
                ++clone_count;
            }
        });
    }
    
    // Run for a bit
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    stop = true;
    
    writer.join();
    for (auto& t : cloners) t.join();
    
    std::cout << "     Writes: " << write_count 
              << ", Clones: " << clone_count
              << ", Corrupted: " << corrupted << "\n";
    
    // After H1 is fixed, this should always be 0
    // Before fix, corrupted > 0 demonstrates the race
    REQUIRE(corrupted == 0);
}

TEST_CASE("H1: Clone of empty tensor is safe") {
    tf_wrap::SharedTensor empty;
    auto clone = empty.Clone();
    REQUIRE(clone.empty());
}

TEST_CASE("H1: Clone preserves data correctly (single-threaded)") {
    auto tensor = tf_wrap::FastTensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    auto clone = tensor.Clone();
    
    auto orig_data = tensor.ToVector<float>();
    auto clone_data = clone.ToVector<float>();
    
    REQUIRE(orig_data.size() == clone_data.size());
    for (std::size_t i = 0; i < orig_data.size(); ++i) {
        REQUIRE(orig_data[i] == clone_data[i]);
    }
    
    // Verify they're independent
    {
        auto view = clone.write<float>();
        view[0] = 999.0f;
    }
    REQUIRE(tensor.ToVector<float>()[0] == 1.0f);  // Original unchanged
}

BUG_DEMO_TEST("H1-BUG: Clone race detection aggressive (longer duration)") {
    // More aggressive test with longer duration and larger tensor
    constexpr int TENSOR_SIZE = 10000;
    
    auto tensor = tf_wrap::SharedTensor::FromVector<float>(
        {TENSOR_SIZE}, std::vector<float>(TENSOR_SIZE, 0.0f));
    
    std::atomic<bool> stop{false};
    std::atomic<int> write_count{0};
    std::atomic<int> clone_count{0};
    std::atomic<int> inconsistent_count{0};
    
    // Writer thread - rapidly changes all values
    auto writer = [&]() {
        float val = 1.0f;
        while (!stop) {
            {
                auto view = tensor.template write<float>();
                for (std::size_t i = 0; i < view.size(); ++i) {
                    view[i] = val;
                }
            }
            val += 1.0f;
            ++write_count;
        }
    };
    
    // Clone checker threads
    auto cloner = [&]() {
        while (!stop) {
            auto cloned = tensor.Clone();
            auto data = cloned.template ToVector<float>();
            
            // Check consistency - all values should be the same
            if (!data.empty()) {
                float first = data[0];
                for (std::size_t i = 1; i < data.size(); ++i) {
                    if (data[i] != first) {
                        ++inconsistent_count;
                        break;
                    }
                }
            }
            ++clone_count;
        }
    };
    
    std::thread writer_thread(writer);
    std::vector<std::thread> cloner_threads;
    for (int i = 0; i < 4; ++i) {
        cloner_threads.emplace_back(cloner);
    }
    
    // Run for longer to stress test
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    stop = true;
    
    writer_thread.join();
    for (auto& t : cloner_threads) {
        t.join();
    }
    
    std::cout << "     Writes: " << write_count 
              << ", Clones: " << clone_count 
              << ", Inconsistent: " << inconsistent_count << "\n";
    
    // After H1 is fixed, this should always be 0
    REQUIRE(inconsistent_count == 0);
}

// ============================================================================
// H2: DebugString() Deadlock Tests
// ============================================================================

BUG_DEMO_TEST("H2-BUG: SafeGraph DebugString deadlock check") {
    // This test demonstrates BUG H2: DebugString() calls num_operations()
    // which tries to acquire the same lock -> deadlock with Mutex policy
    //
    // WARNING: This test WILL DEADLOCK if the bug is not fixed!
    
    tf_wrap::SafeGraph g;  // Uses policy::Mutex
    
    auto tensor = tf_wrap::FastTensor::FromScalar<float>(42.0f);
    (void)g.NewOperation("Const", "test_const")
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Use async with timeout to detect deadlock
    auto future = std::async(std::launch::async, [&]() {
        return g.DebugString();  // Will deadlock before fix!
    });
    
    auto status = future.wait_for(std::chrono::seconds(2));
    
    if (status == std::future_status::timeout) {
        std::cout << "     DEADLOCK DETECTED (expected before fix)\n";
        // The deadlock is the bug - we detected it
        throw std::runtime_error("Deadlock detected in SafeGraph::DebugString");
    } else {
        std::string result = future.get();
        REQUIRE(result.find("test_const") != std::string::npos);
        REQUIRE(result.find("1 operations") != std::string::npos);
        std::cout << "     No deadlock (fix applied)\n";
    }
}

TEST_CASE("H2: FastGraph DebugString works (no deadlock with NoLock)") {
    tf_wrap::FastGraph g;
    
    auto tensor = tf_wrap::FastTensor::FromScalar<float>(42.0f);
    (void)g.NewOperation("Const", "myconst")
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    std::string debug = g.DebugString();
    REQUIRE(debug.find("myconst") != std::string::npos);
}

TEST_CASE("H2: SafeGraph num_operations standalone") {
    tf_wrap::SafeGraph g;
    
    REQUIRE(g.num_operations() == 0);
    
    auto tensor = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "c1")
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    REQUIRE(g.num_operations() == 1);
}

// ============================================================================
// SafeGraph/SafeTensor/SafeSession Coverage
// ============================================================================

TEST_CASE("SafeGraph all methods work") {
    tf_wrap::SafeGraph g;
    
    auto t1 = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    auto t2 = tf_wrap::FastTensor::FromScalar<float>(2.0f);
    
    (void)g.NewOperation("Const", "const1")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    (void)g.NewOperation("Const", "const2")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    REQUIRE(g.num_operations() == 2);
    REQUIRE(g.GetOperation("const1").has_value());
    REQUIRE(g.GetOperation("nonexistent") == std::nullopt);
    
    // GetOperationOrThrow returns TF_Operation*
    TF_Operation* found = g.GetOperationOrThrow("const1");
    REQUIRE(std::string(TF_OperationName(found)) == "const1");
    
    bool threw = false;
    try {
        (void)g.GetOperationOrThrow("nonexistent");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST_CASE("SafeTensor read/write thread safety") {
    auto tensor = tf_wrap::SafeTensor::FromVector<float>({5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    
    // Writer
    {
        auto view = tensor.write<float>();
        view[0] = 100.0f;
    }
    
    // Reader
    {
        auto view = tensor.read<float>();
        REQUIRE(view[0] == 100.0f);
        REQUIRE(view[4] == 5.0f);
    }
}

STRESS_TEST("SafeTensor concurrent access is serialized") {
    std::vector<std::int64_t> shape = {100};
    std::vector<float> init_data(100, 0.0f);
    auto tensor = tf_wrap::SafeTensor::FromVector<float>(shape, init_data);
    
    std::atomic<bool> stop{false};
    std::atomic<int> reads{0};
    std::atomic<int> writes{0};
    std::atomic<int> torn{0};
    
    // Writer
    std::thread writer([&]() {
        float v = 1.0f;
        while (!stop) {
            auto view = tensor.template write<float>();
            std::fill(view.begin(), view.end(), v);
            v += 1.0f;
            ++writes;
        }
    });
    
    // Readers
    std::vector<std::thread> readers;
    for (int i = 0; i < 4; ++i) {
        readers.emplace_back([&]() {
            while (!stop) {
                auto view = tensor.template read<float>();
                float first = view[0];
                for (std::size_t j = 1; j < view.size(); ++j) {
                    if (view[j] != first) {
                        ++torn;
                        break;
                    }
                }
                ++reads;
            }
        });
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop = true;
    
    writer.join();
    for (auto& t : readers) t.join();
    
    std::cout << "     Reads: " << reads << ", Writes: " << writes << ", Torn: " << torn << "\n";
    REQUIRE(torn == 0);
}

STRESS_TEST("SharedTensor allows concurrent readers") {
    std::vector<std::int64_t> shape = {100};
    std::vector<float> init_data(100, 42.0f);
    auto tensor = tf_wrap::SharedTensor::FromVector<float>(shape, init_data);
    
    std::atomic<int> max_concurrent{0};
    std::atomic<int> current_readers{0};
    std::atomic<bool> stop{false};
    
    std::vector<std::thread> readers;
    for (int i = 0; i < 8; ++i) {
        readers.emplace_back([&]() {
            while (!stop) {
                auto view = tensor.template read<float>();
                int cur = ++current_readers;
                
                // Update max if this is highest concurrent count
                int expected = max_concurrent.load();
                while (cur > expected && !max_concurrent.compare_exchange_weak(expected, cur));
                
                // Simulate some work
                volatile float sum = 0;
                for (auto x : view) sum += x;
                (void)sum;
                
                --current_readers;
            }
        });
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop = true;
    
    for (auto& t : readers) t.join();
    
    std::cout << "     Max concurrent readers: " << max_concurrent << "\n";
    REQUIRE(max_concurrent > 1);  // Should have had concurrent readers
}

// ============================================================================
// View Lifetime Tests
// ============================================================================

TEST_CASE("View keeps tensor alive after Tensor destroyed") {
    // Create tensor in inner scope, but extract view
    std::optional<tf_wrap::SharedTensor::ReadView<float>> opt_view;
    
    {
        auto tensor = tf_wrap::SharedTensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
        opt_view.emplace(tensor.read<float>());
    }  // tensor destroyed, but view keeps shared_ptr to state alive
    
    // View should still be valid!
    auto& view = *opt_view;
    REQUIRE(view.size() == 3);
    REQUIRE(view[0] == 1.0f);
    REQUIRE(view[1] == 2.0f);
    REQUIRE(view[2] == 3.0f);
}

TEST_CASE("Write view keeps tensor alive") {
    std::optional<tf_wrap::SharedTensor::WriteView<float>> opt_view;
    
    {
        auto tensor = tf_wrap::SharedTensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
        opt_view.emplace(tensor.write<float>());
    }  // tensor destroyed, but view keeps shared_ptr to state alive
    
    auto& view = *opt_view;
    REQUIRE(view.size() == 3);
    view[0] = 100.0f;
    REQUIRE(view[0] == 100.0f);
}

// ============================================================================
// LoadSavedModel Lifetime Tests (M1)
// ============================================================================

// Note: This test would require a real SavedModel directory
// For stub testing, we just verify the API compiles correctly

TEST_CASE("M1: Session and Graph structured binding") {
    // This demonstrates correct usage
    tf_wrap::FastGraph graph;
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    (void)graph.NewOperation("Const", "c")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Simulate what LoadSavedModel returns - both must stay alive
    // auto [session, graph] = Session::LoadSavedModel("/path");
    // session.Run(...);  // Works as long as graph is alive
}

// ============================================================================
// Error Consistency Tests (L1)
// ============================================================================

TEST_CASE("L1: Empty tensor error handling consistency") {
    tf_wrap::FastTensor empty;
    
    // byte_size() returns 0 silently
    REQUIRE(empty.byte_size() == 0);
    
    // num_elements() returns 0 for empty tensor
    REQUIRE(empty.num_elements() == 0);
    
    // The inconsistency would be if num_elements threw but byte_size didn't
    // Both should handle empty gracefully
}

// ============================================================================
// Additional Edge Cases
// ============================================================================

TEST_CASE("Multiple read views from same SharedTensor") {
    auto tensor = tf_wrap::SharedTensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    
    auto view1 = tensor.read<float>();
    auto view2 = tensor.read<float>();  // Both hold shared locks
    
    REQUIRE(view1[0] == view2[0]);
    REQUIRE(view1[1] == view2[1]);
    REQUIRE(view1[2] == view2[2]);
}

TEST_CASE("Tensor move leaves valid empty state") {
    auto t1 = tf_wrap::FastTensor::FromScalar<float>(42.0f);
    auto t2 = std::move(t1);
    
    // t1 should be in valid empty state
    REQUIRE(t1.empty());
    REQUIRE(t1.handle() == nullptr);
    REQUIRE(t1.byte_size() == 0);
    
    // t2 should have the data
    REQUIRE(!t2.empty());
    REQUIRE(t2.ToScalar<float>() == 42.0f);
}

TEST_CASE("Graph move: moved-from must throw on use") {
    tf_wrap::FastGraph g1;
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    (void)g1.NewOperation("Const", "c")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    REQUIRE(g1.num_operations() == 1);
    
    tf_wrap::FastGraph g2 = std::move(g1);
    
    // g2 should have the operation
    REQUIRE(g2.num_operations() == 1);
    REQUIRE(g2.valid());
    
    // g1 must be invalid and throw on use (not silently return empty)
    REQUIRE(!g1.valid());
    REQUIRE_THROWS_AS(g1.num_operations(), std::runtime_error);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    bool run_stress = false;
    bool run_bug_demos = false;
    
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--stress") run_stress = true;
        if (std::string(argv[i]) == "--bugs") run_bug_demos = true;
        if (std::string(argv[i]) == "--all") { run_stress = true; run_bug_demos = true; }
    }
    
    int passed = 0, failed = 0, skipped = 0;
    const auto& tests = tf_test::registry();
    
    std::cout << "Running comprehensive bug tests...\n\n";
    
    for (const auto& tc : tests) {
        // Skip stress tests unless --stress or --all
        if (tc.is_stress && !run_stress && !tc.expect_fail) {
            ++skipped;
            continue;
        }
        
        // Skip bug demos unless --bugs or --all
        if (tc.expect_fail && !run_bug_demos) {
            ++skipped;
            continue;
        }
        
        std::cout << "[TEST]   " << tc.name << "\n";
        
        try {
            tc.fn();
            if (tc.expect_fail) {
                std::cout << "  UNEXPECTED PASS (bug may be fixed!)\n";
            } else {
                std::cout << "  PASS\n";
            }
            ++passed;
        } catch (const std::exception& e) {
            if (tc.expect_fail) {
                std::cout << "  EXPECTED FAIL: " << e.what() << "\n";
                ++passed;  // Expected failure counts as pass
            } else {
                std::cout << "  FAIL: " << e.what() << "\n";
                ++failed;
            }
        }
    }
    
    std::cout << "\n========================================\n";
    std::cout << "Passed: " << passed << ", Failed: " << failed << ", Skipped: " << skipped << "\n";
    
    if (skipped > 0) {
        std::cout << "(Use --stress for stress tests, --bugs for bug demos, --all for everything)\n";
    }
    
    return failed > 0 ? 1 : 0;
}
