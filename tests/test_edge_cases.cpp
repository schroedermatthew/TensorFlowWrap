// test_edge_cases_new.cpp
// Edge case tests using doctest
//
// Run with: ./test_edge_cases
// Run stress tests: ./test_edge_cases -ts=stress

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

// Suppress warnings from CHECK_THROWS_AS with [[nodiscard]] functions
#if defined(_MSC_VER)
#pragma warning(disable: 4834)  // discarding return value of function with [[nodiscard]]
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wunused-result"
#endif

#include "tf_wrap/all.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

TEST_CASE("empty tensor from zero-sized vector") {
    std::vector<std::int64_t> shape = {0};
    std::vector<float> data = {};
    auto tensor = tf_wrap::Tensor::FromVector<float>(shape, data);
    CHECK(tensor.num_elements() == 0);
    CHECK(tensor.byte_size() == 0);
    CHECK(tensor.rank() == 1);
}

TEST_CASE("empty tensor from zero dimension in middle") {
    std::vector<std::int64_t> shape = {10, 0, 5};
    auto tensor = tf_wrap::Tensor::Allocate<float>(shape);
    CHECK(tensor.num_elements() == 0);
    CHECK(tensor.byte_size() == 0);
    CHECK(tensor.rank() == 3);
}

TEST_CASE("scalar has empty shape") {
    auto tensor = tf_wrap::Tensor::FromScalar<int>(42);
    CHECK(tensor.shape().empty());
    CHECK(tensor.rank() == 0);
    CHECK(tensor.num_elements() == 1);
}

TEST_CASE("empty tensor read view") {
    std::vector<std::int64_t> shape = {0};
    std::vector<float> data = {};
    auto tensor = tf_wrap::Tensor::FromVector<float>(shape, data);
    auto view = tensor.read<float>();
    CHECK(view.size() == 0);
    CHECK(view.begin() == view.end());
}

// ============================================================================
// Boundary Value Tests
// ============================================================================

TEST_CASE("single element 1D tensor") {
    std::vector<std::int64_t> shape = {1};
    std::vector<float> data = {42.0f};
    auto tensor = tf_wrap::Tensor::FromVector<float>(shape, data);
    CHECK(tensor.num_elements() == 1);
    auto view = tensor.read<float>();
    CHECK(view[0] == 42.0f);
}

TEST_CASE("high rank tensor 8 dimensions") {
    std::vector<std::int64_t> shape(8, 2);  // 2^8 = 256 elements
    auto tensor = tf_wrap::Tensor::Allocate<float>(shape);
    CHECK(tensor.rank() == 8);
    CHECK(tensor.num_elements() == 256);
}

TEST_CASE("negative dimension throws") {
    std::vector<std::int64_t> shape = {10, -1, 5};
    CHECK_THROWS_AS(
        tf_wrap::Tensor::Allocate<float>(shape),
        std::invalid_argument);
}

TEST_CASE("overflow in dimension product throws") {
    // Use dimensions that are individually valid but overflow when multiplied
    // INT64_MAX / 2 is positive and valid, but (INT64_MAX/2) * 3 overflows
    std::vector<std::int64_t> shape = {INT64_MAX / 2, 3};
    CHECK_THROWS_AS(
        tf_wrap::Tensor::Allocate<float>(shape),
        std::overflow_error);
}

TEST_CASE("INT64_MAX dimension throws overflow") {
    std::vector<std::int64_t> shape = {INT64_MAX};
    CHECK_THROWS_AS(
        tf_wrap::Tensor::Allocate<float>(shape),
        std::overflow_error);
}

// ============================================================================
// Moved-From Object Tests
// ============================================================================

TEST_CASE("moved-from tensor has null handle") {
    auto t1 = tf_wrap::Tensor::FromScalar<float>(1.0f);
    CHECK(t1.handle() != nullptr);
    
    auto t2 = std::move(t1);
    CHECK(t2.handle() != nullptr);
    // t1 is now in moved-from state
}

TEST_CASE("moved-from status is safe to destroy") {
    tf_wrap::Status s1;
    s1.set(TF_CANCELLED, "test");
    tf_wrap::Status s2 = std::move(s1);
    
    CHECK(s2.code() == TF_CANCELLED);
    // s1 destruction should not double-free
}

TEST_CASE("moved-from graph must throw on use") {
    tf_wrap::Graph g1;
    auto tensor = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g1.NewOperation("Const", "c")
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Graph g2 = std::move(g1);
    CHECK(g2.GetOperation("c").has_value());
    
    // Moved-from must throw, not silently return empty
    CHECK(!g1.valid());
    CHECK_THROWS(g1.num_operations());
}

TEST_CASE("moved-from session options is safe") {
    tf_wrap::SessionOptions opts1;
    tf_wrap::SessionOptions opts2 = std::move(opts1);
    
    CHECK(opts2.get() != nullptr);
    CHECK(opts1.get() == nullptr);
}

// ============================================================================
// Adopt/AdoptMalloc Error Tests
// ============================================================================

TEST_CASE("AdoptMalloc wrong byte_len throws") {
    void* data = std::malloc(100);
    CHECK(data != nullptr);
    
    std::vector<std::int64_t> shape = {10};
    bool threw = false;
    try {
        // Shape {10} with float requires 40 bytes, not 100
        (void)tf_wrap::Tensor::AdoptMalloc<float>(shape, data, 100);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    
    CHECK(threw);
    std::free(data);  // We still own it
}

TEST_CASE("Adopt with null deallocator throws") {
    void* data = std::malloc(40);
    CHECK(data != nullptr);
    
    std::vector<std::int64_t> shape = {10};
    CHECK_THROWS_AS(
        tf_wrap::Tensor::Adopt(TF_FLOAT, shape, data, 40, nullptr),
        std::invalid_argument);
    
    std::free(data);
}

TEST_CASE("Adopt with null data and zero bytes succeeds") {
    auto dealloc = [](void*, std::size_t, void*) {};
    std::vector<std::int64_t> shape = {0};
    
    // Should not throw
    auto tensor = tf_wrap::Tensor::Adopt(TF_FLOAT, shape, nullptr, 0, dealloc);
    CHECK(tensor.num_elements() == 0);
}

TEST_CASE("Adopt with null data and non-zero bytes throws") {
    auto dealloc = [](void*, std::size_t, void*) {};
    std::vector<std::int64_t> shape = {10};
    
    CHECK_THROWS_AS(
        tf_wrap::Tensor::Adopt(TF_FLOAT, shape, nullptr, 40, dealloc),
        std::invalid_argument);
}

// ============================================================================
// Graph Error Tests
// ============================================================================

TEST_CASE("GetOperationOrThrow on empty graph throws") {
    tf_wrap::Graph graph;
    CHECK_THROWS_AS(
        graph.GetOperationOrThrow("nonexistent"),
        std::runtime_error);
}

TEST_CASE("GetOperation on empty graph returns nullopt") {
    tf_wrap::Graph graph;
    auto result = graph.GetOperation("nonexistent");
    CHECK(!result.has_value());
}

// ============================================================================
// Status Edge Cases
// ============================================================================

TEST_CASE("Status set with empty string_view") {
    tf_wrap::Status st;
    st.set(TF_INTERNAL, std::string_view{});
    CHECK(st.code() == TF_INTERNAL);
    CHECK(std::string(st.message()).empty());
}

TEST_CASE("Status set with very long message") {
    tf_wrap::Status st;
    std::string long_msg(10000, 'x');
    st.set(TF_INTERNAL, long_msg);
    CHECK(st.code() == TF_INTERNAL);
    CHECK(std::string(st.message()) == long_msg);
}

TEST_CASE("Status all error codes have names") {
    std::vector<TF_Code> codes = {
        TF_OK, TF_CANCELLED, TF_UNKNOWN, TF_INVALID_ARGUMENT,
        TF_DEADLINE_EXCEEDED, TF_NOT_FOUND, TF_ALREADY_EXISTS,
        TF_PERMISSION_DENIED, TF_UNAUTHENTICATED, TF_RESOURCE_EXHAUSTED,
        TF_FAILED_PRECONDITION, TF_ABORTED, TF_OUT_OF_RANGE,
        TF_UNIMPLEMENTED, TF_INTERNAL, TF_UNAVAILABLE, TF_DATA_LOSS
    };
    
    for (auto code : codes) {
        const char* name = tf_wrap::Status::code_to_string(code);
        CHECK(name != nullptr);
        CHECK(std::string(name) != "UNKNOWN_CODE");
    }
}

// ============================================================================
// Stress Tests
// ============================================================================

TEST_CASE("rapid tensor allocation/deallocation" * doctest::test_suite("stress")) {
    constexpr int iterations = 10000;
    
    auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        std::vector<std::int64_t> shape = {100, 100};
        auto t = tf_wrap::Tensor::Allocate<float>(shape);
    }
    
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    const double ms_d = static_cast<double>(ms);
    std::cout << "    " << iterations << " allocations in " << ms << "ms "
              << "(" << (static_cast<double>(iterations) * 1000.0 / (ms_d + 1.0))
              << " ops/sec)\n";
}

TEST_CASE("concurrent tensor creation" * doctest::test_suite("stress")) {
    constexpr int num_threads = 8;
    constexpr int ops_per_thread = 1000;
    
    std::atomic<int> completed{0};
    std::vector<std::thread> threads;
    
    auto start = std::chrono::steady_clock::now();
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < ops_per_thread; ++i) {
                auto tensor = tf_wrap::Tensor::FromScalar<float>(
                    static_cast<float>(t * ops_per_thread + i));
                ++completed;
            }
        });
    }
    
    for (auto& t : threads) t.join();
    
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    CHECK(completed.load() == num_threads * ops_per_thread);
    std::cout << "    " << completed.load() << " concurrent ops in " << ms << "ms\n";
}

// NOTE: "Tensor reader/writer contention" test removed.
// This test verified that mutex locking prevented torn reads.
// Since the policy-based locking system was removed in v5.0,
// tensors are no longer thread-safe and this test is not applicable.
// Users should not share mutable tensors across threads.

TEST_CASE("Tensor allows concurrent readers" * doctest::test_suite("stress")) {
    std::vector<std::int64_t> shape = {1000};
    std::vector<float> data(1000, 42.0f);
    auto tensor = tf_wrap::Tensor::FromVector<float>(shape, data);
    
    std::atomic<int> max_concurrent{0};
    std::atomic<int> current_readers{0};
    std::atomic<bool> stop{false};
    
    std::vector<std::thread> readers;
    for (int i = 0; i < 8; ++i) {
        readers.emplace_back([&]() {
            while (!stop) {
                auto view = tensor.read<float>();
                
                int current = ++current_readers;
                int expected = max_concurrent.load();
                while (current > expected) {
                    max_concurrent.compare_exchange_weak(expected, current);
                }
                
                float sum = 0;
                for (auto x : view) sum += x;
                (void)sum;
                
                --current_readers;
            }
        });
    }
    
    std::this_thread::sleep_for(std::chrono::seconds(1));
    stop = true;
    
    for (auto& r : readers) r.join();
    
    std::cout << "    Max concurrent readers: " << max_concurrent << "\n";
    CHECK(max_concurrent > 1);
}

TEST_CASE("Status rapid creation/destruction" * doctest::test_suite("stress")) {
    constexpr int iterations = 100000;
    
    auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        tf_wrap::Status st;
        st.set(TF_INTERNAL, "test");
        st.reset();
    }
    
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "    " << iterations << " Status cycles in " << ms << "ms\n";
}

// ============================================================================
// Fuzz Tests - Random inputs to find edge cases
// ============================================================================

TEST_CASE("fuzz: random tensor shapes" * doctest::test_suite("stress")) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> rank_dist(0, 5);
    std::uniform_int_distribution<std::int64_t> dim_dist(0, 50);
    
    int created = 0;
    int failed_expected = 0;
    
    for (int i = 0; i < 500; ++i) {
        std::vector<std::int64_t> shape;
        int rank = rank_dist(gen);
        
        std::int64_t total_elements = 1;
        
        for (int d = 0; d < rank; ++d) {
            std::int64_t dim = dim_dist(gen);
            shape.push_back(dim);
            
            if (dim == 0) {
                total_elements = 0;
            } else if (total_elements > 0 && 
                       total_elements <= std::numeric_limits<std::int64_t>::max() / dim) {
                total_elements *= dim;
            }
        }
        
        try {
            std::vector<float> data(static_cast<std::size_t>(total_elements), 1.0f);
            auto tensor = tf_wrap::Tensor::FromVector<float>(shape, data);
            
            CHECK(tensor.shape() == shape);
            CHECK(static_cast<std::int64_t>(tensor.num_elements()) == total_elements);
            
            if (total_elements > 0) {
                auto view = tensor.read<float>();
                CHECK(view.size() == static_cast<std::size_t>(total_elements));
            }
            
            ++created;
        } catch (const std::exception&) {
            ++failed_expected;
        }
    }
    
    std::cout << "    Created " << created << " tensors, " 
              << failed_expected << " expected failures\n";
    CHECK(created > 200);
}

TEST_CASE("fuzz: random dtype values" * doctest::test_suite("stress")) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    int tested = 0;
    std::vector<std::int64_t> shape = {5, 5};
    
    // Test float with random values
    for (int i = 0; i < 50; ++i) {
        std::vector<float> data(25);
        for (auto& v : data) {
            v = static_cast<float>(gen()) / static_cast<float>(gen() + 1);
        }
        
        auto tensor = tf_wrap::Tensor::FromVector<float>(shape, data);
        auto extracted = tensor.ToVector<float>();
        CHECK(extracted == data);
        ++tested;
    }
    
    // Test int32 with random values  
    for (int i = 0; i < 50; ++i) {
        std::vector<std::int32_t> data(25);
        for (auto& v : data) {
            v = static_cast<std::int32_t>(gen() % 10000);
        }
        
        auto tensor = tf_wrap::Tensor::FromVector<std::int32_t>(shape, data);
        auto extracted = tensor.ToVector<std::int32_t>();
        CHECK(extracted == data);
        ++tested;
    }
    
    std::cout << "    Tested " << tested << " random tensor values\n";
}

TEST_CASE("fuzz: OperationBuilder exception safety" * doctest::test_suite("stress")) {
    tf_wrap::Graph graph;
    int abandoned = 0;
    
    for (int i = 0; i < 50; ++i) {
        try {
            auto builder = graph.NewOperation("Const", "test_" + std::to_string(i));
            
            if (i % 3 == 0) {
                throw std::runtime_error("simulated error");
            }
            
            auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
            builder.SetAttrTensor("value", t.handle());
            builder.SetAttrType("dtype", TF_FLOAT);
            (void)std::move(builder).Finish();
            
        } catch (const std::exception&) {
            ++abandoned;
        }
    }
    
    std::cout << "    " << abandoned << " builders safely abandoned\n";
    CHECK(abandoned > 0);
}

TEST_CASE("fuzz: moved-from object safety" * doctest::test_suite("stress")) {
    for (int i = 0; i < 50; ++i) {
        tf_wrap::Tensor t1 = tf_wrap::Tensor::FromScalar<float>(static_cast<float>(i));
        tf_wrap::Tensor t2 = std::move(t1);
        
        CHECK(t1.empty());
        CHECK(t1.handle() == nullptr);
        CHECK(!t2.empty());
        CHECK(t2.ToScalar<float>() == static_cast<float>(i));
        
        tf_wrap::Tensor t3 = std::move(t2);
        CHECK(t2.empty());
        CHECK(!t3.empty());
        
        t1 = std::move(t3);
        CHECK(t3.empty());
        CHECK(!t1.empty());
    }
    
    std::cout << "    50 move sequences completed safely\n";
}

// ============================================================================
// Graph Moved-From State Tests (P1 #4 fix verification)
// ============================================================================

TEST_CASE("Graph moved-from: must throw on use after move") {
    tf_wrap::Graph g;

    // Create a real op so the destination graph is non-empty.
    auto t = tf_wrap::Tensor::FromVector<float>({1}, {1.0f});
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();

    tf_wrap::Graph g2(std::move(g));

    // Destination must still work.
    CHECK(g2.num_operations() == 1u);
    CHECK(g2.HasOperation("A"));
    CHECK(g2.valid());

    // Moved-from must be invalid and throw on handle-touching calls.
    CHECK(!g.valid());
    CHECK_THROWS(g.num_operations());
    CHECK_THROWS(g.GetAllOperations());
    CHECK_THROWS(g.GetOperation("A"));
    CHECK_THROWS(g.HasOperation("A"));
    CHECK_THROWS(g.DebugString());
}

TEST_CASE("Graph move-assign: moved-from must throw on use") {
    tf_wrap::Graph g1;
    tf_wrap::Graph g2;

    auto t = tf_wrap::Tensor::FromVector<float>({1}, {1.0f});
    (void)g1.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();

    g2 = std::move(g1);

    // Destination has the operation
    CHECK(g2.num_operations() == 1u);
    CHECK(g2.HasOperation("A"));
    CHECK(g2.valid());

    // Moved-from must throw
    CHECK(!g1.valid());
    CHECK_THROWS(g1.num_operations());
    CHECK_THROWS(g1.GetAllOperations());
    CHECK_THROWS(g1.NewOperation("Const", "B"));
}

TEST_CASE("Graph self-move assignment leaves graph intact") {
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);

    (void)g.NewOperation("Const", "c")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();

    // Intentional self-move to test robustness
    // Use volatile pointer to prevent compiler from detecting self-move
    tf_wrap::Graph* volatile p = &g;
    g = std::move(*p);

    CHECK(g.handle() != nullptr);
    CHECK(g.valid());
    CHECK(g.HasOperation("c"));
}

TEST_CASE("fuzz: moved-from graph safety (handle semantics)" * doctest::test_suite("stress")) {
    for (int i = 0; i < 50; ++i) {
        tf_wrap::Graph g1;
        auto t = tf_wrap::Tensor::FromScalar<float>(static_cast<float>(i));

        (void)g1.NewOperation("Const", "c")
            .SetAttrTensor("value", t.handle())
            .SetAttrType("dtype", TF_FLOAT)
            .Finish();

        tf_wrap::Graph g2 = std::move(g1);
        CHECK(g2.HasOperation("c"));

        // Moved-from must throw (our chosen contract)
        CHECK(!g1.valid());
        CHECK_THROWS(g1.num_operations());
        CHECK_THROWS(g1.DebugString());

        tf_wrap::Graph g3 = std::move(g2);
        CHECK(g3.HasOperation("c"));
        CHECK(!g2.valid());
    }
    
    std::cout << "    50 graph move sequences completed safely\n";
}

// ============================================================================
// Session/Graph Freeze Tests (enforces TF's immutability requirement)
// ============================================================================

TEST_CASE("Graph mutation after Session creation must throw") {
    tf_wrap::Graph g;

    // Build a minimal graph
    auto a = tf_wrap::Tensor::FromVector<float>({2}, {1.0f, 2.0f});
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", a.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();

    // Graph should not be frozen yet
    CHECK_FALSE(g.is_frozen());

    // Create session - this freezes the graph
    tf_wrap::Session s(g);
    
    // Graph must now be frozen
    CHECK(g.is_frozen());

    // Any mutation attempt must throw
    auto b = tf_wrap::Tensor::FromScalar<float>(0.0f);
    CHECK_THROWS(g.NewOperation("Const", "B")
        .SetAttrTensor("value", b.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish());
    
    // Session should still work
    auto result = s.Run("A", 0);
    CHECK((result.ToVector<float>() == std::vector<float>{1.0f, 2.0f}));
    
    std::cout << "    Graph freeze enforced after Session creation\n";
}

TEST_CASE("Graph freeze works with different Session/Graph policies") {
    // Graph with Session - policies can differ
    tf_wrap::Graph g;

    auto a = tf_wrap::Tensor::FromVector<float>({2}, {1.0f, 2.0f});
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", a.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();

    tf_wrap::Session s(g);
    
    CHECK(g.is_frozen());
    
    // Mutation must throw
    auto b = tf_wrap::Tensor::FromScalar<float>(0.0f);
    CHECK_THROWS(g.NewOperation("Const", "B")
        .SetAttrTensor("value", b.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish());
}

// ============================================================================
// CRITICAL: Session moved-from state tests
// ============================================================================

TEST_CASE("Session move constructor leaves source invalid") {
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Session s1(g);
    tf_wrap::Session s2(std::move(s1));
    
    // s2 should work
    CHECK(s2.handle() != nullptr);
    
    // s1 should be invalid
    CHECK(s1.handle() == nullptr);
}

TEST_CASE("Session move assignment leaves source invalid") {
    tf_wrap::Graph g1;
    auto t1 = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g1.NewOperation("Const", "A")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Graph g2;
    auto t2 = tf_wrap::Tensor::FromScalar<float>(2.0f);
    (void)g2.NewOperation("Const", "B")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Session s1(g1);
    tf_wrap::Session s2(g2);
    
    s2 = std::move(s1);
    
    CHECK(s2.handle() != nullptr);
    CHECK(s1.handle() == nullptr);
}

TEST_CASE("Moved-from session throws on Run") {
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Session s1(g);
    tf_wrap::Session s2(std::move(s1));
    
    CHECK_THROWS(s1.Run("A", 0));
}

TEST_CASE("Moved-from session is safe to destroy") {
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    {
        tf_wrap::Session s1(g);
        tf_wrap::Session s2(std::move(s1));
        // s1 destroyed here - should not crash
    }
    // If we get here, destruction was safe
    CHECK(true);
}

// ============================================================================
// CRITICAL: Session from invalid Graph tests
// ============================================================================

TEST_CASE("Session from moved-from graph throws") {
    tf_wrap::Graph g1;
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g1.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Graph g2(std::move(g1));
    
    // Creating session from moved-from graph should throw
    bool threw = false;
    try {
        tf_wrap::Session s(g1);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

// ============================================================================
// CRITICAL: OperationBuilder lifecycle tests
// ============================================================================

TEST_CASE("OperationBuilder Finish completes operation") {
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    
    // Create and finish an operation
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Should be able to create another operation immediately
    auto t2 = tf_wrap::Tensor::FromScalar<float>(2.0f);
    (void)g.NewOperation("Const", "B")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    CHECK(g.num_operations() == 2u);
}

TEST_CASE("OperationBuilder without Finish is safe") {
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    
    {
        // Create builder but don't finish - explicitly abandon it
        auto builder = g.NewOperation("Const", "A")
            .SetAttrTensor("value", t.handle())
            .SetAttrType("dtype", TF_FLOAT);
        builder.Abandon();  // Explicitly abandon to avoid debug assertion
    }
    
    // Should be able to create another operation
    auto t2 = tf_wrap::Tensor::FromScalar<float>(2.0f);
    (void)g.NewOperation("Const", "B")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Only B should exist (A was never finished)
    CHECK(g.num_operations() == 1u);
    CHECK(g.HasOperation("B"));
}

TEST_CASE("OperationBuilder from frozen graph throws") {
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Freeze the graph
    g.freeze();
    
    // Creating new operation should throw
    auto t2 = tf_wrap::Tensor::FromScalar<float>(2.0f);
    CHECK_THROWS(g.NewOperation("Const", "B")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish());
}

// ============================================================================
// CRITICAL: Graph freeze completeness tests
// ============================================================================

TEST_CASE("is_frozen returns false initially") {
    tf_wrap::Graph g;
    CHECK_FALSE(g.is_frozen());
}

TEST_CASE("is_frozen returns true after freeze") {
    tf_wrap::Graph g;
    g.freeze();
    CHECK(g.is_frozen());
}

TEST_CASE("freeze is idempotent") {
    tf_wrap::Graph g;
    g.freeze();
    g.freeze();  // Should not throw or change state
    g.freeze();
    CHECK(g.is_frozen());
}

TEST_CASE("Read operations work after freeze") {
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    g.freeze();
    
    // All read operations should still work
    CHECK(g.valid());
    CHECK(g.is_frozen());
    CHECK(g.num_operations() == 1u);
    CHECK(g.HasOperation("A"));
    CHECK(g.GetOperation("A").has_value());
    CHECK(g.GetOperationOrThrow("A") != nullptr);
    CHECK(g.GetAllOperations().size() == 1u);
    CHECK(!g.DebugString().empty());
    CHECK(g.handle() != nullptr);
}

TEST_CASE("ImportGraphDef on frozen graph throws") {
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Get a GraphDef to import
    auto graphdef = g.ToGraphDef();
    
    // Freeze the graph
    g.freeze();
    
    // ImportGraphDef should throw
    CHECK_THROWS(g.ImportGraphDef(graphdef.data(), graphdef.size(), "imported_"));
}


// ============================================================================
// HIGH: Session/Graph lifetime tests
// ============================================================================

TEST_CASE("Multiple Sessions from same Graph works") {
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(42.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Session s1(g);
    tf_wrap::Session s2(g);
    tf_wrap::Session s3(g);
    
    // All sessions should work
    auto r1 = s1.Run("A", 0);
    auto r2 = s2.Run("A", 0);
    auto r3 = s3.Run("A", 0);
    
    CHECK(r1.ToScalar<float>() == 42.0f);
    CHECK(r2.ToScalar<float>() == 42.0f);
    CHECK(r3.ToScalar<float>() == 42.0f);
}

TEST_CASE("Session still works after Graph frozen") {
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(42.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Session s(g);
    
    // Graph is now frozen
    CHECK(g.is_frozen());
    
    // Session should still work
    auto result = s.Run("A", 0);
    CHECK(result.ToScalar<float>() == 42.0f);
}

// ============================================================================
// HIGH: Session Run edge cases
// ============================================================================

TEST_CASE("Session Run with empty fetches") {
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Session s(g);
    
    // Real TensorFlow requires at least one fetch or target and throws INVALID_ARGUMENT.
    // Stub mode returns empty results. Both behaviors are acceptable.
    std::vector<tf_wrap::Fetch> empty_fetches;
    try {
        auto results = s.Run({}, empty_fetches, {});
        // Stub mode: returns empty
        CHECK(results.empty());
    } catch (const std::runtime_error& e) {
        // Real TF: throws INVALID_ARGUMENT
        std::string msg = e.what();
        CHECK(msg.find("INVALID_ARGUMENT") != std::string::npos);
    }
}

TEST_CASE("Session Run with non-existent operation throws") {
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Session s(g);
    
    CHECK_THROWS(s.Run("NonExistent", 0));
}

TEST_CASE("Session Run with non-existent feed operation throws") {
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Session s(g);
    
    auto feed_tensor = tf_wrap::Tensor::FromScalar<float>(2.0f);
    std::vector<tf_wrap::Feed> feeds = {
        tf_wrap::Feed{"NonExistent", 0, feed_tensor.handle()}
    };
    std::vector<tf_wrap::Fetch> fetches = {tf_wrap::Fetch{"A", 0}};
    
    CHECK_THROWS(s.Run(feeds, fetches, {}));
}

// ============================================================================
// HIGH: OperationBuilder error paths
// ============================================================================

TEST_CASE("OperationBuilder duplicate name handled") {
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Creating another op with same name - TF behavior varies
    // At minimum it should not crash
    auto t2 = tf_wrap::Tensor::FromScalar<float>(2.0f);
    try {
        (void)g.NewOperation("Const", "A")
            .SetAttrTensor("value", t2.handle())
            .SetAttrType("dtype", TF_FLOAT)
            .Finish();
        // If it succeeded, there should be an op (TF may rename it)
    } catch (const std::runtime_error&) {
        // Throwing is also acceptable
    }
    
    // Graph should still be valid
    CHECK(g.valid());
}

// ============================================================================
// MEDIUM: Consistent moved-from contracts
// ============================================================================

TEST_CASE("All movable types: moved-from has defined behavior") {
    // Status
    {
        tf_wrap::Status s1;
        s1.set(TF_INVALID_ARGUMENT, "test");
        tf_wrap::Status s2(std::move(s1));
        CHECK(s2.code() == TF_INVALID_ARGUMENT);
        // s1 is moved-from but safe to query
    }
    
    // Tensor
    {
        auto t1 = tf_wrap::Tensor::FromScalar<float>(1.0f);
        auto t2 = std::move(t1);
        CHECK(t2.handle() != nullptr);
        CHECK(t1.handle() == nullptr);
    }
    
    // Graph
    {
        tf_wrap::Graph g1;
        tf_wrap::Graph g2(std::move(g1));
        CHECK(g2.valid());
        CHECK(!g1.valid());
    }
    
    // SessionOptions
    {
        tf_wrap::SessionOptions o1;
        tf_wrap::SessionOptions o2(std::move(o1));
        CHECK(o2.get() != nullptr);
        CHECK(o1.get() == nullptr);
    }
    
    // Buffer
    {
        tf_wrap::Buffer b1;  // Default constructor
        tf_wrap::Buffer b2(std::move(b1));
        CHECK(b2.get() != nullptr);
        CHECK(b1.get() == nullptr);
    }
}

TEST_CASE("Types with valid(): moved-from is not valid") {
    // Graph
    {
        tf_wrap::Graph g1;
        tf_wrap::Graph g2(std::move(g1));
        CHECK(!g1.valid());
    }
}

TEST_CASE("Types with handle(): moved-from has null handle") {
    // Tensor
    {
        auto t1 = tf_wrap::Tensor::FromScalar<float>(1.0f);
        auto t2 = std::move(t1);
        CHECK(t1.handle() == nullptr);
    }
    
    // Graph
    {
        tf_wrap::Graph g1;
        tf_wrap::Graph g2(std::move(g1));
        CHECK(g1.handle() == nullptr);
    }
    
    // Session
    {
        tf_wrap::Graph g;
        auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
        (void)g.NewOperation("Const", "A")
            .SetAttrTensor("value", t.handle())
            .SetAttrType("dtype", TF_FLOAT)
            .Finish();
        
        tf_wrap::Session s1(g);
        tf_wrap::Session s2(std::move(s1));
        CHECK(s1.handle() == nullptr);
    }
    
    // SessionOptions
    {
        tf_wrap::SessionOptions o1;
        tf_wrap::SessionOptions o2(std::move(o1));
        CHECK(o1.get() == nullptr);
    }
    
    // Buffer
    {
        tf_wrap::Buffer b1;  // Default constructor
        tf_wrap::Buffer b2(std::move(b1));
        CHECK(b1.get() == nullptr);
    }
}

// ============================================================================
// MEDIUM: Thread safety claims verification
// ============================================================================

TEST_CASE("Graph concurrent operation creation" * doctest::test_suite("stress")) {
    tf_wrap::Graph g;
    std::atomic<int> op_count{0};
    std::vector<std::thread> threads;
    
    // Multiple threads try to add operations
    // Note: Without locking, this tests that concurrent access doesn't crash
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&g, &op_count, i] {
            auto t = tf_wrap::Tensor::FromScalar<float>(static_cast<float>(i));
            std::string name = "Op" + std::to_string(i);
            try {
                (void)g.NewOperation("Const", name)
                    .SetAttrTensor("value", t.handle())
                    .SetAttrType("dtype", TF_FLOAT)
                    .Finish();
                op_count.fetch_add(1);
            } catch (...) {
                // May throw if graph gets frozen, that's OK
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // All operations that succeeded should be in the graph
    CHECK(g.num_operations() == static_cast<std::size_t>(op_count.load()));
}

TEST_CASE("Graph allows concurrent reads" * doctest::test_suite("stress")) {
    tf_wrap::Graph g;
    auto tensor = tf_wrap::Tensor::FromScalar<float>(42.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    g.freeze();  // Prevent writes
    
    std::atomic<int> read_count{0};
    std::vector<std::thread> threads;
    
    // Multiple threads read concurrently
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&g, &read_count] {
            for (int j = 0; j < 100; ++j) {
                CHECK(g.HasOperation("A"));
                CHECK(g.num_operations() == 1u);
                read_count.fetch_add(1);
            }
        });
    }
    
    for (auto& th : threads) {
        th.join();
    }
    
    CHECK(read_count.load() == 1000);
}

TEST_CASE("Session Run is thread-safe (TensorFlow guarantee)" * doctest::test_suite("stress")) {
    tf_wrap::Graph g;
    auto tensor = tf_wrap::Tensor::FromScalar<float>(42.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Session s(g);
    
    std::atomic<int> run_count{0};
    std::vector<std::thread> threads;
    
    // Multiple threads run concurrently - this is safe per TensorFlow's API
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&s, &run_count] {
            for (int j = 0; j < 10; ++j) {
                auto result = s.Run("A", 0);
                CHECK(result.ToScalar<float>() == 42.0f);
                run_count.fetch_add(1);
            }
        });
    }
    
    for (auto& th : threads) {
        th.join();
    }
    
    CHECK(run_count.load() == 100);
}

// ============================================================================
// MEDIUM: Error message quality tests
// ============================================================================

TEST_CASE("Error messages include operation name") {
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "MyConstOp")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Session s(g);
    
    try {
        (void)s.Run("NonExistentOp", 0);
        CHECK(false);  // Should have thrown
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        CHECK(msg.find("NonExistentOp") != std::string::npos);
    }
}

TEST_CASE("Frozen graph error is actionable") {
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    g.freeze();
    
    try {
        auto t2 = tf_wrap::Tensor::FromScalar<float>(2.0f);
        (void)g.NewOperation("Const", "B")
            .SetAttrTensor("value", t2.handle())
            .SetAttrType("dtype", TF_FLOAT)
            .Finish();
        CHECK(false);
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        // Error should mention frozen/immutable and Session
        CHECK((msg.find("frozen") != std::string::npos || 
                 msg.find("immutable") != std::string::npos));
    }
}

TEST_CASE("Moved-from graph error is actionable") {
    tf_wrap::Graph g1;
    tf_wrap::Graph g2(std::move(g1));
    
    try {
        (void)g1.num_operations();
        CHECK(false);
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        CHECK(msg.find("moved") != std::string::npos);
    }
}

// ============================================================================
// Cross-Policy Tests (use only Const - stub supports this)
// ============================================================================

TEST_CASE("cross policy safe session with fast graph") {
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromVector<float>({2}, {1.0f, 2.0f});
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"X", 0}}, {});
    CHECK(results[0].ToVector<float>().size() == 2);
}

TEST_CASE("cross policy fast session with safe graph") {
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"X", 0}}, {});
    CHECK(results[0].ToVector<float>().size() == 3);
}

TEST_CASE("cross policy shared tensor with fast graph") {
    auto tensor = tf_wrap::Tensor::FromVector<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    
    tf_wrap::Graph g;
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"X", 0}}, {});
    auto v = results[0].ToVector<float>();
    CHECK(v[0] == 1.0f);
    CHECK(v[3] == 4.0f);
}

TEST_CASE("cross policy safe tensor with safe graph and fast session") {
    auto tensor = tf_wrap::Tensor::FromScalar<int32_t>(42);
    
    tf_wrap::Graph g;
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"X", 0}}, {});
    CHECK(results[0].ToScalar<int32_t>() == 42);
}

// ============================================================================
// Multi-Session Tests (using Const - stub safe)
// ============================================================================

TEST_CASE("multiple sessions same graph const only") {
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(42.0f);
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Create multiple sessions from same graph
    tf_wrap::Session s1(g);
    tf_wrap::Session s2(g);
    tf_wrap::Session s3(g);
    tf_wrap::Session s4(g);
    
    // All should work independently
    auto r1 = s1.Run({}, {{"X", 0}}, {});
    auto r2 = s2.Run({}, {{"X", 0}}, {});
    auto r3 = s3.Run({}, {{"X", 0}}, {});
    auto r4 = s4.Run({}, {{"X", 0}}, {});
    
    CHECK(r1[0].ToScalar<float>() == 42.0f);
    CHECK(r2[0].ToScalar<float>() == 42.0f);
    CHECK(r3[0].ToScalar<float>() == 42.0f);
    CHECK(r4[0].ToScalar<float>() == 42.0f);
}

// ============================================================================
// Identity Operation Tests (stub supports Identity)
// ============================================================================

TEST_CASE("identity operation passthrough") {
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    (void)g.NewOperation("Const", "Input")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* input_op = g.GetOperationOrThrow("Input");
    (void)g.NewOperation("Identity", "Output")
        .AddInput(tf_wrap::Output(input_op, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Output", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    CHECK(v.size() == 3);
    CHECK(v[0] == 1.0f);
    CHECK(v[1] == 2.0f);
    CHECK(v[2] == 3.0f);
}

// ============================================================================
// Add/Mul Operation Tests (stub supports Add and Mul)
// ============================================================================

TEST_CASE("add operation two constants") {
    tf_wrap::Graph g;
    auto t1 = tf_wrap::Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
    auto t2 = tf_wrap::Tensor::FromVector<float>({3}, {10.0f, 20.0f, 30.0f});
    
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    (void)g.NewOperation("Const", "B")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* a = g.GetOperationOrThrow("A");
    auto* b = g.GetOperationOrThrow("B");
    (void)g.NewOperation("Add", "Sum")
        .AddInput(tf_wrap::Output(a, 0))
        .AddInput(tf_wrap::Output(b, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Sum", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    CHECK(v.size() == 3);
    CHECK(v[0] == 11.0f);
    CHECK(v[1] == 22.0f);
    CHECK(v[2] == 33.0f);
}

TEST_CASE("mul operation two constants") {
    tf_wrap::Graph g;
    auto t1 = tf_wrap::Tensor::FromVector<float>({3}, {2.0f, 3.0f, 4.0f});
    auto t2 = tf_wrap::Tensor::FromVector<float>({3}, {10.0f, 10.0f, 10.0f});
    
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    (void)g.NewOperation("Const", "B")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    auto* a = g.GetOperationOrThrow("A");
    auto* b = g.GetOperationOrThrow("B");
    (void)g.NewOperation("Mul", "Product")
        .AddInput(tf_wrap::Output(a, 0))
        .AddInput(tf_wrap::Output(b, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"Product", 0}}, {});
    
    auto v = results[0].ToVector<float>();
    CHECK(v.size() == 3);
    CHECK(v[0] == 20.0f);
    CHECK(v[1] == 30.0f);
    CHECK(v[2] == 40.0f);
}

TEST_CASE("chained add operations") {
    tf_wrap::Graph g;
    auto t1 = tf_wrap::Tensor::FromScalar<int32_t>(1);
    auto t2 = tf_wrap::Tensor::FromScalar<int32_t>(2);
    auto t3 = tf_wrap::Tensor::FromScalar<int32_t>(3);
    
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    (void)g.NewOperation("Const", "B")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    (void)g.NewOperation("Const", "C")
        .SetAttrTensor("value", t3.handle())
        .SetAttrType("dtype", TF_INT32)
        .Finish();
    
    auto* a = g.GetOperationOrThrow("A");
    auto* b = g.GetOperationOrThrow("B");
    auto* c = g.GetOperationOrThrow("C");
    
    // A + B
    (void)g.NewOperation("Add", "AB")
        .AddInput(tf_wrap::Output(a, 0))
        .AddInput(tf_wrap::Output(b, 0))
        .Finish();
    
    // (A + B) + C
    auto* ab = g.GetOperationOrThrow("AB");
    (void)g.NewOperation("Add", "ABC")
        .AddInput(tf_wrap::Output(ab, 0))
        .AddInput(tf_wrap::Output(c, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    auto results = s.Run({}, {{"ABC", 0}}, {});
    
    CHECK(results[0].ToScalar<int32_t>() == 6);  // 1 + 2 + 3 = 6
}

// ============================================================================
// Concurrent Multi-Session Stress Test (Const only - stub safe)
// ============================================================================

TEST_CASE("multiple sessions concurrent const only" * doctest::test_suite("stress")) {
    tf_wrap::Graph g;
    auto t = tf_wrap::Tensor::FromScalar<float>(42.0f);
    (void)g.NewOperation("Const", "X")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    std::vector<std::unique_ptr<tf_wrap::Session>> sessions;
    for (int i = 0; i < 4; ++i) {
        sessions.push_back(std::make_unique<tf_wrap::Session>(g));
    }
    
    std::atomic<int> success_count{0};
    
    auto worker = [&](int id) {
        auto& session = *sessions[id];
        for (int i = 0; i < 100; ++i) {
            auto results = session.Run({}, {{"X", 0}}, {});
            if (results[0].ToScalar<float>() == 42.0f) {
                ++success_count;
            }
        }
    };
    
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back(worker, i);
    }
    for (auto& th : threads) {
        th.join();
    }
    
    CHECK(success_count == 400);
}

TEST_CASE("multiple sessions same graph with placeholder and square" * doctest::test_suite("stress")) {
    tf_wrap::Graph g;
    
    (void)g.NewOperation("Placeholder", "X")
        .SetAttrType("dtype", TF_FLOAT)
        .SetAttrShape("shape", {})
        .Finish();
    
    auto* x = g.GetOperationOrThrow("X");
    (void)g.NewOperation("Square", "Y")
        .AddInput(tf_wrap::Output(x, 0))
        .Finish();
    
    std::vector<std::unique_ptr<tf_wrap::Session>> sessions;
    for (int i = 0; i < 4; ++i) {
        sessions.push_back(std::make_unique<tf_wrap::Session>(g));
    }
    
    std::atomic<int> success_count{0};
    
    auto worker = [&](int id) {
        auto& session = *sessions[id];
        for (int i = 0; i < 100; ++i) {
            float val = static_cast<float>(id * 100 + i);
            auto input = tf_wrap::Tensor::FromScalar<float>(val);
            auto results = session.Run(
                {{"X", 0, input.handle()}},
                {{"Y", 0}},
                {}
            );
            if (std::abs(results[0].ToScalar<float>() - val * val) < 0.01f) {
                ++success_count;
            }
        }
    };
    
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back(worker, i);
    }
    for (auto& th : threads) {
        th.join();
    }
    
    CHECK(success_count == 400);
}

TEST_CASE("session recovers after shape error with matmul") {
    tf_wrap::Graph g;
    
    (void)g.NewOperation("Placeholder", "X")
        .SetAttrType("dtype", TF_FLOAT)
        .SetAttrShape("shape", {2, 2})
        .Finish();
    (void)g.NewOperation("Placeholder", "Y")
        .SetAttrType("dtype", TF_FLOAT)
        .SetAttrShape("shape", {2, 2})
        .Finish();
    
    auto* x_op = g.GetOperationOrThrow("X");
    auto* y_op = g.GetOperationOrThrow("Y");
    
    (void)g.NewOperation("MatMul", "Result")
        .AddInput(tf_wrap::Output(x_op, 0))
        .AddInput(tf_wrap::Output(y_op, 0))
        .Finish();
    
    tf_wrap::Session s(g);
    
    // First cause an error with incompatible shapes (3x3 vs 2x2)
    auto bad_x = tf_wrap::Tensor::FromVector<float>({3, 3}, std::vector<float>(9, 1.0f));
    auto bad_y = tf_wrap::Tensor::FromVector<float>({2, 2}, std::vector<float>(4, 1.0f));
    
    bool threw = false;
    try {
        (void)s.Run(
            {{"X", 0, bad_x.handle()}, {"Y", 0, bad_y.handle()}},
            {{"Result", 0}},
            {}
        );
    } catch (...) {
        threw = true;
    }
    CHECK(threw);
    
    // Now run with correct shapes - should succeed
    // Identity matrix multiplication: [[1,0],[0,1]] * [[1,2],[3,4]] = [[1,2],[3,4]]
    auto good_x = tf_wrap::Tensor::FromVector<float>({2, 2}, {1, 0, 0, 1});
    auto good_y = tf_wrap::Tensor::FromVector<float>({2, 2}, {1, 2, 3, 4});
    
    auto results = s.Run(
        {{"X", 0, good_x.handle()}, {"Y", 0, good_y.handle()}},
        {{"Result", 0}},
        {}
    );
    
    CHECK(results.size() == 1);
    auto v = results[0].ToVector<float>();
    CHECK(v[0] == 1.0f);  // (1,1)
    CHECK(v[1] == 2.0f);  // (1,2)
    CHECK(v[2] == 3.0f);  // (2,1)
    CHECK(v[3] == 4.0f);  // (2,2)
}

// ============================================================================
// Tensor Shape and Type Tests (no Session::Run - stub safe)
// ============================================================================

TEST_CASE("tensor reshape preserves data") {
    auto original = tf_wrap::Tensor::FromVector<float>({2, 3}, 
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    
    CHECK(original.shape().size() == 2);
    CHECK(original.shape()[0] == 2);
    CHECK(original.shape()[1] == 3);
    CHECK(original.num_elements() == 6);
    
    auto v = original.ToVector<float>();
    CHECK(v[0] == 1.0f);
    CHECK(v[5] == 6.0f);
}

TEST_CASE("tensor different dtypes") {
    auto f32 = tf_wrap::Tensor::FromScalar<float>(1.5f);
    auto f64 = tf_wrap::Tensor::FromScalar<double>(2.5);
    auto i32 = tf_wrap::Tensor::FromScalar<int32_t>(42);
    auto i64 = tf_wrap::Tensor::FromScalar<int64_t>(123456789LL);
    auto u8 = tf_wrap::Tensor::FromScalar<uint8_t>(255);
    
    CHECK(f32.dtype() == TF_FLOAT);
    CHECK(f64.dtype() == TF_DOUBLE);
    CHECK(i32.dtype() == TF_INT32);
    CHECK(i64.dtype() == TF_INT64);
    CHECK(u8.dtype() == TF_UINT8);
    
    CHECK(f32.ToScalar<float>() == 1.5f);
    CHECK(f64.ToScalar<double>() == 2.5);
    CHECK(i32.ToScalar<int32_t>() == 42);
    CHECK(i64.ToScalar<int64_t>() == 123456789LL);
    CHECK(u8.ToScalar<uint8_t>() == 255);
}

TEST_CASE("tensor multidimensional shapes") {
    auto t1d = tf_wrap::Tensor::FromVector<float>({5}, {1, 2, 3, 4, 5});
    auto t2d = tf_wrap::Tensor::FromVector<float>({2, 3}, {1, 2, 3, 4, 5, 6});
    auto t3d = tf_wrap::Tensor::FromVector<float>({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    auto t4d = tf_wrap::Tensor::FromVector<float>({2, 2, 2, 2}, 
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    
    CHECK(t1d.shape().size() == 1);
    CHECK(t2d.shape().size() == 2);
    CHECK(t3d.shape().size() == 3);
    CHECK(t4d.shape().size() == 4);
    
    CHECK(t1d.num_elements() == 5);
    CHECK(t2d.num_elements() == 6);
    CHECK(t3d.num_elements() == 8);
    CHECK(t4d.num_elements() == 16);
}

TEST_CASE("tensor clone independence") {
    auto original = tf_wrap::Tensor::FromVector<float>({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto clone1 = original.Clone();
    auto clone2 = original.Clone();
    
    // Modify clone1
    {
        auto view = clone1.write<float>();
        view[0] = 100.0f;
    }
    
    // Modify clone2
    {
        auto view = clone2.write<float>();
        view[0] = 200.0f;
    }
    
    // All should be independent
    CHECK(original.ToVector<float>()[0] == 1.0f);
    CHECK(clone1.ToVector<float>()[0] == 100.0f);
    CHECK(clone2.ToVector<float>()[0] == 200.0f);
}

// ============================================================================
// Interleaved View Tests (no Session::Run - stub safe)
// ============================================================================

TEST_CASE("tensor interleaved read write views") {
    auto tensor = tf_wrap::Tensor::FromVector<float>({10}, 
        {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    
    {
        auto write_view = tensor.write<float>();
        write_view[0] = 100.0f;
    }
    
    {
        auto read_view = tensor.read<float>();
        CHECK(read_view[0] == 100.0f);
    }
    
    for (int i = 0; i < 10; ++i) {
        {
            auto write_view = tensor.write<float>();
            write_view[i] = static_cast<float>(i * 10);
        }
        {
            auto read_view = tensor.read<float>();
            CHECK(read_view[i] == static_cast<float>(i * 10));
        }
    }
}

TEST_CASE("tensor concurrent read views") {
    auto tensor = tf_wrap::Tensor::FromVector<float>({1000},
        std::vector<float>(1000, 42.0f));
    
    std::atomic<int> success_count{0};
    
    auto reader = [&]() {
        for (int i = 0; i < 100; ++i) {
            auto view = tensor.read<float>();
            bool all_correct = true;
            for (size_t j = 0; j < view.size(); ++j) {
                if (view[j] != 42.0f) {
                    all_correct = false;
                    break;
                }
            }
            if (all_correct) ++success_count;
        }
    };
    
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back(reader);
    }
    for (auto& t : threads) {
        t.join();
    }
    
    CHECK(success_count == 800);
}

// ============================================================================
// AdoptMalloc Success Tests (no Session::Run - stub safe)
// ============================================================================

TEST_CASE("adopt malloc success basic") {
    std::vector<int64_t> shape = {2, 3};
    size_t num_elements = 6;
    size_t byte_size = num_elements * sizeof(float);
    
    float* data = static_cast<float*>(std::malloc(byte_size));
    CHECK(data != nullptr);
    
    for (size_t i = 0; i < num_elements; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    auto tensor = tf_wrap::Tensor::AdoptMalloc<float>(shape, data, byte_size);
    
    CHECK(tensor.valid());
    CHECK(tensor.num_elements() == 6);
    CHECK(tensor.dtype() == TF_FLOAT);
    
    auto v = tensor.ToVector<float>();
    CHECK(v[0] == 0.0f);
    CHECK(v[5] == 5.0f);
}

TEST_CASE("adopt malloc success large tensor") {
    std::vector<int64_t> shape = {1000, 1000};
    size_t num_elements = 1000000;
    size_t byte_size = num_elements * sizeof(float);
    
    float* data = static_cast<float*>(std::malloc(byte_size));
    CHECK(data != nullptr);
    
    for (size_t i = 0; i < num_elements; ++i) {
        data[i] = static_cast<float>(i % 100);
    }
    
    auto tensor = tf_wrap::Tensor::AdoptMalloc<float>(shape, data, byte_size);
    
    CHECK(tensor.shape()[0] == 1000);
    CHECK(tensor.shape()[1] == 1000);
    
    auto v = tensor.ToVector<float>();
    CHECK(v[0] == 0.0f);
    CHECK(v[99] == 99.0f);
    CHECK(v[100] == 0.0f);
}

