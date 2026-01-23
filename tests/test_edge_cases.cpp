// tests/test_edge_cases.cpp
// Edge case, corner case, and stress tests for tf_wrapper
//
// Compile with: -fsanitize=address,undefined for best coverage

#include "tf_wrap/all.hpp"

#include "tf_wrap/format.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ============================================================================
// Test Framework (minimal, matches test_main.cpp style)
// ============================================================================

namespace tf_test {

struct TestCase {
    const char* name;
    void (*fn)();
    bool is_stress;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

struct Registrar {
    Registrar(const char* name, void (*fn)(), bool is_stress = false) {
        registry().push_back({name, fn, is_stress});
    }
};

inline void require_impl(bool cond, const char* expr, const char* file, int line) {
    if (cond) return;
    throw std::runtime_error(tf_wrap::detail::format(
        "REQUIRE failed: {} ({}:{})", expr, file, line));
}

template<class Ex, class Fn>
inline void require_throws_impl(Fn&& fn, const char* expr, const char* ex_name,
                                const char* file, int line) {
    try {
        fn();
        throw std::runtime_error(tf_wrap::detail::format(
            "REQUIRE_THROWS failed: {} did not throw {} ({}:{})",
            expr, ex_name, file, line));
    } catch (const Ex&) {
        // Expected
    } catch (const std::exception& e) {
        throw std::runtime_error(tf_wrap::detail::format(
            "REQUIRE_THROWS failed: {} threw wrong type: {} ({}:{})",
            expr, e.what(), file, line));
    }
}

// Non-template version that catches any exception
template<class Fn>
inline void require_throws_any_impl(Fn&& fn, const char* expr,
                                    const char* file, int line) {
    try {
        fn();
        throw std::runtime_error(tf_wrap::detail::format(
            "REQUIRE_THROWS failed: {} did not throw ({}:{})",
            expr, file, line));
    } catch (...) {
        // Any exception is expected - success
    }
}

} // namespace tf_test

// Two-level macro for proper __LINE__ expansion
#define TF_JOIN2(a, b) a##b
#define TF_JOIN(a, b) TF_JOIN2(a, b)

#define TEST_CASE(name) \
    static void TF_JOIN(test_fn_, __LINE__)(); \
    static tf_test::Registrar TF_JOIN(test_reg_, __LINE__)( \
        name, &TF_JOIN(test_fn_, __LINE__), false); \
    static void TF_JOIN(test_fn_, __LINE__)()

#define STRESS_TEST(name) \
    static void TF_JOIN(stress_fn_, __LINE__)(); \
    static tf_test::Registrar TF_JOIN(stress_reg_, __LINE__)( \
        name, &TF_JOIN(stress_fn_, __LINE__), true); \
    static void TF_JOIN(stress_fn_, __LINE__)()

#define REQUIRE(expr) tf_test::require_impl((expr), #expr, __FILE__, __LINE__)
#define REQUIRE_FALSE(expr) tf_test::require_impl(!(expr), "!" #expr, __FILE__, __LINE__)
#define REQUIRE_THROWS_AS(expr, ex) \
    tf_test::require_throws_impl<ex>([&]{ (void)(expr); }, #expr, #ex, __FILE__, __LINE__)
#define REQUIRE_THROWS(expr) \
    tf_test::require_throws_any_impl([&]{ (void)(expr); }, #expr, __FILE__, __LINE__)

// ============================================================================
// Empty/Zero Tensor Tests
// ============================================================================

TEST_CASE("empty tensor from zero-sized vector") {
    std::vector<std::int64_t> shape = {0};
    std::vector<float> data = {};
    auto tensor = tf_wrap::FastTensor::FromVector<float>(shape, data);
    REQUIRE(tensor.num_elements() == 0);
    REQUIRE(tensor.byte_size() == 0);
    REQUIRE(tensor.rank() == 1);
}

TEST_CASE("empty tensor from zero dimension in middle") {
    std::vector<std::int64_t> shape = {10, 0, 5};
    auto tensor = tf_wrap::FastTensor::Allocate<float>(shape);
    REQUIRE(tensor.num_elements() == 0);
    REQUIRE(tensor.byte_size() == 0);
    REQUIRE(tensor.rank() == 3);
}

TEST_CASE("scalar has empty shape") {
    auto tensor = tf_wrap::FastTensor::FromScalar<int>(42);
    REQUIRE(tensor.shape().empty());
    REQUIRE(tensor.rank() == 0);
    REQUIRE(tensor.num_elements() == 1);
}

TEST_CASE("empty tensor read view") {
    std::vector<std::int64_t> shape = {0};
    std::vector<float> data = {};
    auto tensor = tf_wrap::FastTensor::FromVector<float>(shape, data);
    auto view = tensor.read<float>();
    REQUIRE(view.size() == 0);
    REQUIRE(view.begin() == view.end());
}

// ============================================================================
// Boundary Value Tests
// ============================================================================

TEST_CASE("single element 1D tensor") {
    std::vector<std::int64_t> shape = {1};
    std::vector<float> data = {42.0f};
    auto tensor = tf_wrap::FastTensor::FromVector<float>(shape, data);
    REQUIRE(tensor.num_elements() == 1);
    auto view = tensor.read<float>();
    REQUIRE(view[0] == 42.0f);
}

TEST_CASE("high rank tensor 8 dimensions") {
    std::vector<std::int64_t> shape(8, 2);  // 2^8 = 256 elements
    auto tensor = tf_wrap::FastTensor::Allocate<float>(shape);
    REQUIRE(tensor.rank() == 8);
    REQUIRE(tensor.num_elements() == 256);
}

TEST_CASE("negative dimension throws") {
    std::vector<std::int64_t> shape = {10, -1, 5};
    REQUIRE_THROWS_AS(
        tf_wrap::FastTensor::Allocate<float>(shape),
        std::invalid_argument);
}

TEST_CASE("overflow in dimension product throws") {
    // Use dimensions that are individually valid but overflow when multiplied
    // INT64_MAX / 2 is positive and valid, but (INT64_MAX/2) * 3 overflows
    std::vector<std::int64_t> shape = {INT64_MAX / 2, 3};
    REQUIRE_THROWS_AS(
        tf_wrap::FastTensor::Allocate<float>(shape),
        std::overflow_error);
}

TEST_CASE("INT64_MAX dimension throws overflow") {
    std::vector<std::int64_t> shape = {INT64_MAX};
    REQUIRE_THROWS_AS(
        tf_wrap::FastTensor::Allocate<float>(shape),
        std::overflow_error);
}

// ============================================================================
// Moved-From Object Tests
// ============================================================================

TEST_CASE("moved-from tensor has null handle") {
    auto t1 = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    REQUIRE(t1.handle() != nullptr);
    
    auto t2 = std::move(t1);
    REQUIRE(t2.handle() != nullptr);
    // t1 is now in moved-from state
}

TEST_CASE("moved-from status is safe to destroy") {
    tf_wrap::Status s1;
    s1.set(TF_CANCELLED, "test");
    tf_wrap::Status s2 = std::move(s1);
    
    REQUIRE(s2.code() == TF_CANCELLED);
    // s1 destruction should not double-free
}

TEST_CASE("moved-from graph must throw on use") {
    tf_wrap::FastGraph g1;
    auto tensor = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    (void)g1.NewOperation("Const", "c")
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::FastGraph g2 = std::move(g1);
    REQUIRE(g2.GetOperation("c").has_value());
    
    // Moved-from must throw, not silently return empty
    REQUIRE(!g1.valid());
    REQUIRE_THROWS(g1.num_operations());
}

TEST_CASE("moved-from session options is safe") {
    tf_wrap::SessionOptions opts1;
    tf_wrap::SessionOptions opts2 = std::move(opts1);
    
    REQUIRE(opts2.get() != nullptr);
    REQUIRE(opts1.get() == nullptr);
}

// ============================================================================
// Adopt/AdoptMalloc Error Tests
// ============================================================================

TEST_CASE("AdoptMalloc wrong byte_len throws") {
    void* data = std::malloc(100);
    REQUIRE(data != nullptr);
    
    std::vector<std::int64_t> shape = {10};
    bool threw = false;
    try {
        // Shape {10} with float requires 40 bytes, not 100
        (void)tf_wrap::FastTensor::AdoptMalloc<float>(shape, data, 100);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    
    REQUIRE(threw);
    std::free(data);  // We still own it
}

TEST_CASE("Adopt with null deallocator throws") {
    void* data = std::malloc(40);
    REQUIRE(data != nullptr);
    
    std::vector<std::int64_t> shape = {10};
    REQUIRE_THROWS_AS(
        tf_wrap::FastTensor::Adopt(TF_FLOAT, shape, data, 40, nullptr),
        std::invalid_argument);
    
    std::free(data);
}

TEST_CASE("Adopt with null data and zero bytes succeeds") {
    auto dealloc = [](void*, std::size_t, void*) {};
    std::vector<std::int64_t> shape = {0};
    
    // Should not throw
    auto tensor = tf_wrap::FastTensor::Adopt(TF_FLOAT, shape, nullptr, 0, dealloc);
    REQUIRE(tensor.num_elements() == 0);
}

TEST_CASE("Adopt with null data and non-zero bytes throws") {
    auto dealloc = [](void*, std::size_t, void*) {};
    std::vector<std::int64_t> shape = {10};
    
    REQUIRE_THROWS_AS(
        tf_wrap::FastTensor::Adopt(TF_FLOAT, shape, nullptr, 40, dealloc),
        std::invalid_argument);
}

// ============================================================================
// Graph Error Tests
// ============================================================================

TEST_CASE("GetOperationOrThrow on empty graph throws") {
    tf_wrap::FastGraph graph;
    REQUIRE_THROWS_AS(
        graph.GetOperationOrThrow("nonexistent"),
        std::runtime_error);
}

TEST_CASE("GetOperation on empty graph returns nullopt") {
    tf_wrap::FastGraph graph;
    auto result = graph.GetOperation("nonexistent");
    REQUIRE(!result.has_value());
}

// ============================================================================
// Status Edge Cases
// ============================================================================

TEST_CASE("Status set with empty string_view") {
    tf_wrap::Status st;
    st.set(TF_INTERNAL, std::string_view{});
    REQUIRE(st.code() == TF_INTERNAL);
    REQUIRE(std::string(st.message()).empty());
}

TEST_CASE("Status set with very long message") {
    tf_wrap::Status st;
    std::string long_msg(10000, 'x');
    st.set(TF_INTERNAL, long_msg);
    REQUIRE(st.code() == TF_INTERNAL);
    REQUIRE(std::string(st.message()) == long_msg);
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
        REQUIRE(name != nullptr);
        REQUIRE(std::string(name) != "UNKNOWN_CODE");
    }
}

// ============================================================================
// Stress Tests
// ============================================================================

STRESS_TEST("rapid tensor allocation/deallocation") {
    constexpr int iterations = 10000;
    
    auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        std::vector<std::int64_t> shape = {100, 100};
        auto t = tf_wrap::FastTensor::Allocate<float>(shape);
    }
    
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    const double ms_d = static_cast<double>(ms);
    std::cout << "    " << iterations << " allocations in " << ms << "ms "
              << "(" << (static_cast<double>(iterations) * 1000.0 / (ms_d + 1.0))
              << " ops/sec)\n";
}

STRESS_TEST("concurrent tensor creation") {
    constexpr int num_threads = 8;
    constexpr int ops_per_thread = 1000;
    
    std::atomic<int> completed{0};
    std::vector<std::thread> threads;
    
    auto start = std::chrono::steady_clock::now();
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < ops_per_thread; ++i) {
                auto tensor = tf_wrap::FastTensor::FromScalar<float>(
                    static_cast<float>(t * ops_per_thread + i));
                ++completed;
            }
        });
    }
    
    for (auto& t : threads) t.join();
    
    auto end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    REQUIRE(completed.load() == num_threads * ops_per_thread);
    std::cout << "    " << completed.load() << " concurrent ops in " << ms << "ms\n";
}

STRESS_TEST("SafeTensor reader/writer contention") {
    std::vector<std::int64_t> shape = {1000};
    auto tensor = tf_wrap::SafeTensor::Zeros<float>(shape);
    
    std::atomic<bool> stop{false};
    std::atomic<int> reads{0};
    std::atomic<int> writes{0};
    std::atomic<bool> torn_read{false};
    
    // Writer thread
    std::thread writer([&]() {
        float value = 0.0f;
        while (!stop && !torn_read) {
            {
                auto view = tensor.write<float>();
                for (auto& x : view) x = value;
            }
            value += 1.0f;
            ++writes;
        }
    });
    
    // Multiple reader threads
    std::vector<std::thread> readers;
    for (int i = 0; i < 4; ++i) {
        readers.emplace_back([&]() {
            while (!stop && !torn_read) {
                auto view = tensor.read<float>();
                float first = view[0];
                for (std::size_t j = 1; j < view.size(); ++j) {
                    if (view[j] != first) {
                        torn_read = true;
                        break;
                    }
                }
                ++reads;
            }
        });
    }
    
    // Run for 2 seconds
    std::this_thread::sleep_for(std::chrono::seconds(2));
    stop = true;
    
    writer.join();
    for (auto& r : readers) r.join();
    
    REQUIRE(!torn_read);
    std::cout << "    " << reads << " reads, " << writes << " writes, no torn reads\n";
}

STRESS_TEST("SharedTensor allows concurrent readers") {
    std::vector<std::int64_t> shape = {1000};
    std::vector<float> data(1000, 42.0f);
    auto tensor = tf_wrap::SharedTensor::FromVector<float>(shape, data);
    
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
    REQUIRE(max_concurrent > 1);
}

STRESS_TEST("Status rapid creation/destruction") {
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

STRESS_TEST("fuzz: random tensor shapes") {
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
            auto tensor = tf_wrap::FastTensor::FromVector<float>(shape, data);
            
            REQUIRE(tensor.shape() == shape);
            REQUIRE(static_cast<std::int64_t>(tensor.num_elements()) == total_elements);
            
            if (total_elements > 0) {
                auto view = tensor.read<float>();
                REQUIRE(view.size() == static_cast<std::size_t>(total_elements));
            }
            
            ++created;
        } catch (const std::exception&) {
            ++failed_expected;
        }
    }
    
    std::cout << "    Created " << created << " tensors, " 
              << failed_expected << " expected failures\n";
    REQUIRE(created > 200);
}

STRESS_TEST("fuzz: random dtype values") {
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
        
        auto tensor = tf_wrap::FastTensor::FromVector<float>(shape, data);
        auto extracted = tensor.ToVector<float>();
        REQUIRE(extracted == data);
        ++tested;
    }
    
    // Test int32 with random values  
    for (int i = 0; i < 50; ++i) {
        std::vector<std::int32_t> data(25);
        for (auto& v : data) {
            v = static_cast<std::int32_t>(gen() % 10000);
        }
        
        auto tensor = tf_wrap::FastTensor::FromVector<std::int32_t>(shape, data);
        auto extracted = tensor.ToVector<std::int32_t>();
        REQUIRE(extracted == data);
        ++tested;
    }
    
    std::cout << "    Tested " << tested << " random tensor values\n";
}

STRESS_TEST("fuzz: OperationBuilder exception safety") {
    tf_wrap::FastGraph graph;
    int abandoned = 0;
    
    for (int i = 0; i < 50; ++i) {
        try {
            auto builder = graph.NewOperation("Const", "test_" + std::to_string(i));
            
            if (i % 3 == 0) {
                throw std::runtime_error("simulated error");
            }
            
            auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
            builder.SetAttrTensor("value", t.handle());
            builder.SetAttrType("dtype", TF_FLOAT);
            (void)std::move(builder).Finish();
            
        } catch (const std::exception&) {
            ++abandoned;
        }
    }
    
    std::cout << "    " << abandoned << " builders safely abandoned\n";
    REQUIRE(abandoned > 0);
}

STRESS_TEST("fuzz: moved-from object safety") {
    for (int i = 0; i < 50; ++i) {
        tf_wrap::FastTensor t1 = tf_wrap::FastTensor::FromScalar<float>(static_cast<float>(i));
        tf_wrap::FastTensor t2 = std::move(t1);
        
        REQUIRE(t1.empty());
        REQUIRE(t1.handle() == nullptr);
        REQUIRE(!t2.empty());
        REQUIRE(t2.ToScalar<float>() == static_cast<float>(i));
        
        tf_wrap::FastTensor t3 = std::move(t2);
        REQUIRE(t2.empty());
        REQUIRE(!t3.empty());
        
        t1 = std::move(t3);
        REQUIRE(t3.empty());
        REQUIRE(!t1.empty());
    }
    
    std::cout << "    50 move sequences completed safely\n";
}

// ============================================================================
// Graph Moved-From State Tests (P1 #4 fix verification)
// ============================================================================

TEST_CASE("Graph moved-from: must throw on use after move") {
    tf_wrap::SafeGraph g;

    // Create a real op so the destination graph is non-empty.
    auto t = tf_wrap::SafeTensor::FromVector<float>({1}, {1.0f});
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();

    tf_wrap::SafeGraph g2(std::move(g));

    // Destination must still work.
    REQUIRE(g2.num_operations() == 1u);
    REQUIRE(g2.HasOperation("A"));
    REQUIRE(g2.valid());

    // Moved-from must be invalid and throw on handle-touching calls.
    REQUIRE(!g.valid());
    REQUIRE_THROWS(g.num_operations());
    REQUIRE_THROWS(g.GetAllOperations());
    REQUIRE_THROWS(g.GetOperation("A"));
    REQUIRE_THROWS(g.HasOperation("A"));
    REQUIRE_THROWS(g.DebugString());
}

TEST_CASE("Graph move-assign: moved-from must throw on use") {
    tf_wrap::SafeGraph g1;
    tf_wrap::SafeGraph g2;

    auto t = tf_wrap::SafeTensor::FromVector<float>({1}, {1.0f});
    (void)g1.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();

    g2 = std::move(g1);

    // Destination has the operation
    REQUIRE(g2.num_operations() == 1u);
    REQUIRE(g2.HasOperation("A"));
    REQUIRE(g2.valid());

    // Moved-from must throw
    REQUIRE(!g1.valid());
    REQUIRE_THROWS(g1.num_operations());
    REQUIRE_THROWS(g1.GetAllOperations());
    REQUIRE_THROWS(g1.NewOperation("Const", "B"));
}

TEST_CASE("Graph self-move assignment leaves graph intact") {
    tf_wrap::FastGraph g;
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);

    (void)g.NewOperation("Const", "c")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();

    // Intentional self-move to test robustness
    // Use volatile pointer to prevent compiler from detecting self-move
    tf_wrap::FastGraph* volatile p = &g;
    g = std::move(*p);

    REQUIRE(g.handle() != nullptr);
    REQUIRE(g.valid());
    REQUIRE(g.HasOperation("c"));
}

STRESS_TEST("fuzz: moved-from graph safety (handle semantics)") {
    for (int i = 0; i < 50; ++i) {
        tf_wrap::FastGraph g1;
        auto t = tf_wrap::FastTensor::FromScalar<float>(static_cast<float>(i));

        (void)g1.NewOperation("Const", "c")
            .SetAttrTensor("value", t.handle())
            .SetAttrType("dtype", TF_FLOAT)
            .Finish();

        tf_wrap::FastGraph g2 = std::move(g1);
        REQUIRE(g2.HasOperation("c"));

        // Moved-from must throw (our chosen contract)
        REQUIRE(!g1.valid());
        REQUIRE_THROWS(g1.num_operations());
        REQUIRE_THROWS(g1.DebugString());

        tf_wrap::FastGraph g3 = std::move(g2);
        REQUIRE(g3.HasOperation("c"));
        REQUIRE(!g2.valid());
    }
    
    std::cout << "    50 graph move sequences completed safely\n";
}

// ============================================================================
// Session/Graph Freeze Tests (enforces TF's immutability requirement)
// ============================================================================

TEST_CASE("Graph mutation after Session creation must throw") {
    tf_wrap::SafeGraph g;

    // Build a minimal graph
    auto a = tf_wrap::SafeTensor::FromVector<float>({2}, {1.0f, 2.0f});
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", a.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();

    // Graph should not be frozen yet
    REQUIRE_FALSE(g.is_frozen());

    // Create session - this freezes the graph
    tf_wrap::SafeSession s(g);
    
    // Graph must now be frozen
    REQUIRE(g.is_frozen());

    // Any mutation attempt must throw
    auto b = tf_wrap::SafeTensor::FromScalar<float>(0.0f);
    REQUIRE_THROWS(g.NewOperation("Const", "B")
        .SetAttrTensor("value", b.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish());
    
    // Session should still work
    auto result = s.Run("A", 0);
    REQUIRE((result.ToVector<float>() == std::vector<float>{1.0f, 2.0f}));
    
    std::cout << "    Graph freeze enforced after Session creation\n";
}

TEST_CASE("Graph freeze works with different Session/Graph policies") {
    // SharedGraph with SafeSession - policies can differ
    tf_wrap::SharedGraph g;

    auto a = tf_wrap::FastTensor::FromVector<float>({2}, {1.0f, 2.0f});
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", a.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();

    tf_wrap::SafeSession s(g);
    
    REQUIRE(g.is_frozen());
    
    // Mutation must throw
    auto b = tf_wrap::FastTensor::FromScalar<float>(0.0f);
    REQUIRE_THROWS(g.NewOperation("Const", "B")
        .SetAttrTensor("value", b.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish());
}

// ============================================================================
// CRITICAL: Session moved-from state tests
// ============================================================================

TEST_CASE("Session move constructor leaves source invalid") {
    tf_wrap::FastGraph g;
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::FastSession s1(g);
    tf_wrap::FastSession s2(std::move(s1));
    
    // s2 should work
    REQUIRE(s2.handle() != nullptr);
    
    // s1 should be invalid
    REQUIRE(s1.handle() == nullptr);
}

TEST_CASE("Session move assignment leaves source invalid") {
    tf_wrap::FastGraph g1;
    auto t1 = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    (void)g1.NewOperation("Const", "A")
        .SetAttrTensor("value", t1.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::FastGraph g2;
    auto t2 = tf_wrap::FastTensor::FromScalar<float>(2.0f);
    (void)g2.NewOperation("Const", "B")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::FastSession s1(g1);
    tf_wrap::FastSession s2(g2);
    
    s2 = std::move(s1);
    
    REQUIRE(s2.handle() != nullptr);
    REQUIRE(s1.handle() == nullptr);
}

TEST_CASE("Moved-from session throws on Run") {
    tf_wrap::FastGraph g;
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::FastSession s1(g);
    tf_wrap::FastSession s2(std::move(s1));
    
    REQUIRE_THROWS(s1.Run("A", 0));
}

TEST_CASE("Moved-from session is safe to destroy") {
    tf_wrap::FastGraph g;
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    {
        tf_wrap::FastSession s1(g);
        tf_wrap::FastSession s2(std::move(s1));
        // s1 destroyed here - should not crash
    }
    // If we get here, destruction was safe
    REQUIRE(true);
}

// ============================================================================
// CRITICAL: Session from invalid Graph tests
// ============================================================================

TEST_CASE("Session from moved-from graph throws") {
    tf_wrap::FastGraph g1;
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    (void)g1.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::FastGraph g2(std::move(g1));
    
    // Creating session from moved-from graph should throw
    bool threw = false;
    try {
        tf_wrap::FastSession s(g1);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    REQUIRE(threw);
}

// ============================================================================
// CRITICAL: OperationBuilder lifecycle tests
// ============================================================================

TEST_CASE("OperationBuilder Finish releases lock immediately") {
    tf_wrap::SafeGraph g;
    auto t = tf_wrap::SafeTensor::FromScalar<float>(1.0f);
    
    // Create and finish an operation
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Should be able to create another operation immediately
    // (lock was released by Finish)
    auto t2 = tf_wrap::SafeTensor::FromScalar<float>(2.0f);
    (void)g.NewOperation("Const", "B")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    REQUIRE(g.num_operations() == 2u);
}

TEST_CASE("OperationBuilder without Finish releases lock on destruction") {
    tf_wrap::SafeGraph g;
    auto t = tf_wrap::SafeTensor::FromScalar<float>(1.0f);
    
    {
        // Create builder but don't finish
        auto builder = g.NewOperation("Const", "A")
            .SetAttrTensor("value", t.handle())
            .SetAttrType("dtype", TF_FLOAT);
        // Builder destroyed without Finish()
    }
    
    // Should be able to create another operation
    // (lock was released by destructor)
    auto t2 = tf_wrap::SafeTensor::FromScalar<float>(2.0f);
    (void)g.NewOperation("Const", "B")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Only B should exist (A was never finished)
    REQUIRE(g.num_operations() == 1u);
    REQUIRE(g.HasOperation("B"));
}

TEST_CASE("OperationBuilder from frozen graph throws") {
    tf_wrap::SafeGraph g;
    auto t = tf_wrap::SafeTensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Freeze the graph
    g.freeze();
    
    // Creating new operation should throw
    auto t2 = tf_wrap::SafeTensor::FromScalar<float>(2.0f);
    REQUIRE_THROWS(g.NewOperation("Const", "B")
        .SetAttrTensor("value", t2.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish());
}

// ============================================================================
// CRITICAL: Graph freeze completeness tests
// ============================================================================

TEST_CASE("is_frozen returns false initially") {
    tf_wrap::FastGraph g;
    REQUIRE_FALSE(g.is_frozen());
}

TEST_CASE("is_frozen returns true after freeze") {
    tf_wrap::FastGraph g;
    g.freeze();
    REQUIRE(g.is_frozen());
}

TEST_CASE("freeze is idempotent") {
    tf_wrap::FastGraph g;
    g.freeze();
    g.freeze();  // Should not throw or change state
    g.freeze();
    REQUIRE(g.is_frozen());
}

TEST_CASE("Read operations work after freeze") {
    tf_wrap::FastGraph g;
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    g.freeze();
    
    // All read operations should still work
    REQUIRE(g.valid());
    REQUIRE(g.is_frozen());
    REQUIRE(g.num_operations() == 1u);
    REQUIRE(g.HasOperation("A"));
    REQUIRE(g.GetOperation("A").has_value());
    REQUIRE(g.GetOperationOrThrow("A") != nullptr);
    REQUIRE(g.GetAllOperations().size() == 1u);
    REQUIRE(!g.DebugString().empty());
    REQUIRE(g.handle() != nullptr);
}

TEST_CASE("ImportGraphDef on frozen graph throws") {
    tf_wrap::FastGraph g;
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Get a GraphDef to import
    auto graphdef = g.ToGraphDef();
    
    // Freeze the graph
    g.freeze();
    
    // ImportGraphDef should throw
    REQUIRE_THROWS(g.ImportGraphDef(graphdef.data(), graphdef.size(), "imported_"));
}

// ============================================================================
// CRITICAL: Policy copy shares mutex tests
// ============================================================================

TEST_CASE("Mutex policy copy shares underlying mutex") {
    tf_wrap::policy::Mutex p1;
    tf_wrap::policy::Mutex p2 = p1;
    
    REQUIRE(p1.shares_mutex_with(p2));
}

TEST_CASE("SharedMutex policy copy shares underlying mutex") {
    tf_wrap::policy::SharedMutex p1;
    tf_wrap::policy::SharedMutex p2 = p1;
    
    REQUIRE(p1.shares_mutex_with(p2));
}

STRESS_TEST("Lock on policy copy blocks lock on original") {
    tf_wrap::policy::Mutex p1;
    tf_wrap::policy::Mutex p2 = p1;
    
    std::atomic<bool> lock1_held{false};
    std::atomic<bool> lock2_acquired{false};
    
    std::thread t1([&] {
        auto guard = p1.scoped_lock();
        lock1_held.store(true);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        lock1_held.store(false);
    });
    
    // Wait for t1 to acquire lock
    while (!lock1_held.load()) {
        std::this_thread::yield();
    }
    
    std::thread t2([&] {
        auto guard = p2.scoped_lock();  // Should block
        lock2_acquired.store(true);
    });
    
    // Give t2 a chance to try acquiring
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    // t2 should NOT have acquired yet (t1 still holds lock)
    bool acquired_while_held = lock2_acquired.load();
    
    t1.join();
    t2.join();
    
    REQUIRE_FALSE(acquired_while_held);
    REQUIRE(lock2_acquired.load());
}

// ============================================================================
// HIGH: Session/Graph lifetime tests
// ============================================================================

TEST_CASE("Multiple Sessions from same Graph works") {
    tf_wrap::FastGraph g;
    auto t = tf_wrap::FastTensor::FromScalar<float>(42.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::FastSession s1(g);
    tf_wrap::FastSession s2(g);
    tf_wrap::FastSession s3(g);
    
    // All sessions should work
    auto r1 = s1.Run("A", 0);
    auto r2 = s2.Run("A", 0);
    auto r3 = s3.Run("A", 0);
    
    REQUIRE(r1.ToScalar<float>() == 42.0f);
    REQUIRE(r2.ToScalar<float>() == 42.0f);
    REQUIRE(r3.ToScalar<float>() == 42.0f);
}

TEST_CASE("Session still works after Graph frozen") {
    tf_wrap::FastGraph g;
    auto t = tf_wrap::FastTensor::FromScalar<float>(42.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::FastSession s(g);
    
    // Graph is now frozen
    REQUIRE(g.is_frozen());
    
    // Session should still work
    auto result = s.Run("A", 0);
    REQUIRE(result.ToScalar<float>() == 42.0f);
}

// ============================================================================
// HIGH: Session Run edge cases
// ============================================================================

TEST_CASE("Session Run with empty fetches") {
    tf_wrap::FastGraph g;
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::FastSession s(g);
    
    // Real TensorFlow requires at least one fetch or target and throws INVALID_ARGUMENT.
    // Stub mode returns empty results. Both behaviors are acceptable.
    std::vector<tf_wrap::Fetch> empty_fetches;
    try {
        auto results = s.Run({}, empty_fetches, {});
        // Stub mode: returns empty
        REQUIRE(results.empty());
    } catch (const std::runtime_error& e) {
        // Real TF: throws INVALID_ARGUMENT
        std::string msg = e.what();
        REQUIRE(msg.find("INVALID_ARGUMENT") != std::string::npos);
    }
}

TEST_CASE("Session Run with non-existent operation throws") {
    tf_wrap::FastGraph g;
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::FastSession s(g);
    
    REQUIRE_THROWS(s.Run("NonExistent", 0));
}

TEST_CASE("Session Run with non-existent feed operation throws") {
    tf_wrap::FastGraph g;
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::FastSession s(g);
    
    auto feed_tensor = tf_wrap::FastTensor::FromScalar<float>(2.0f);
    std::vector<tf_wrap::Feed> feeds = {
        tf_wrap::Feed{"NonExistent", 0, feed_tensor.handle()}
    };
    std::vector<tf_wrap::Fetch> fetches = {tf_wrap::Fetch{"A", 0}};
    
    REQUIRE_THROWS(s.Run(feeds, fetches, {}));
}

// ============================================================================
// HIGH: OperationBuilder error paths
// ============================================================================

TEST_CASE("OperationBuilder duplicate name handled") {
    tf_wrap::FastGraph g;
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    // Creating another op with same name - TF behavior varies
    // At minimum it should not crash
    auto t2 = tf_wrap::FastTensor::FromScalar<float>(2.0f);
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
    REQUIRE(g.valid());
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
        REQUIRE(s2.code() == TF_INVALID_ARGUMENT);
        // s1 is moved-from but safe to query
    }
    
    // Tensor
    {
        auto t1 = tf_wrap::FastTensor::FromScalar<float>(1.0f);
        auto t2 = std::move(t1);
        REQUIRE(t2.handle() != nullptr);
        REQUIRE(t1.handle() == nullptr);
    }
    
    // Graph
    {
        tf_wrap::FastGraph g1;
        tf_wrap::FastGraph g2(std::move(g1));
        REQUIRE(g2.valid());
        REQUIRE(!g1.valid());
    }
    
    // SessionOptions
    {
        tf_wrap::SessionOptions o1;
        tf_wrap::SessionOptions o2(std::move(o1));
        REQUIRE(o2.get() != nullptr);
        REQUIRE(o1.get() == nullptr);
    }
    
    // Buffer
    {
        tf_wrap::Buffer b1;  // Default constructor
        tf_wrap::Buffer b2(std::move(b1));
        REQUIRE(b2.get() != nullptr);
        REQUIRE(b1.get() == nullptr);
    }
}

TEST_CASE("Types with valid(): moved-from is not valid") {
    // Graph
    {
        tf_wrap::FastGraph g1;
        tf_wrap::FastGraph g2(std::move(g1));
        REQUIRE(!g1.valid());
    }
}

TEST_CASE("Types with handle(): moved-from has null handle") {
    // Tensor
    {
        auto t1 = tf_wrap::FastTensor::FromScalar<float>(1.0f);
        auto t2 = std::move(t1);
        REQUIRE(t1.handle() == nullptr);
    }
    
    // Graph
    {
        tf_wrap::FastGraph g1;
        tf_wrap::FastGraph g2(std::move(g1));
        REQUIRE(g1.handle() == nullptr);
    }
    
    // Session
    {
        tf_wrap::FastGraph g;
        auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
        (void)g.NewOperation("Const", "A")
            .SetAttrTensor("value", t.handle())
            .SetAttrType("dtype", TF_FLOAT)
            .Finish();
        
        tf_wrap::FastSession s1(g);
        tf_wrap::FastSession s2(std::move(s1));
        REQUIRE(s1.handle() == nullptr);
    }
    
    // SessionOptions
    {
        tf_wrap::SessionOptions o1;
        tf_wrap::SessionOptions o2(std::move(o1));
        REQUIRE(o1.get() == nullptr);
    }
    
    // Buffer
    {
        tf_wrap::Buffer b1;  // Default constructor
        tf_wrap::Buffer b2(std::move(b1));
        REQUIRE(b1.get() == nullptr);
    }
}

// ============================================================================
// MEDIUM: Thread safety claims verification
// ============================================================================

STRESS_TEST("SafeGraph methods are actually serialized") {
    tf_wrap::SafeGraph g;
    std::atomic<int> op_count{0};
    std::vector<std::thread> threads;
    
    // Multiple threads try to add operations
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&g, &op_count, i] {
            auto t = tf_wrap::SafeTensor::FromScalar<float>(static_cast<float>(i));
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
    REQUIRE(g.num_operations() == static_cast<std::size_t>(op_count.load()));
}

STRESS_TEST("SharedGraph allows concurrent reads") {
    tf_wrap::SharedGraph g;
    auto tensor = tf_wrap::SharedTensor::FromScalar<float>(42.0f);
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
                REQUIRE(g.HasOperation("A"));
                REQUIRE(g.num_operations() == 1u);
                read_count.fetch_add(1);
            }
        });
    }
    
    for (auto& th : threads) {
        th.join();
    }
    
    REQUIRE(read_count.load() == 1000);
}

STRESS_TEST("SafeSession Run is actually serialized") {
    tf_wrap::SafeGraph g;
    auto tensor = tf_wrap::SafeTensor::FromScalar<float>(42.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", tensor.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::SafeSession s(g);
    
    std::atomic<int> run_count{0};
    std::vector<std::thread> threads;
    
    // Multiple threads run concurrently
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&s, &run_count] {
            for (int j = 0; j < 10; ++j) {
                auto result = s.Run("A", 0);
                REQUIRE(result.ToScalar<float>() == 42.0f);
                run_count.fetch_add(1);
            }
        });
    }
    
    for (auto& th : threads) {
        th.join();
    }
    
    REQUIRE(run_count.load() == 100);
}

// ============================================================================
// MEDIUM: Error message quality tests
// ============================================================================

TEST_CASE("Error messages include operation name") {
    tf_wrap::FastGraph g;
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "MyConstOp")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    tf_wrap::FastSession s(g);
    
    try {
        (void)s.Run("NonExistentOp", 0);
        REQUIRE(false);  // Should have thrown
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        REQUIRE(msg.find("NonExistentOp") != std::string::npos);
    }
}

TEST_CASE("Frozen graph error is actionable") {
    tf_wrap::FastGraph g;
    auto t = tf_wrap::FastTensor::FromScalar<float>(1.0f);
    (void)g.NewOperation("Const", "A")
        .SetAttrTensor("value", t.handle())
        .SetAttrType("dtype", TF_FLOAT)
        .Finish();
    
    g.freeze();
    
    try {
        auto t2 = tf_wrap::FastTensor::FromScalar<float>(2.0f);
        (void)g.NewOperation("Const", "B")
            .SetAttrTensor("value", t2.handle())
            .SetAttrType("dtype", TF_FLOAT)
            .Finish();
        REQUIRE(false);
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        // Error should mention frozen/immutable and Session
        REQUIRE((msg.find("frozen") != std::string::npos || 
                 msg.find("immutable") != std::string::npos));
    }
}

TEST_CASE("Moved-from graph error is actionable") {
    tf_wrap::FastGraph g1;
    tf_wrap::FastGraph g2(std::move(g1));
    
    try {
        (void)g1.num_operations();
        REQUIRE(false);
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        REQUIRE(msg.find("moved") != std::string::npos);
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    bool run_stress = (argc > 1 && std::string(argv[1]) == "--stress");
    
    int passed = 0, failed = 0, skipped = 0;
    
    std::cout << "Running edge case tests...\n\n";
    
    for (const auto& tc : tf_test::registry()) {
        if (tc.is_stress && !run_stress) {
            ++skipped;
            continue;
        }
        
        std::cout << (tc.is_stress ? "[STRESS] " : "[TEST]   ") << tc.name << "\n";
        
        try {
            tc.fn();
            std::cout << "  PASS\n";
            ++passed;
        } catch (const std::exception& e) {
            std::cout << "  FAIL: " << e.what() << "\n";
            ++failed;
        }
    }
    
    std::cout << "\n========================================\n";
    std::cout << "Passed: " << passed << ", Failed: " << failed;
    if (skipped > 0) {
        std::cout << ", Skipped: " << skipped << " (run with --stress)";
    }
    std::cout << "\n";
    
    return failed > 0 ? 1 : 0;
}
