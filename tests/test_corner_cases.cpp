// test_corner_cases.cpp
// Comprehensive corner case and boundary condition tests
//
// Framework: doctest (header-only)
// Runs with: TF stub (all platforms)
//
// Tests edge cases, boundary conditions, and error handling that
// are often missed in happy-path testing.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "tf_wrap/tensor.hpp"
#include "tf_wrap/session.hpp"
#include "tf_wrap/graph.hpp"
#include "tf_wrap/small_vector.hpp"
#include "tf_wrap/status.hpp"
#include "tf_wrap/scope_guard.hpp"

#include <limits>
#include <string>
#include <vector>

using namespace tf_wrap;

// ============================================================================
// Tensor Corner Cases
// ============================================================================

TEST_SUITE("Tensor Corner Cases") {

    TEST_CASE("empty vector - zero elements") {
        std::vector<float> empty_data;
        auto tensor = Tensor::FromVector<float>({0}, empty_data);
        
        CHECK(tensor.valid());
        CHECK(tensor.num_elements() == 0);
        CHECK(tensor.byte_size() == 0);
        CHECK(tensor.rank() == 1);
        CHECK(tensor.shape()[0] == 0);
        
        auto view = tensor.read<float>();
        CHECK(view.empty());
    }
    
    TEST_CASE("zero in middle of shape") {
        std::vector<float> empty_data;
        auto tensor = Tensor::FromVector<float>({3, 0, 4}, empty_data);
        
        CHECK(tensor.valid());
        CHECK(tensor.num_elements() == 0);
        CHECK(tensor.rank() == 3);
    }
    
    TEST_CASE("scalar vs single-element vector") {
        auto scalar = Tensor::FromScalar<float>(42.0f);
        auto vector = Tensor::FromVector<float>({1}, {42.0f});
        
        CHECK(scalar.rank() == 0);
        CHECK(vector.rank() == 1);
        CHECK(scalar.num_elements() == 1);
        CHECK(vector.num_elements() == 1);
        CHECK(scalar.ToScalar<float>() == vector.ToScalar<float>());
    }
    
    TEST_CASE("high-dimensional tensor (8D)") {
        std::vector<float> data(2*2*2*2*2*2*2*2, 1.0f); // 256 elements
        auto tensor = Tensor::FromVector<float>({2,2,2,2,2,2,2,2}, data);
        
        CHECK(tensor.valid());
        CHECK(tensor.rank() == 8);
        CHECK(tensor.num_elements() == 256);
    }
    
    TEST_CASE("negative dimension throws") {
        std::vector<float> data(6, 1.0f);
        CHECK_THROWS_AS(
            Tensor::FromVector<float>({-2, 3}, data),
            std::exception
        );
    }
    
    TEST_CASE("shape mismatch - too few elements") {
        std::vector<float> data(5, 1.0f);
        CHECK_THROWS_AS(
            Tensor::FromVector<float>({2, 3}, data), // expects 6
            std::invalid_argument
        );
    }
    
    TEST_CASE("shape mismatch - too many elements") {
        std::vector<float> data(10, 1.0f);
        CHECK_THROWS_AS(
            Tensor::FromVector<float>({2, 3}, data), // expects 6
            std::invalid_argument
        );
    }
    
    TEST_CASE("reshape to same total elements") {
        auto tensor = Tensor::FromVector<float>({2, 3}, {1,2,3,4,5,6});
        auto reshaped = tensor.reshape({3, 2});
        
        CHECK(reshaped.num_elements() == 6);
        CHECK(reshaped.rank() == 2);
        CHECK(reshaped.shape()[0] == 3);
        CHECK(reshaped.shape()[1] == 2);
    }
    
    TEST_CASE("reshape to different total throws") {
        auto tensor = Tensor::FromVector<float>({2, 3}, {1,2,3,4,5,6});
        CHECK_THROWS_AS(
            tensor.reshape({2, 2}), // 4 != 6
            std::exception
        );
    }
    
    TEST_CASE("reshape empty tensor") {
        auto tensor = Tensor::FromVector<float>({0}, {});
        auto reshaped = tensor.reshape({0, 5}); // still 0 elements
        
        CHECK(reshaped.num_elements() == 0);
    }
    
    TEST_CASE("dtype mismatch on read") {
        auto tensor = Tensor::FromScalar<float>(1.0f);
        CHECK_THROWS_AS(
            tensor.read<int32_t>(),
            std::exception
        );
    }
    
    TEST_CASE("dtype mismatch on ToScalar") {
        auto tensor = Tensor::FromScalar<double>(1.0);
        CHECK_THROWS_AS(
            tensor.ToScalar<float>(),
            std::exception
        );
    }
    
    TEST_CASE("ToScalar on non-scalar throws") {
        auto tensor = Tensor::FromVector<float>({2}, {1.0f, 2.0f});
        CHECK_THROWS_AS(
            tensor.ToScalar<float>(),
            std::exception
        );
    }
    
    TEST_CASE("clone empty tensor") {
        auto tensor = Tensor::FromVector<float>({0}, {});
        auto cloned = tensor.Clone();
        
        CHECK(cloned.valid());
        CHECK(cloned.num_elements() == 0);
    }
    
    TEST_CASE("clone preserves data independence") {
        auto original = Tensor::FromVector<float>({3}, {1.0f, 2.0f, 3.0f});
        auto cloned = original.Clone();
        
        // Modify clone via write
        {
            auto view = cloned.write<float>();
            view[0] = 99.0f;
        }
        
        // Original unchanged
        auto orig_view = original.read<float>();
        CHECK(orig_view[0] == 1.0f);
    }
    
    TEST_CASE("moved-from tensor is invalid") {
        auto t1 = Tensor::FromScalar<float>(1.0f);
        auto t2 = std::move(t1);
        
        CHECK_FALSE(t1.valid());
        CHECK(t2.valid());
    }
    
    TEST_CASE("operations on moved-from tensor") {
        auto t1 = Tensor::FromScalar<float>(1.0f);
        auto t2 = std::move(t1);
        
        // These should throw or return safe defaults
        CHECK_THROWS(t1.dtype());
        // rank() and num_elements() return 0 for null tensor (safe default)
        CHECK(t1.rank() == 0);
        CHECK(t1.num_elements() == 0);
        CHECK_THROWS(t1.read<float>());
    }
    
    TEST_CASE("empty string tensor") {
        auto tensor = Tensor::FromString("");
        CHECK(tensor.valid());
        CHECK(tensor.ToString() == "");
    }
    
    TEST_CASE("string with null bytes") {
        std::string data("hello\0world", 11);
        auto tensor = Tensor::FromString(data);
        CHECK(tensor.ToString() == data);
        CHECK(tensor.ToString().size() == 11);
    }
    
    TEST_CASE("very long string") {
        std::string long_str(100000, 'x');
        auto tensor = Tensor::FromString(long_str);
        CHECK(tensor.ToString() == long_str);
    }
    
    TEST_CASE("all supported dtypes") {
        CHECK_NOTHROW(Tensor::FromScalar<float>(1.0f));
        CHECK_NOTHROW(Tensor::FromScalar<double>(1.0));
        CHECK_NOTHROW(Tensor::FromScalar<int8_t>(1));
        CHECK_NOTHROW(Tensor::FromScalar<int16_t>(1));
        CHECK_NOTHROW(Tensor::FromScalar<int32_t>(1));
        CHECK_NOTHROW(Tensor::FromScalar<int64_t>(1));
        CHECK_NOTHROW(Tensor::FromScalar<uint8_t>(1));
        CHECK_NOTHROW(Tensor::FromScalar<uint16_t>(1));
        CHECK_NOTHROW(Tensor::FromScalar<uint32_t>(1));
        CHECK_NOTHROW(Tensor::FromScalar<uint64_t>(1));
        CHECK_NOTHROW(Tensor::FromScalar<bool>(true));
    }
    
    TEST_CASE("extreme float values") {
        CHECK_NOTHROW(Tensor::FromScalar<float>(std::numeric_limits<float>::max()));
        CHECK_NOTHROW(Tensor::FromScalar<float>(std::numeric_limits<float>::min()));
        CHECK_NOTHROW(Tensor::FromScalar<float>(std::numeric_limits<float>::lowest()));
        CHECK_NOTHROW(Tensor::FromScalar<float>(std::numeric_limits<float>::infinity()));
        CHECK_NOTHROW(Tensor::FromScalar<float>(-std::numeric_limits<float>::infinity()));
        
        auto nan_tensor = Tensor::FromScalar<float>(std::numeric_limits<float>::quiet_NaN());
        CHECK(std::isnan(nan_tensor.ToScalar<float>()));
    }
    
    TEST_CASE("extreme integer values") {
        auto max_i64 = Tensor::FromScalar<int64_t>(std::numeric_limits<int64_t>::max());
        auto min_i64 = Tensor::FromScalar<int64_t>(std::numeric_limits<int64_t>::min());
        
        CHECK(max_i64.ToScalar<int64_t>() == std::numeric_limits<int64_t>::max());
        CHECK(min_i64.ToScalar<int64_t>() == std::numeric_limits<int64_t>::min());
    }

}

// ============================================================================
// SmallVector Corner Cases
// ============================================================================

TEST_SUITE("SmallVector Corner Cases") {

    TEST_CASE("empty vector state") {
        SmallVector<int, 4> v;
        
        CHECK(v.empty());
        CHECK(v.size() == 0);
        CHECK(v.capacity() >= 4);
        CHECK(v.begin() == v.end());
    }
    
    TEST_CASE("exact inline capacity") {
        SmallVector<int, 4> v;
        v.push_back(1);
        v.push_back(2);
        v.push_back(3);
        v.push_back(4);
        
        CHECK(v.size() == 4);
        CHECK(v.capacity() == 4); // Should still be inline
    }
    
    TEST_CASE("one past inline capacity - heap allocation") {
        SmallVector<int, 4> v;
        for (int i = 0; i < 5; ++i) {
            v.push_back(i);
        }
        
        CHECK(v.size() == 5);
        CHECK(v.capacity() > 4); // Should have grown
        
        // Verify data integrity after reallocation
        for (int i = 0; i < 5; ++i) {
            CHECK(v[i] == i);
        }
    }
    
    TEST_CASE("reserve zero does nothing harmful") {
        SmallVector<int, 4> v;
        v.push_back(1);
        v.reserve(0);
        
        CHECK(v.size() == 1);
        CHECK(v[0] == 1);
    }
    
    TEST_CASE("reserve less than current capacity") {
        SmallVector<int, 4> v;
        for (int i = 0; i < 4; ++i) v.push_back(i);
        
        auto old_cap = v.capacity();
        v.reserve(2);
        
        CHECK(v.capacity() == old_cap); // No shrink
        CHECK(v.size() == 4);
    }
    
    TEST_CASE("resize to zero") {
        SmallVector<int, 4> v;
        v.push_back(1);
        v.push_back(2);
        v.resize(0);
        
        CHECK(v.empty());
        CHECK(v.size() == 0);
    }
    
    TEST_CASE("resize grow with value") {
        SmallVector<int, 4> v;
        v.resize(3, 42);
        
        CHECK(v.size() == 3);
        CHECK(v[0] == 42);
        CHECK(v[1] == 42);
        CHECK(v[2] == 42);
    }
    
    TEST_CASE("pop_back to empty") {
        SmallVector<int, 4> v;
        v.push_back(1);
        v.pop_back();
        
        CHECK(v.empty());
    }
    
    TEST_CASE("clear preserves capacity") {
        SmallVector<int, 4> v;
        for (int i = 0; i < 10; ++i) v.push_back(i);
        
        auto cap_before = v.capacity();
        v.clear();
        
        CHECK(v.empty());
        CHECK(v.capacity() == cap_before);
    }
    
    TEST_CASE("self-assignment is safe") {
        SmallVector<int, 4> v;
        v.push_back(1);
        v.push_back(2);
        
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wself-assign-overloaded"
        v = v;
        #pragma GCC diagnostic pop
        
        CHECK(v.size() == 2);
        CHECK(v[0] == 1);
        CHECK(v[1] == 2);
    }
    
    TEST_CASE("copy same capacity") {
        SmallVector<int, 4> src;
        for (int i = 0; i < 3; ++i) src.push_back(i);
        
        SmallVector<int, 4> dst;
        dst.push_back(99);
        
        dst = src;
        
        CHECK(dst.size() == 3);
        for (int i = 0; i < 3; ++i) {
            CHECK(dst[i] == i);
        }
    }
    
    TEST_CASE("move leaves source empty") {
        SmallVector<int, 4> src;
        src.push_back(1);
        src.push_back(2);
        
        SmallVector<int, 4> dst = std::move(src);
        
        CHECK(dst.size() == 2);
        CHECK(src.empty());
    }
    
    TEST_CASE("at() throws on out of bounds") {
        SmallVector<int, 4> v;
        v.push_back(1);
        
        CHECK_NOTHROW(v.at(0));
        CHECK_THROWS_AS(v.at(1), std::out_of_range);
        CHECK_THROWS_AS(v.at(100), std::out_of_range);
    }
    
    TEST_CASE("front and back on single element") {
        SmallVector<int, 4> v;
        v.push_back(42);
        
        CHECK(v.front() == 42);
        CHECK(v.back() == 42);
        CHECK(&v.front() == &v.back());
    }
    
    TEST_CASE("data pointer stability after reserve") {
        SmallVector<int, 4> v;
        v.reserve(100);
        v.push_back(1);
        
        int* ptr = v.data();
        
        for (int i = 0; i < 50; ++i) {
            v.push_back(i);
        }
        
        // Pointer should be stable since we reserved enough
        CHECK(v.data() == ptr);
    }
    
    TEST_CASE("swap different sizes") {
        SmallVector<int, 4> a, b;
        a.push_back(1);
        a.push_back(2);
        b.push_back(10);
        
        a.swap(b);
        
        CHECK(a.size() == 1);
        CHECK(a[0] == 10);
        CHECK(b.size() == 2);
        CHECK(b[0] == 1);
        CHECK(b[1] == 2);
    }
    
    TEST_CASE("swap with empty") {
        SmallVector<int, 4> a, b;
        a.push_back(1);
        
        a.swap(b);
        
        CHECK(a.empty());
        CHECK(b.size() == 1);
        CHECK(b[0] == 1);
    }

}

// ============================================================================
// Session/Graph Corner Cases
// ============================================================================

TEST_SUITE("Session Corner Cases") {

    TEST_CASE("empty graph operations") {
        Graph g;
        
        CHECK(g.handle() != nullptr);
        auto ops = g.GetAllOperations();
        CHECK(ops.empty());
        CHECK(g.num_operations() == 0);
    }
    
    TEST_CASE("get operation with empty name") {
        Graph g;
        auto op = g.GetOperation("");
        CHECK_FALSE(op.has_value());
    }
    
    TEST_CASE("get operation with very long name") {
        Graph g;
        std::string long_name(10000, 'x');
        auto op = g.GetOperation(long_name);
        CHECK_FALSE(op.has_value());
    }
    
    TEST_CASE("session from empty graph") {
        Graph g;
        SessionOptions opts;
        
        CHECK_NOTHROW(Session(g, opts));
    }
    
    TEST_CASE("multiple sessions from same graph") {
        Graph g;
        SessionOptions opts;
        
        Session s1(g, opts);
        Session s2(g, opts);
        
        CHECK(s1.handle() != nullptr);
        CHECK(s2.handle() != nullptr);
        CHECK(s1.handle() != s2.handle());
    }
    
    TEST_CASE("session move leaves source null") {
        Graph g;
        SessionOptions opts;
        Session s1(g, opts);
        
        Session s2 = std::move(s1);
        
        CHECK(s2.handle() != nullptr);
        CHECK(s1.handle() == nullptr);
    }
    
    TEST_CASE("operations on moved-from session") {
        Graph g;
        SessionOptions opts;
        Session s1(g, opts);
        Session s2 = std::move(s1);
        
        CHECK_THROWS(s1.ListDevices());
    }
    
    TEST_CASE("SessionOptions move") {
        SessionOptions o1;
        o1.SetTarget("local");
        
        SessionOptions o2 = std::move(o1);
        
        CHECK(o2.handle() != nullptr);
        CHECK(o1.handle() == nullptr);
    }
    
    TEST_CASE("Buffer empty") {
        Buffer b;
        CHECK(b.data() == nullptr);
        CHECK(b.to_bytes().empty());
    }
    
    TEST_CASE("Buffer from empty data") {
        Buffer b(nullptr, 0);
        auto bytes = b.to_bytes();
        CHECK(bytes.empty());
    }
    
    TEST_CASE("Buffer from data") {
        const char* data = "hello";
        Buffer b(data, 5);
        
        auto bytes = b.to_bytes();
        CHECK(bytes.size() == 5);
        CHECK(std::memcmp(bytes.data(), data, 5) == 0);
    }

}

// ============================================================================
// Status Corner Cases
// ============================================================================

TEST_SUITE("Status Corner Cases") {

    TEST_CASE("default status is OK") {
        Status st;
        CHECK(st.ok());
        CHECK(st.code() == TF_OK);
    }
    
    TEST_CASE("throw_if_error on OK does nothing") {
        Status st;
        CHECK_NOTHROW(st.throw_if_error("test"));
    }
    
    TEST_CASE("all TF error codes") {
        // Just verify we can create status - actual error setting
        // would require TF operations
        Status st;
        CHECK(st.code() == TF_OK);
    }

}

// ============================================================================
// ScopeGuard Corner Cases
// ============================================================================

TEST_SUITE("ScopeGuard Corner Cases") {

    TEST_CASE("dismiss prevents execution") {
        int counter = 0;
        {
            auto guard = tf_wrap::makeScopeGuard([&]{ counter++; });
            guard.dismiss();
        }
        CHECK(counter == 0);
    }
    
    TEST_CASE("double dismiss is safe") {
        int counter = 0;
        {
            auto guard = tf_wrap::makeScopeGuard([&]{ counter++; });
            guard.dismiss();
            guard.dismiss(); // Should be safe
        }
        CHECK(counter == 0);
    }
    
    TEST_CASE("move transfers ownership") {
        int counter = 0;
        {
            auto guard1 = tf_wrap::makeScopeGuard([&]{ counter++; });
            auto guard2 = std::move(guard1);
            // guard1 is dismissed by move
        }
        CHECK(counter == 1); // Only executed once
    }
    
    TEST_CASE("empty lambda works") {
        CHECK_NOTHROW({
            auto guard = tf_wrap::makeScopeGuard([]{ });
        });
    }
    
    TEST_CASE("executes on exception") {
        int counter = 0;
        try {
            auto guard = tf_wrap::makeScopeGuard([&]{ counter++; });
            throw std::runtime_error("test");
        } catch (...) {
            // Expected
        }
        CHECK(counter == 1);
    }

}

// ============================================================================
// Integration Corner Cases
// ============================================================================

TEST_SUITE("Integration Corner Cases") {

    TEST_CASE("tensor in SmallVector") {
        SmallVector<Tensor, 4> tensors;
        
        tensors.push_back(Tensor::FromScalar<float>(1.0f));
        tensors.push_back(Tensor::FromScalar<float>(2.0f));
        
        CHECK(tensors.size() == 2);
        CHECK(tensors[0].ToScalar<float>() == 1.0f);
        CHECK(tensors[1].ToScalar<float>() == 2.0f);
    }
    
    TEST_CASE("move tensor in SmallVector") {
        SmallVector<Tensor, 4> tensors;
        
        auto t = Tensor::FromScalar<float>(42.0f);
        tensors.push_back(std::move(t));
        
        CHECK_FALSE(t.valid());
        CHECK(tensors[0].valid());
        CHECK(tensors[0].ToScalar<float>() == 42.0f);
    }
    
    TEST_CASE("graph survives session destruction") {
        Graph g;
        {
            SessionOptions opts;
            Session s(g, opts);
            // Session destroyed here
        }
        // Graph should still be valid
        CHECK(g.handle() != nullptr);
    }

}
