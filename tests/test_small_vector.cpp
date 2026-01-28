// test_small_vector.cpp
// Comprehensive tests for tf_wrap::SmallVector
//
// Framework: doctest
// Runs with: No TensorFlow dependency (all platforms)
//
// These tests cover:
// - Construction: default, count, initializer_list, range
// - Copy/Move semantics
// - Element access: operator[], at(), front(), back(), data()
// - Iterators: begin/end, rbegin/rend, range-for
// - Modifiers: push_back, emplace_back, pop_back, clear, resize
// - Capacity: reserve, shrink_to_fit, is_inline
// - Swap: inline-inline, heap-heap, mixed
// - Comparison operators
// - MultiIndex/Shape type aliases

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "tf_wrap/small_vector.hpp"

#include <algorithm>
#include <string>
#include <vector>

using namespace tf_wrap;

// ============================================================================
// Construction Tests
// ============================================================================

TEST_CASE("SmallVector - default construction") {
    SmallVector<int, 4> v;
    CHECK(v.empty());
    CHECK(v.size() == 0);
    CHECK(v.capacity() == 4);
    CHECK(v.is_inline());
}

TEST_CASE("SmallVector - count construction") {
    SmallVector<int, 4> v(3);
    CHECK(v.size() == 3);
    CHECK(v.is_inline());
}

TEST_CASE("SmallVector - count and value construction") {
    SmallVector<int, 4> v(3, 42);
    CHECK(v.size() == 3);
    CHECK(v[0] == 42);
    CHECK(v[1] == 42);
    CHECK(v[2] == 42);
}

TEST_CASE("SmallVector - initializer list within capacity") {
    SmallVector<int, 4> v = {1, 2, 3};
    CHECK(v.size() == 3);
    CHECK(v[0] == 1);
    CHECK(v[1] == 2);
    CHECK(v[2] == 3);
    CHECK(v.is_inline());
}

TEST_CASE("SmallVector - initializer list exceeding capacity") {
    SmallVector<int, 4> v = {1, 2, 3, 4, 5, 6};
    CHECK(v.size() == 6);
    CHECK_FALSE(v.is_inline());
    CHECK(v[5] == 6);
}

TEST_CASE("SmallVector - range construction") {
    std::vector<int> src = {10, 20, 30};
    SmallVector<int, 4> v(src.begin(), src.end());
    CHECK(v.size() == 3);
    CHECK(v[0] == 10);
    CHECK(v[1] == 20);
    CHECK(v[2] == 30);
}

// ============================================================================
// Copy/Move Tests
// ============================================================================

TEST_CASE("SmallVector - copy construction") {
    SmallVector<int, 4> v1 = {1, 2, 3};
    SmallVector<int, 4> v2(v1);
    
    CHECK(v2.size() == 3);
    CHECK(v2[0] == 1);
    CHECK(v1.size() == 3);  // Original unchanged
}

TEST_CASE("SmallVector - move construction inline") {
    SmallVector<int, 4> v1 = {1, 2, 3};
    SmallVector<int, 4> v2(std::move(v1));
    
    CHECK(v2.size() == 3);
    CHECK(v2[0] == 1);
    CHECK(v1.empty());  // Moved-from is empty
}

TEST_CASE("SmallVector - move construction heap") {
    SmallVector<int, 2> v1 = {1, 2, 3, 4, 5};  // Forces heap
    CHECK_FALSE(v1.is_inline());
    
    SmallVector<int, 2> v2(std::move(v1));
    CHECK(v2.size() == 5);
    CHECK_FALSE(v2.is_inline());
    CHECK(v1.empty());
    CHECK(v1.is_inline());  // Moved-from reverts to inline
}

TEST_CASE("SmallVector - copy assignment") {
    SmallVector<int, 4> v1 = {1, 2, 3};
    SmallVector<int, 4> v2;
    
    v2 = v1;
    CHECK(v2.size() == 3);
    CHECK(v2[2] == 3);
}

TEST_CASE("SmallVector - move assignment") {
    SmallVector<int, 4> v1 = {1, 2, 3};
    SmallVector<int, 4> v2;
    
    v2 = std::move(v1);
    CHECK(v2.size() == 3);
    CHECK(v1.empty());
}

// ============================================================================
// Element Access Tests
// ============================================================================

TEST_CASE("SmallVector - operator[]") {
    SmallVector<int, 4> v = {10, 20, 30, 40};
    
    CHECK(v[0] == 10);
    CHECK(v[3] == 40);
    
    v[1] = 25;
    CHECK(v[1] == 25);
}

TEST_CASE("SmallVector - at() valid index") {
    SmallVector<int, 4> v = {10, 20, 30};
    CHECK(v.at(0) == 10);
    CHECK(v.at(2) == 30);
}

TEST_CASE("SmallVector - at() invalid index throws") {
    SmallVector<int, 4> v = {10, 20, 30};
    
    bool threw = false;
    try {
        (void)v.at(10);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("SmallVector - front and back") {
    SmallVector<int, 4> v = {10, 20, 30, 40};
    CHECK(v.front() == 10);
    CHECK(v.back() == 40);
}

TEST_CASE("SmallVector - data()") {
    SmallVector<int, 4> v = {10, 20, 30};
    int* p = v.data();
    CHECK(p[0] == 10);
    CHECK(p[1] == 20);
}

// ============================================================================
// Iterator Tests
// ============================================================================

TEST_CASE("SmallVector - range-based for") {
    SmallVector<int, 4> v = {1, 2, 3, 4};
    
    int sum = 0;
    for (int x : v) sum += x;
    CHECK(sum == 10);
}

TEST_CASE("SmallVector - begin/end") {
    SmallVector<int, 4> v = {1, 2, 3, 4};
    
    CHECK(*v.begin() == 1);
    CHECK(*(v.end() - 1) == 4);
}

TEST_CASE("SmallVector - rbegin/rend") {
    SmallVector<int, 4> v = {1, 2, 3, 4};
    
    CHECK(*v.rbegin() == 4);
    CHECK(*(v.rend() - 1) == 1);
}

TEST_CASE("SmallVector - std::algorithm compatibility") {
    SmallVector<int, 4> v = {1, 2, 3, 4};
    
    auto it = std::find(v.begin(), v.end(), 3);
    CHECK(it != v.end());
    CHECK(*it == 3);
}

// ============================================================================
// Modifier Tests
// ============================================================================

TEST_CASE("SmallVector - push_back within capacity") {
    SmallVector<int, 4> v;
    
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);
    
    CHECK(v.size() == 3);
    CHECK(v.is_inline());
}

TEST_CASE("SmallVector - push_back triggering growth") {
    SmallVector<int, 4> v = {1, 2, 3, 4};
    v.push_back(5);  // Exceeds inline capacity
    
    CHECK(v.size() == 5);
    CHECK_FALSE(v.is_inline());
    CHECK(v[4] == 5);
}

TEST_CASE("SmallVector - emplace_back") {
    SmallVector<int, 4> v = {1, 2};
    v.emplace_back(3);
    
    CHECK(v.size() == 3);
    CHECK(v.back() == 3);
}

TEST_CASE("SmallVector - pop_back") {
    SmallVector<int, 4> v = {1, 2, 3};
    v.pop_back();
    
    CHECK(v.size() == 2);
    CHECK(v.back() == 2);
}

TEST_CASE("SmallVector - clear") {
    SmallVector<int, 4> v = {1, 2, 3, 4, 5};  // On heap
    CHECK_FALSE(v.is_inline());
    
    v.clear();
    CHECK(v.empty());
    CHECK_FALSE(v.is_inline());  // Capacity preserved
}

TEST_CASE("SmallVector - resize grow") {
    SmallVector<int, 4> v;
    v.resize(3);
    
    CHECK(v.size() == 3);
}

TEST_CASE("SmallVector - resize grow with value") {
    SmallVector<int, 4> v = {1, 2};
    v.resize(5, 42);
    
    CHECK(v.size() == 5);
    CHECK(v[2] == 42);
    CHECK(v[3] == 42);
    CHECK(v[4] == 42);
}

TEST_CASE("SmallVector - resize shrink") {
    SmallVector<int, 4> v = {1, 2, 3, 4, 5};
    v.resize(2);
    
    CHECK(v.size() == 2);
    CHECK(v[1] == 2);
}

// ============================================================================
// Capacity Tests
// ============================================================================

TEST_CASE("SmallVector - initial capacity") {
    SmallVector<int, 4> v = {1, 2};
    CHECK(v.capacity() == 4);
    CHECK(v.is_inline());
}

TEST_CASE("SmallVector - reserve within inline") {
    SmallVector<int, 4> v = {1, 2};
    v.reserve(3);
    
    CHECK(v.capacity() == 4);  // Still inline capacity
    CHECK(v.is_inline());
}

TEST_CASE("SmallVector - reserve forcing heap") {
    SmallVector<int, 4> v = {1, 2};
    v.reserve(10);
    
    CHECK(v.capacity() >= 10);
    CHECK_FALSE(v.is_inline());
    CHECK(v[0] == 1);  // Data preserved
    CHECK(v[1] == 2);
}

TEST_CASE("SmallVector - shrink_to_fit back to inline") {
    SmallVector<int, 4> v = {1, 2};
    v.reserve(10);  // Force heap
    CHECK_FALSE(v.is_inline());
    
    v.shrink_to_fit();
    CHECK(v.is_inline());
    CHECK(v.capacity() == 4);
    CHECK(v[0] == 1);
    CHECK(v[1] == 2);
}

// ============================================================================
// Swap Tests
// ============================================================================

TEST_CASE("SmallVector - swap both inline") {
    SmallVector<int, 4> a = {1, 2, 3};
    SmallVector<int, 4> b = {10, 20};
    
    a.swap(b);
    
    CHECK(a.size() == 2);
    CHECK(a[0] == 10);
    CHECK(b.size() == 3);
    CHECK(b[0] == 1);
}

TEST_CASE("SmallVector - swap both heap") {
    SmallVector<int, 2> a = {1, 2, 3, 4, 5};
    SmallVector<int, 2> b = {10, 20, 30};
    CHECK_FALSE(a.is_inline());
    CHECK_FALSE(b.is_inline());
    
    a.swap(b);
    
    CHECK(a.size() == 3);
    CHECK(a[0] == 10);
    CHECK(b.size() == 5);
    CHECK(b[0] == 1);
}

TEST_CASE("SmallVector - swap mixed inline and heap") {
    SmallVector<int, 4> a = {1, 2};
    SmallVector<int, 4> b = {10, 20, 30, 40, 50};
    CHECK(a.is_inline());
    CHECK_FALSE(b.is_inline());
    
    a.swap(b);
    
    CHECK(a.size() == 5);
    CHECK_FALSE(a.is_inline());
    CHECK(b.size() == 2);
    CHECK(b.is_inline());
}

// ============================================================================
// Comparison Tests
// ============================================================================

TEST_CASE("SmallVector - equality") {
    SmallVector<int, 4> a = {1, 2, 3};
    SmallVector<int, 4> b = {1, 2, 3};
    SmallVector<int, 4> c = {1, 2, 4};
    
    CHECK(a == b);
    CHECK_FALSE(a == c);
    CHECK(a != c);
}

TEST_CASE("SmallVector - less than") {
    SmallVector<int, 4> a = {1, 2, 3};
    SmallVector<int, 4> b = {1, 2, 4};
    
    CHECK(a < b);
    CHECK(b > a);
    CHECK(a <= b);
    CHECK(b >= a);
}

TEST_CASE("SmallVector - less than or equal") {
    SmallVector<int, 4> a = {1, 2, 3};
    SmallVector<int, 4> b = {1, 2, 3};
    
    CHECK(a <= b);
    CHECK(a >= b);
}

TEST_CASE("SmallVector - cross-capacity comparison") {
    SmallVector<int, 4> a = {1, 2, 3};
    SmallVector<int, 8> b = {1, 2, 3};  // Different inline capacity
    
    CHECK(a == b);
}

// ============================================================================
// MultiIndex/Shape Type Alias Tests
// ============================================================================

TEST_CASE("MultiIndex - basic usage") {
    MultiIndex idx = {2, 3, 4};
    
    CHECK(idx.size() == 3);
    CHECK(idx.is_inline());  // Critical for performance
    CHECK(idx[0] == 2);
    CHECK(idx[1] == 3);
    CHECK(idx[2] == 4);
}

TEST_CASE("Shape - basic usage") {
    Shape shape = {10, 20, 30};
    
    CHECK(shape.size() == 3);
    CHECK(shape.is_inline());
    CHECK(shape[0] == 10);
}

TEST_CASE("MultiIndex - tensor offset calculation pattern") {
    MultiIndex idx = {2, 3, 4};
    Shape shape = {10, 20, 30};
    
    // Simulate tensor offset calculation
    std::int64_t offset = 0;
    std::int64_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        offset += idx[i] * stride;
        stride *= shape[i];
    }
    
    // Expected: 2*600 + 3*30 + 4 = 1294
    CHECK(offset == 1294);
}

// ============================================================================
// String Element Tests
// ============================================================================

TEST_CASE("SmallVector - string elements") {
    SmallVector<std::string, 4> v = {"hello", "world"};
    
    CHECK(v.size() == 2);
    CHECK(v[0] == "hello");
    CHECK(v[1] == "world");
}

TEST_CASE("SmallVector - string push_back") {
    SmallVector<std::string, 2> v;
    v.push_back("one");
    v.push_back("two");
    v.push_back("three");  // Forces heap
    
    CHECK(v.size() == 3);
    CHECK_FALSE(v.is_inline());
    CHECK(v[2] == "three");
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_CASE("SmallVector - empty operations") {
    SmallVector<int, 4> v;
    
    CHECK(v.empty());
    CHECK(v.begin() == v.end());
    CHECK(v.data() != nullptr);  // data() valid even when empty
}

TEST_CASE("SmallVector - single element") {
    SmallVector<int, 4> v = {42};
    
    CHECK(v.size() == 1);
    CHECK(v.front() == 42);
    CHECK(v.back() == 42);
    CHECK(v.front() == v.back());
}

TEST_CASE("SmallVector - capacity 1") {
    SmallVector<int, 1> v;
    v.push_back(1);
    CHECK(v.is_inline());
    
    v.push_back(2);  // Forces heap
    CHECK_FALSE(v.is_inline());
    CHECK(v.size() == 2);
}
