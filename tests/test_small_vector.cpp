// test_small_vector.cpp - Tests for SmallVector implementation
// Compile: g++ -std=c++20 -O3 -I../include test_small_vector.cpp -o test_small_vector

#include <tf_wrap/small_vector.hpp>
#include <cassert>
#include <iostream>
#include <string>
#include <chrono>
#include <vector>

using namespace tf_wrap;

void test_construction() {
    std::cout << "Testing construction... ";
    
    // Default construction
    SmallVector<int, 4> v1;
    assert(v1.empty());
    assert(v1.size() == 0);
    assert(v1.capacity() == 4);
    assert(v1.is_inline());
    
    // Count construction
    SmallVector<int, 4> v2(3);
    assert(v2.size() == 3);
    assert(v2.is_inline());
    
    // Count + value construction
    SmallVector<int, 4> v3(3, 42);
    assert(v3.size() == 3);
    assert(v3[0] == 42 && v3[1] == 42 && v3[2] == 42);
    
    // Initializer list (the primary use case!)
    SmallVector<int, 4> v4 = {1, 2, 3};
    assert(v4.size() == 3);
    assert(v4[0] == 1 && v4[1] == 2 && v4[2] == 3);
    assert(v4.is_inline());
    
    // Initializer list exceeding inline capacity
    SmallVector<int, 4> v5 = {1, 2, 3, 4, 5, 6};
    assert(v5.size() == 6);
    assert(!v5.is_inline());
    
    // Range construction
    std::vector<int> src = {10, 20, 30};
    SmallVector<int, 4> v6(src.begin(), src.end());
    assert(v6.size() == 3);
    assert(v6[1] == 20);
    
    std::cout << "PASSED\n";
}

void test_copy_move() {
    std::cout << "Testing copy/move... ";
    
    SmallVector<int, 4> v1 = {1, 2, 3};
    
    // Copy construction
    SmallVector<int, 4> v2(v1);
    assert(v2.size() == 3);
    assert(v2[0] == 1);
    assert(v1.size() == 3); // Original unchanged
    
    // Move construction (inline)
    SmallVector<int, 4> v3(std::move(v2));
    assert(v3.size() == 3);
    assert(v3[0] == 1);
    assert(v2.empty()); // Moved-from is empty
    
    // Move construction (heap)
    SmallVector<int, 2> v4 = {1, 2, 3, 4, 5}; // Forces heap
    assert(!v4.is_inline());
    SmallVector<int, 2> v5(std::move(v4));
    assert(v5.size() == 5);
    assert(!v5.is_inline());
    assert(v4.empty());
    assert(v4.is_inline()); // Moved-from reverts to inline
    
    // Copy assignment
    SmallVector<int, 4> v6;
    v6 = v1;
    assert(v6.size() == 3);
    assert(v6[2] == 3);
    
    // Move assignment
    SmallVector<int, 4> v7;
    v7 = std::move(v6);
    assert(v7.size() == 3);
    assert(v6.empty());
    
    std::cout << "PASSED\n";
}

void test_element_access() {
    std::cout << "Testing element access... ";
    
    SmallVector<int, 4> v = {10, 20, 30, 40};
    
    // operator[]
    assert(v[0] == 10);
    assert(v[3] == 40);
    v[1] = 25;
    assert(v[1] == 25);
    
    // at() with bounds checking
    assert(v.at(0) == 10);
    bool threw = false;
    try {
        (void)v.at(10);  // Cast to void to suppress [[nodiscard]] warning
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
    
    // front/back
    assert(v.front() == 10);
    assert(v.back() == 40);
    
    // data()
    int* p = v.data();
    assert(p[0] == 10);
    
    std::cout << "PASSED\n";
}

void test_iterators() {
    std::cout << "Testing iterators... ";
    
    SmallVector<int, 4> v = {1, 2, 3, 4};
    
    // Range-based for
    int sum = 0;
    for (int x : v) sum += x;
    assert(sum == 10);
    
    // begin/end
    assert(*v.begin() == 1);
    assert(*(v.end() - 1) == 4);
    
    // Reverse iterators
    assert(*v.rbegin() == 4);
    assert(*(v.rend() - 1) == 1);
    
    // std::algorithm compatibility
    auto it = std::find(v.begin(), v.end(), 3);
    assert(it != v.end() && *it == 3);
    
    std::cout << "PASSED\n";
}

void test_modifiers() {
    std::cout << "Testing modifiers... ";
    
    SmallVector<int, 4> v;
    
    // push_back
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);
    assert(v.size() == 3);
    assert(v.is_inline());
    
    // push_back triggering growth
    v.push_back(4);
    v.push_back(5); // Exceeds inline capacity
    assert(v.size() == 5);
    assert(!v.is_inline());
    assert(v[4] == 5);
    
    // emplace_back
    v.emplace_back(6);
    assert(v.back() == 6);
    
    // pop_back
    v.pop_back();
    assert(v.size() == 5);
    assert(v.back() == 5);
    
    // clear
    v.clear();
    assert(v.empty());
    assert(!v.is_inline()); // Capacity preserved
    
    // resize
    v.resize(3);
    assert(v.size() == 3);
    
    v.resize(5, 42);
    assert(v.size() == 5);
    assert(v[3] == 42 && v[4] == 42);
    
    std::cout << "PASSED\n";
}

void test_capacity() {
    std::cout << "Testing capacity... ";
    
    SmallVector<int, 4> v = {1, 2};
    assert(v.capacity() == 4);
    assert(v.is_inline());
    
    // reserve within inline
    v.reserve(3);
    assert(v.capacity() == 4);
    assert(v.is_inline());
    
    // reserve forcing heap
    v.reserve(10);
    assert(v.capacity() >= 10);
    assert(!v.is_inline());
    assert(v[0] == 1 && v[1] == 2); // Data preserved
    
    // shrink_to_fit back to inline
    v.shrink_to_fit();
    assert(v.is_inline());
    assert(v.capacity() == 4);
    assert(v[0] == 1 && v[1] == 2);
    
    std::cout << "PASSED\n";
}

void test_swap() {
    std::cout << "Testing swap... ";
    
    // Both inline
    SmallVector<int, 4> a = {1, 2, 3};
    SmallVector<int, 4> b = {10, 20};
    a.swap(b);
    assert(a.size() == 2 && a[0] == 10);
    assert(b.size() == 3 && b[0] == 1);
    
    // Both heap
    SmallVector<int, 2> c = {1, 2, 3, 4, 5};
    SmallVector<int, 2> d = {10, 20, 30};
    assert(!c.is_inline() && !d.is_inline());
    c.swap(d);
    assert(c.size() == 3 && c[0] == 10);
    assert(d.size() == 5 && d[0] == 1);
    
    // Mixed (one inline, one heap)
    SmallVector<int, 4> e = {1, 2};
    SmallVector<int, 4> f = {10, 20, 30, 40, 50};
    assert(e.is_inline() && !f.is_inline());
    e.swap(f);
    assert(e.size() == 5 && !e.is_inline());
    assert(f.size() == 2 && f.is_inline());
    
    std::cout << "PASSED\n";
}

void test_comparison() {
    std::cout << "Testing comparison... ";
    
    SmallVector<int, 4> a = {1, 2, 3};
    SmallVector<int, 4> b = {1, 2, 3};
    SmallVector<int, 4> c = {1, 2, 4};
    SmallVector<int, 8> d = {1, 2, 3}; // Different capacity
    
    assert(a == b);
    assert(a != c);
    assert(a < c);
    assert(c > a);
    assert(a <= b);
    assert(a >= b);
    
    // Cross-capacity comparison
    assert(a == d);
    
    std::cout << "PASSED\n";
}

void test_multi_index_usage() {
    std::cout << "Testing MultiIndex usage pattern... ";
    
    // This is the primary use case - tensor indexing
    MultiIndex idx = {2, 3, 4};
    assert(idx.size() == 3);
    assert(idx.is_inline()); // Critical for performance!
    
    // Simulate tensor offset calculation
    Shape shape = {10, 20, 30};
    std::int64_t offset = 0;
    std::int64_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        offset += idx[i] * stride;
        stride *= shape[i];
    }
    assert(offset == 2 * 600 + 3 * 30 + 4); // 1294
    
    std::cout << "PASSED\n";
}

void benchmark_vs_vector() {
    std::cout << "\nBenchmark: SmallVector vs std::vector for multi-index construction\n";
    
    constexpr int iterations = 10'000'000;
    volatile std::int64_t sink = 0;
    
    // Warm up
    for (int i = 0; i < 1000; ++i) {
        MultiIndex idx = {1, 2, 3};
        sink = sink + idx[0];  // Avoid compound assignment on volatile (deprecated in C++20)
    }
    
    // SmallVector timing
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        MultiIndex idx = {static_cast<std::int64_t>(i % 10), 
                          static_cast<std::int64_t>(i % 20), 
                          static_cast<std::int64_t>(i % 30)};
        sink = sink + idx[0] + idx[1] + idx[2];
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto small_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    // std::vector timing
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        std::vector<std::int64_t> idx = {static_cast<std::int64_t>(i % 10), 
                                          static_cast<std::int64_t>(i % 20), 
                                          static_cast<std::int64_t>(i % 30)};
        sink = sink + idx[0] + idx[1] + idx[2];
    }
    end = std::chrono::high_resolution_clock::now();
    auto vec_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    double small_per_op = static_cast<double>(small_ns) / iterations;
    double vec_per_op = static_cast<double>(vec_ns) / iterations;
    double speedup = vec_per_op / small_per_op;
    
    std::cout << "  SmallVector: " << small_per_op << " ns/op\n";
    std::cout << "  std::vector: " << vec_per_op << " ns/op\n";
    std::cout << "  Speedup: " << speedup << "x\n";
    
    if (speedup < 2.0) {
        std::cout << "  (Note: speedup may be lower in debug builds or with optimizations disabled)\n";
    }
}

int main() {
    std::cout << "=== SmallVector Test Suite ===\n\n";
    
    test_construction();
    test_copy_move();
    test_element_access();
    test_iterators();
    test_modifiers();
    test_capacity();
    test_swap();
    test_comparison();
    test_multi_index_usage();
    
    std::cout << "\nAll tests PASSED!\n";
    
    benchmark_vs_vector();
    
    return 0;
}
