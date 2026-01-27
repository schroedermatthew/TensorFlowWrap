// SPDX-License-Identifier: MIT
// Copyright (c) 2024 TensorFlowWrap Contributors
//
// small_vector.hpp - Stack-optimized vector for multi-index tensor access
//
// This is a minimal SmallVector implementation tailored for tensor multi-indices.
// It avoids heap allocation for small collections (up to InlineCapacity elements),
// which is critical for performance in tight tensor loops.
//
// Design rationale (see SmallVector case study):
// - std::vector<int> for indices causes ~11ns overhead per construction (malloc/free)
// - SmallVector with inline storage: ~1.6ns (5-8× faster)
// - For a 100³ tensor with 3 accesses per iteration: 31ms saved
//
// This implementation is self-contained with no external dependencies.

#pragma once

// GCC 14+ has false positives with -Wstringop-overflow in placement new contexts
// See: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=110952
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 14
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tf_wrap {

/// @brief Stack-optimized vector for small collections (e.g., tensor multi-indices)
///
/// Elements up to InlineCapacity are stored directly in the object (no heap allocation).
/// When size exceeds InlineCapacity, storage automatically transitions to heap.
///
/// Optimized for the tensor indexing use case where:
/// - Indices are typically small (rank 1-8)
/// - Indices are created/destroyed millions of times in tight loops
/// - Heap allocation overhead dominates execution time with std::vector
///
/// @tparam T Element type (must be trivially copyable for optimal performance)
/// @tparam InlineCapacity Number of elements to store inline (default: 8, suitable for most tensors)
template <typename T, std::size_t InlineCapacity = 8>
class SmallVector {
    static_assert(InlineCapacity > 0, "InlineCapacity must be positive");
    static_assert(std::is_trivially_destructible_v<T> || std::is_nothrow_destructible_v<T>,
                  "T must be trivially or nothrow destructible");

public:
    // Standard container type aliases
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = T*;
    using const_iterator = const T*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    // ==================================================================================
    // Constructors and Destructor
    // ==================================================================================

    /// @brief Default constructor - empty vector using inline storage
    constexpr SmallVector() noexcept
        : data_(inline_ptr())
        , size_(0)
        , capacity_(InlineCapacity)
    {}

    /// @brief Construct with count default-initialized elements
    explicit SmallVector(size_type count)
        : SmallVector()
    {
        resize(count);
    }

    /// @brief Construct with count copies of value
    SmallVector(size_type count, const T& value)
        : SmallVector()
    {
        assign(count, value);
    }

    /// @brief Construct from initializer list (the primary use case for multi-indices)
    SmallVector(std::initializer_list<T> init)
        : SmallVector()
    {
        assign(init);
    }

    /// @brief Range constructor
    template <typename InputIt,
              std::enable_if_t<!std::is_integral_v<InputIt>, int> = 0>
    SmallVector(InputIt first, InputIt last)
        : SmallVector()
    {
        assign(first, last);
    }

    /// @brief Copy constructor
    SmallVector(const SmallVector& other)
        : SmallVector()
    {
        assign(other.begin(), other.end());
    }

    /// @brief Move constructor
    SmallVector(SmallVector&& other) noexcept(std::is_nothrow_move_constructible_v<T>)
        : data_(inline_ptr())
        , size_(0)
        , capacity_(InlineCapacity)
    {
        if (other.is_inline()) {
            // Move elements from other's inline buffer to ours
            for (size_type i = 0; i < other.size_; ++i) {
                construct_at(data_ + i, std::move(other.data_[i]));
            }
            size_ = other.size_;
            // Destroy moved-from elements
            std::destroy_n(other.data_, other.size_);
        } else {
            // Steal heap pointer
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            // Reset other to empty inline state
            other.data_ = other.inline_ptr();
            other.capacity_ = InlineCapacity;
        }
        other.size_ = 0;
    }

    /// @brief Destructor
    ~SmallVector() noexcept {
        std::destroy_n(data_, size_);
        if (!is_inline()) {
            ::operator delete(data_);
        }
    }

    // ==================================================================================
    // Assignment
    // ==================================================================================

    SmallVector& operator=(const SmallVector& other) {
        if (this != &other) {
            assign(other.begin(), other.end());
        }
        return *this;
    }

    SmallVector& operator=(SmallVector&& other) noexcept(std::is_nothrow_move_constructible_v<T>) {
        if (this != &other) {
            clear();
            if (!is_inline()) {
                ::operator delete(data_);
                data_ = inline_ptr();
                capacity_ = InlineCapacity;
            }
            
            if (other.is_inline()) {
                for (size_type i = 0; i < other.size_; ++i) {
                    construct_at(data_ + i, std::move(other.data_[i]));
                }
                size_ = other.size_;
                std::destroy_n(other.data_, other.size_);
            } else {
                data_ = other.data_;
                size_ = other.size_;
                capacity_ = other.capacity_;
                other.data_ = other.inline_ptr();
                other.capacity_ = InlineCapacity;
            }
            other.size_ = 0;
        }
        return *this;
    }

    SmallVector& operator=(std::initializer_list<T> init) {
        assign(init);
        return *this;
    }

    void assign(size_type count, const T& value) {
        clear();
        reserve(count);
        for (size_type i = 0; i < count; ++i) {
            construct_at(data_ + i, value);
        }
        size_ = count;
    }

    template <typename InputIt,
              std::enable_if_t<!std::is_integral_v<InputIt>, int> = 0>
    void assign(InputIt first, InputIt last) {
        clear();
        if constexpr (std::is_base_of_v<std::forward_iterator_tag,
                                        typename std::iterator_traits<InputIt>::iterator_category>) {
            const auto count = static_cast<size_type>(std::distance(first, last));
            reserve(count);
            for (auto it = first; it != last; ++it) {
                construct_at(data_ + size_, *it);
                ++size_;
            }
        } else {
            for (; first != last; ++first) {
                push_back(*first);
            }
        }
    }

    void assign(std::initializer_list<T> init) {
        assign(init.begin(), init.end());
    }

    // ==================================================================================
    // Element Access
    // ==================================================================================

    [[nodiscard]] constexpr reference operator[](size_type pos) noexcept {
        return data_[pos];
    }

    [[nodiscard]] constexpr const_reference operator[](size_type pos) const noexcept {
        return data_[pos];
    }

    [[nodiscard]] reference at(size_type pos) {
        if (pos >= size_) {
            throw std::out_of_range("SmallVector::at: index out of range");
        }
        return data_[pos];
    }

    [[nodiscard]] const_reference at(size_type pos) const {
        if (pos >= size_) {
            throw std::out_of_range("SmallVector::at: index out of range");
        }
        return data_[pos];
    }

    [[nodiscard]] constexpr reference front() noexcept { return data_[0]; }
    [[nodiscard]] constexpr const_reference front() const noexcept { return data_[0]; }
    [[nodiscard]] constexpr reference back() noexcept { return data_[size_ - 1]; }
    [[nodiscard]] constexpr const_reference back() const noexcept { return data_[size_ - 1]; }
    [[nodiscard]] constexpr pointer data() noexcept { return data_; }
    [[nodiscard]] constexpr const_pointer data() const noexcept { return data_; }

    // ==================================================================================
    // Iterators
    // ==================================================================================

    [[nodiscard]] constexpr iterator begin() noexcept { return data_; }
    [[nodiscard]] constexpr const_iterator begin() const noexcept { return data_; }
    [[nodiscard]] constexpr const_iterator cbegin() const noexcept { return data_; }
    [[nodiscard]] constexpr iterator end() noexcept { return data_ + size_; }
    [[nodiscard]] constexpr const_iterator end() const noexcept { return data_ + size_; }
    [[nodiscard]] constexpr const_iterator cend() const noexcept { return data_ + size_; }
    
    [[nodiscard]] reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
    [[nodiscard]] const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
    [[nodiscard]] const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }
    [[nodiscard]] reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
    [[nodiscard]] const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
    [[nodiscard]] const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }

    // ==================================================================================
    // Capacity
    // ==================================================================================

    [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] constexpr size_type size() const noexcept { return size_; }
    [[nodiscard]] constexpr size_type capacity() const noexcept { return capacity_; }
    [[nodiscard]] constexpr bool is_inline() const noexcept { return data_ == inline_ptr(); }
    
    [[nodiscard]] static constexpr size_type inline_capacity() noexcept { 
        return InlineCapacity; 
    }

    [[nodiscard]] static constexpr size_type max_size() noexcept {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }

    void reserve(size_type new_cap) {
        if (new_cap <= capacity_) return;
        grow(new_cap);
    }

    void shrink_to_fit() {
        if (is_inline() || size_ == capacity_) return;
        
        if (size_ <= InlineCapacity) {
            // Can shrink back to inline
            T* old_data = data_;
            
            data_ = inline_ptr();
            capacity_ = InlineCapacity;
            
            for (size_type i = 0; i < size_; ++i) {
                construct_at(data_ + i, std::move(old_data[i]));
            }
            std::destroy_n(old_data, size_);
            ::operator delete(old_data);
        } else {
            // Shrink heap allocation
            T* new_data = static_cast<T*>(::operator new(size_ * sizeof(T)));
            for (size_type i = 0; i < size_; ++i) {
                construct_at(new_data + i, std::move(data_[i]));
            }
            std::destroy_n(data_, size_);
            ::operator delete(data_);
            data_ = new_data;
            capacity_ = size_;
        }
    }

    // ==================================================================================
    // Modifiers
    // ==================================================================================

    void clear() noexcept {
        std::destroy_n(data_, size_);
        size_ = 0;
    }

    void push_back(const T& value) {
        if (size_ >= capacity_) {
            grow(capacity_ * 2);
        }
        construct_at(data_ + size_, value);
        ++size_;
    }

    void push_back(T&& value) {
        if (size_ >= capacity_) {
            grow(capacity_ * 2);
        }
        construct_at(data_ + size_, std::move(value));
        ++size_;
    }

    template <typename... Args>
    reference emplace_back(Args&&... args) {
        if (size_ >= capacity_) {
            grow(capacity_ * 2);
        }
        construct_at(data_ + size_, std::forward<Args>(args)...);
        return data_[size_++];
    }

    void pop_back() noexcept {
        assert(size_ > 0 && "SmallVector::pop_back on empty");
        --size_;
        std::destroy_at(data_ + size_);
    }

    void resize(size_type count) {
        if (count > size_) {
            reserve(count);
            for (size_type i = size_; i < count; ++i) {
                construct_at(data_ + i);
            }
        } else {
            std::destroy_n(data_ + count, size_ - count);
        }
        size_ = count;
    }

    void resize(size_type count, const T& value) {
        if (count > size_) {
            reserve(count);
            for (size_type i = size_; i < count; ++i) {
                construct_at(data_ + i, value);
            }
        } else {
            std::destroy_n(data_ + count, size_ - count);
        }
        size_ = count;
    }

    void swap(SmallVector& other) noexcept(std::is_nothrow_move_constructible_v<T>) {
        if (this == &other) return;
        
        // Both inline - swap elements
        if (is_inline() && other.is_inline()) {
            const size_type max_size = std::max(size_, other.size_);
            for (size_type i = 0; i < max_size; ++i) {
                if (i < size_ && i < other.size_) {
                    using std::swap;
                    swap(data_[i], other.data_[i]);
                } else if (i < size_) {
                    construct_at(other.data_ + i, std::move(data_[i]));
                    std::destroy_at(data_ + i);
                } else {
                    construct_at(data_ + i, std::move(other.data_[i]));
                    std::destroy_at(other.data_ + i);
                }
            }
            std::swap(size_, other.size_);
            return;
        }
        
        // Both heap - just swap pointers
        if (!is_inline() && !other.is_inline()) {
            std::swap(data_, other.data_);
            std::swap(size_, other.size_);
            std::swap(capacity_, other.capacity_);
            return;
        }
        
        // One inline, one heap
        SmallVector* inline_vec = is_inline() ? this : &other;
        SmallVector* heap_vec = is_inline() ? &other : this;
        
        T* heap_data = heap_vec->data_;
        size_type heap_size = heap_vec->size_;
        size_type heap_cap = heap_vec->capacity_;
        size_type inline_size = inline_vec->size_;
        
        // Move inline elements to heap_vec's inline buffer
        heap_vec->data_ = heap_vec->inline_ptr();
        heap_vec->capacity_ = InlineCapacity;
        for (size_type i = 0; i < inline_size; ++i) {
            construct_at(heap_vec->data_ + i, std::move(inline_vec->data_[i]));
        }
        heap_vec->size_ = inline_size;
        
        // Destroy inline elements and give heap to inline_vec
        std::destroy_n(inline_vec->data_, inline_vec->size_);
        inline_vec->data_ = heap_data;
        inline_vec->size_ = heap_size;
        inline_vec->capacity_ = heap_cap;
    }

private:
    // Pointer to current storage (inline buffer or heap)
    T* data_;
    size_type size_;
    size_type capacity_;
    
    // Inline buffer - placed last to keep hot fields in first cache line
    alignas(T) std::byte inline_buffer_[InlineCapacity * sizeof(T)];

    [[nodiscard]] T* inline_ptr() noexcept {
        return reinterpret_cast<T*>(inline_buffer_);
    }

    [[nodiscard]] const T* inline_ptr() const noexcept {
        return reinterpret_cast<const T*>(inline_buffer_);
    }

    template <typename... Args>
    static void construct_at(T* p, Args&&... args) {
        ::new (static_cast<void*>(p)) T(std::forward<Args>(args)...);
    }

    void grow(size_type min_capacity) {
        // Overflow check
        if (min_capacity > max_size()) {
            throw std::length_error("SmallVector::grow: capacity overflow");
        }
        
        const size_type doubled = (capacity_ <= max_size() / 2)
            ? capacity_ * 2
            : max_size();
        const size_type new_cap = std::max(min_capacity, doubled);
        if (new_cap > max_size()) {
            throw std::length_error("SmallVector::grow: capacity overflow");
        }
        T* new_data = static_cast<T*>(::operator new(new_cap * sizeof(T)));
        
        // Move elements to new storage
        for (size_type i = 0; i < size_; ++i) {
            construct_at(new_data + i, std::move_if_noexcept(data_[i]));
        }
        
        // Destroy old elements
        std::destroy_n(data_, size_);
        
        // Free old heap storage (if any)
        if (!is_inline()) {
            ::operator delete(data_);
        }
        
        data_ = new_data;
        capacity_ = new_cap;
    }
};

// ==================================================================================
// Non-member functions
// ==================================================================================

template <typename T, std::size_t N>
void swap(SmallVector<T, N>& lhs, SmallVector<T, N>& rhs) 
    noexcept(noexcept(lhs.swap(rhs))) 
{
    lhs.swap(rhs);
}

template <typename T, std::size_t N1, std::size_t N2>
bool operator==(const SmallVector<T, N1>& lhs, const SmallVector<T, N2>& rhs) {
    return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

template <typename T, std::size_t N1, std::size_t N2>
bool operator!=(const SmallVector<T, N1>& lhs, const SmallVector<T, N2>& rhs) {
    return !(lhs == rhs);
}

template <typename T, std::size_t N1, std::size_t N2>
bool operator<(const SmallVector<T, N1>& lhs, const SmallVector<T, N2>& rhs) {
    return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}

template <typename T, std::size_t N1, std::size_t N2>
bool operator<=(const SmallVector<T, N1>& lhs, const SmallVector<T, N2>& rhs) {
    return !(rhs < lhs);
}

template <typename T, std::size_t N1, std::size_t N2>
bool operator>(const SmallVector<T, N1>& lhs, const SmallVector<T, N2>& rhs) {
    return rhs < lhs;
}

template <typename T, std::size_t N1, std::size_t N2>
bool operator>=(const SmallVector<T, N1>& lhs, const SmallVector<T, N2>& rhs) {
    return !(lhs < rhs);
}

// ==================================================================================
// Type alias for tensor multi-indices
// ==================================================================================

/// @brief Recommended type for tensor multi-indices
/// 
/// Use this instead of std::vector<int64_t> for tensor indexing:
/// @code
/// // BAD - heap allocation per access (~11ns overhead)
/// double& at(std::vector<int64_t> idx);
/// 
/// // GOOD - inline storage, no heap allocation (~1.6ns)
/// double& at(const MultiIndex& idx);
/// @endcode
using MultiIndex = SmallVector<std::int64_t, 8>;

/// @brief Shape type for tensor dimensions
using Shape = SmallVector<std::int64_t, 8>;

// ==================================================================================
// CTAD deduction guides (C++17)
// ==================================================================================

template <typename T>
SmallVector(std::initializer_list<T>) -> SmallVector<T, 8>;

} // namespace tf_wrap

#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 14
#pragma GCC diagnostic pop
#endif
