// tf/small_vector.hpp
// Stack-optimized vector for small collections (small buffer optimization)

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tf_wrap {

template<class T, std::size_t N>
class SmallVector {
public:
    using value_type = T;

    SmallVector() noexcept
        : data_(reinterpret_cast<T*>(storage_))
        , size_(0)
        , capacity_(N)
        , using_inline_(true)
    {}

    explicit SmallVector(std::size_t count)
        : SmallVector()
    {
        reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            emplace_back();
        }
    }

    SmallVector(std::initializer_list<T> init)
        : SmallVector()
    {
        reserve(init.size());
        for (const auto& v : init) {
            push_back(v);
        }
    }

    ~SmallVector() {
        clear();
        if (!using_inline_) {
            ::operator delete[](data_);
        }
    }

    SmallVector(const SmallVector&) = delete;
    SmallVector& operator=(const SmallVector&) = delete;

    SmallVector(SmallVector&& other) noexcept
        : data_(reinterpret_cast<T*>(storage_))
        , size_(0)
        , capacity_(N)
        , using_inline_(true)
    {
        move_from_(std::move(other));
    }

    SmallVector& operator=(SmallVector&& other) noexcept {
        if (this != &other) {
            clear();
            if (!using_inline_) {
                ::operator delete[](data_);
            }
            data_ = reinterpret_cast<T*>(storage_);
            size_ = 0;
            capacity_ = N;
            using_inline_ = true;
            move_from_(std::move(other));
        }
        return *this;
    }

    // ─────────────────────────────────────────────────────────────────
    // Capacity
    // ─────────────────────────────────────────────────────────────────

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    void reserve(std::size_t new_cap) {
        if (new_cap <= capacity_) return;

        T* new_data = static_cast<T*>(::operator new[](new_cap * sizeof(T)));
        for (std::size_t i = 0; i < size_; ++i) {
            new (new_data + i) T(std::move(data_[i]));
            std::destroy_at(data_ + i);
        }

        if (!using_inline_) {
            ::operator delete[](data_);
        }

        data_ = new_data;
        capacity_ = new_cap;
        using_inline_ = false;
    }

    void shrink_to_fit() {
        if (using_inline_) return;
        if (size_ <= N) {
            // Move back to inline storage
            T* new_data = reinterpret_cast<T*>(storage_);
            for (std::size_t i = 0; i < size_; ++i) {
                new (new_data + i) T(std::move(data_[i]));
                std::destroy_at(data_ + i);
            }
            ::operator delete[](data_);
            data_ = new_data;
            capacity_ = N;
            using_inline_ = true;
        } else {
            // Reallocate to exactly size_
            T* new_data = static_cast<T*>(::operator new[](size_ * sizeof(T)));
            for (std::size_t i = 0; i < size_; ++i) {
                new (new_data + i) T(std::move(data_[i]));
                std::destroy_at(data_ + i);
            }
            ::operator delete[](data_);
            data_ = new_data;
            capacity_ = size_;
            using_inline_ = false;
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // Element access
    // ─────────────────────────────────────────────────────────────────

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }

    [[nodiscard]] T& operator[](std::size_t i) noexcept { return data_[i]; }
    [[nodiscard]] const T& operator[](std::size_t i) const noexcept { return data_[i]; }

    [[nodiscard]] T& at(std::size_t i) {
        if (i >= size_) throw std::out_of_range("SmallVector::at");
        return data_[i];
    }

    [[nodiscard]] const T& at(std::size_t i) const {
        if (i >= size_) throw std::out_of_range("SmallVector::at");
        return data_[i];
    }

    [[nodiscard]] T& front() noexcept { return data_[0]; }
    [[nodiscard]] const T& front() const noexcept { return data_[0]; }

    [[nodiscard]] T& back() noexcept { return data_[size_ - 1]; }
    [[nodiscard]] const T& back() const noexcept { return data_[size_ - 1]; }

    // ─────────────────────────────────────────────────────────────────
    // Iteration
    // ─────────────────────────────────────────────────────────────────

    [[nodiscard]] T* begin() noexcept { return data_; }
    [[nodiscard]] T* end() noexcept { return data_ + size_; }

    [[nodiscard]] const T* begin() const noexcept { return data_; }
    [[nodiscard]] const T* end() const noexcept { return data_ + size_; }

    [[nodiscard]] const T* cbegin() const noexcept { return data_; }
    [[nodiscard]] const T* cend() const noexcept { return data_ + size_; }

    // ─────────────────────────────────────────────────────────────────
    // Modifiers
    // ─────────────────────────────────────────────────────────────────

    void clear() noexcept {
        for (std::size_t i = 0; i < size_; ++i) {
            std::destroy_at(data_ + i);
        }
        size_ = 0;
    }

    void push_back(const T& v) {
        if (size_ == capacity_) reserve(capacity_ * 2);
        new (data_ + size_) T(v);
        ++size_;
    }

    void push_back(T&& v) {
        if (size_ == capacity_) reserve(capacity_ * 2);
        new (data_ + size_) T(std::move(v));
        ++size_;
    }

    template<class... Args>
    T& emplace_back(Args&&... args) {
        if (size_ == capacity_) reserve(capacity_ * 2);
        new (data_ + size_) T(std::forward<Args>(args)...);
        ++size_;
        return back();
    }

    void pop_back() noexcept {
        assert(size_ > 0 && "SmallVector::pop_back on empty");
        --size_;
        std::destroy_at(data_ + size_);
    }

private:
    void move_from_(SmallVector&& other) noexcept {
        reserve(other.size_);
        for (std::size_t i = 0; i < other.size_; ++i) {
            emplace_back(std::move(other.data_[i]));
        }
        other.clear();
        other.shrink_to_fit();
    }

    alignas(T) unsigned char storage_[sizeof(T) * N]{};
    T* data_{nullptr};
    std::size_t size_{0};
    std::size_t capacity_{0};
    bool using_inline_{true};
};

} // namespace tf_wrap
