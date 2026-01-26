// tf/tensor.hpp
// RAII wrapper for TF_Tensor with lifetime-safe data access
//
// v5 - Thread safety policies REMOVED (simplification):
// - Single Tensor class (no policy templates)
// - Views hold shared_ptr to state (prevents dangling)
// - No mutex/locking machinery
//
// Thread safety contract:
// - Tensors are NOT thread-safe for concurrent mutation
// - Session::Run() is thread-safe (TensorFlow's guarantee)
// - For multi-threaded serving, each request should have its own tensors

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <complex>
#include <concepts>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <source_location>
#include <string>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

extern "C" {
#include <tensorflow/c/c_api.h>
#include <tensorflow/c/tf_tstring.h>
}

#include "tf_wrap/format.hpp"
#include "tf_wrap/status.hpp"

// ============================================================================
// P0 Safety: Platform bool size check
// TensorFlow's TF_BOOL requires 1-byte bool for correct memory layout.
// This static_assert prevents silent data corruption on non-conforming platforms.
// ============================================================================
static_assert(sizeof(bool) == 1,
    "TensorFlowWrap requires sizeof(bool) == 1 for TF_BOOL compatibility. "
    "Your platform has a non-standard bool size which would cause data corruption.");

namespace tf_wrap {

// ============================================================================
// Helpers
// ============================================================================

namespace detail {
    template<class>
    inline constexpr bool always_false_v = false;

    [[nodiscard]] inline int checked_int(std::size_t v,
                                        const char* context = "size conversion") {
        if (v > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error(tf_wrap::detail::format(
                "Integer overflow in {}: {} exceeds int max",
                context, v));
        }
        return static_cast<int>(v);
    }

    [[nodiscard]] inline std::size_t checked_mul(std::size_t a, std::size_t b,
                                                  const char* context = "size calculation") {
        if (a == 0 || b == 0) return 0;
        if (b > std::numeric_limits<std::size_t>::max() / a) {
            throw std::overflow_error(tf_wrap::detail::format(
                "Integer overflow in {}: {} * {} exceeds size_t max",
                context, a, b));
        }
        return a * b;
    }

    [[nodiscard]] inline std::size_t checked_product(std::span<const std::int64_t> dims,
                                                      const char* context = "shape calculation") {
        std::size_t result = 1;
        for (auto d : dims) {
            if (d < 0) {
                throw std::invalid_argument(tf_wrap::detail::format(
                    "Negative dimension {} in {}", d, context));
            }
            result = checked_mul(result, static_cast<std::size_t>(d), context);
        }
        return result;
    }

    [[nodiscard]] inline const std::vector<std::int64_t>& empty_shape() noexcept {
        static const std::vector<std::int64_t> kEmpty;
        return kEmpty;
    }

    [[nodiscard]] inline std::string shape_to_string(std::span<const std::int64_t> dims) {
        if (dims.empty()) return "[]";
        std::string out;
        out.reserve(dims.size() * 4 + 2);
        out.push_back('[');
        for (std::size_t i = 0; i < dims.size(); ++i) {
            if (i != 0) out.push_back(',');
            out += std::to_string(dims[i]);
        }
        out.push_back(']');
        return out;
    }
} // namespace detail

// ============================================================================
// Concepts
// ============================================================================

template<class T>
concept TensorScalar =
    std::same_as<T, float>         ||
    std::same_as<T, double>        ||
    std::same_as<T, std::int8_t>   ||
    std::same_as<T, std::int16_t>  ||
    std::same_as<T, std::int32_t>  ||
    std::same_as<T, std::int64_t>  ||
    std::same_as<T, std::uint8_t>  ||
    std::same_as<T, std::uint16_t> ||
    std::same_as<T, std::uint32_t> ||
    std::same_as<T, std::uint64_t> ||
    std::same_as<T, bool>          ||
    std::same_as<T, std::complex<float>>  ||
    std::same_as<T, std::complex<double>>;

// ============================================================================
// Tensor Views
// ============================================================================

namespace detail {

struct TensorState {
    TF_Tensor* tensor{nullptr};
    std::vector<std::int64_t> shape;
    // Optional keepalive state for zero-copy tensor views (e.g. reshape).
    // When set, this state is kept alive for the lifetime of this TensorState.
    std::shared_ptr<TensorState> keepalive;

    TensorState() = default;

    TensorState(TF_Tensor* t, std::vector<std::int64_t> s)
        : tensor(t), shape(std::move(s)) {}

    ~TensorState() {
        if (tensor) TF_DeleteTensor(tensor);
    }

    TensorState(const TensorState&) = delete;
    TensorState& operator=(const TensorState&) = delete;
};

} // namespace detail

template<class T>
class ReadView {
public:
    ReadView() = default;

    ReadView(std::shared_ptr<detail::TensorState> st, std::span<const T> sp)
        : state_(std::move(st))
        , span_(sp)
    {}

    [[nodiscard]] std::span<const T> span() const noexcept { return span_; }
    [[nodiscard]] const T* data() const noexcept { return span_.data(); }
    [[nodiscard]] std::size_t size() const noexcept { return span_.size(); }

    [[nodiscard]] const T& operator[](std::size_t i) const noexcept { return span_[i]; }
    [[nodiscard]] const T& at(std::size_t i) const {
        if (i >= span_.size()) throw std::out_of_range("ReadView::at");
        return span_[i];
    }

    [[nodiscard]] const T* begin() const noexcept { return span_.data(); }
    [[nodiscard]] const T* end() const noexcept { return span_.data() + span_.size(); }

private:
    std::shared_ptr<detail::TensorState> state_;
    std::span<const T> span_;
};

template<class T>
class WriteView {
public:
    WriteView() = default;

    WriteView(std::shared_ptr<detail::TensorState> st, std::span<T> sp)
        : state_(std::move(st))
        , span_(sp)
    {}

    [[nodiscard]] std::span<T> span() const noexcept { return span_; }
    [[nodiscard]] T* data() const noexcept { return span_.data(); }
    [[nodiscard]] std::size_t size() const noexcept { return span_.size(); }

    [[nodiscard]] T& operator[](std::size_t i) const noexcept { return span_[i]; }
    [[nodiscard]] T& at(std::size_t i) const {
        if (i >= span_.size()) throw std::out_of_range("WriteView::at");
        return span_[i];
    }

    [[nodiscard]] T* begin() const noexcept { return span_.data(); }
    [[nodiscard]] T* end() const noexcept { return span_.data() + span_.size(); }

private:
    std::shared_ptr<detail::TensorState> state_;
    std::span<T> span_;
};

// ============================================================================
// Tensor
// ============================================================================

class Tensor {
public:
    Tensor() : state_(std::make_shared<detail::TensorState>()) {}
    ~Tensor() = default;

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept
        : state_(std::exchange(other.state_, moved_from_state())) {}

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            state_ = std::exchange(other.state_, moved_from_state());
        }
        return *this;
    }

    // ────────────────────────────────────────
    // Basic properties
    // ────────────────────────────────────────

    [[nodiscard]] const std::vector<std::int64_t>& shape() const noexcept {
        return state_ ? state_->shape : detail::empty_shape();
    }

    [[nodiscard]] int rank() const noexcept {
        return state_ ? static_cast<int>(state_->shape.size()) : 0;
    }

    [[nodiscard]] TF_DataType dtype() const {
        if (!state_ || !state_->tensor) {
            throw std::runtime_error("dtype() called on empty tensor");
        }
        return TF_TensorType(state_->tensor);
    }

    [[nodiscard]] const char* dtype_name() const {
        return TF_DataTypeString(dtype());
    }

    [[nodiscard]] std::size_t byte_size() const noexcept {
        return (state_ && state_->tensor) ? TF_TensorByteSize(state_->tensor) : 0;
    }

    [[nodiscard]] std::size_t num_elements() const {
        if (!state_ || !state_->tensor) return 0;
        const std::int64_t n = TF_TensorElementCount(state_->tensor);
        if (n < 0) throw std::runtime_error("TF_TensorElementCount returned negative");
        return static_cast<std::size_t>(n);
    }

    [[nodiscard]] bool empty() const noexcept {
        if (!state_ || !state_->tensor) return true;
        const std::int64_t n = TF_TensorElementCount(state_->tensor);
        return n <= 0;
    }

    [[nodiscard]] bool valid() const noexcept {
        return state_ && state_->tensor != nullptr;
    }

    [[nodiscard]] explicit operator bool() const noexcept {
        return state_ && state_->tensor != nullptr;
    }

    [[nodiscard]] TF_Tensor* handle() const noexcept {
        return state_ ? state_->tensor : nullptr;
    }

    /// True if tensor shape exactly matches `dims`.
    [[nodiscard]] bool matches_shape(std::span<const std::int64_t> dims) const noexcept {
        const auto& s = shape();
        if (s.size() != dims.size()) return false;
        for (std::size_t i = 0; i < dims.size(); ++i) {
            if (s[i] != dims[i]) return false;
        }
        return true;
    }

    // ────────────────────────────────────────
    // Reshape
    // ────────────────────────────────────────

    [[nodiscard]] Tensor reshape(std::span<const std::int64_t> new_dims,
                                std::source_location loc = std::source_location::current()) const
    {
        ensure_tensor_("reshape");

        const std::size_t old_elems = num_elements();
        const std::size_t new_elems = detail::checked_product(new_dims, "Tensor::reshape");
        if (new_elems != old_elems) {
            throw std::invalid_argument(tf_wrap::detail::format(
                "Tensor::reshape at {}:{}: element count mismatch: old shape {} ({} elems) vs new shape {} ({} elems)",
                loc.file_name(), loc.line(),
                detail::shape_to_string(shape()), old_elems,
                detail::shape_to_string(new_dims), new_elems));
        }

        Tensor t;
        std::vector<std::int64_t> shape_vec(new_dims.begin(), new_dims.end());

        // Create a view tensor that aliases the same buffer
        TF_Tensor* view = TF_NewTensor(
            TF_TensorType(state_->tensor),
            shape_vec.data(),
            detail::checked_int(shape_vec.size(), "Tensor::reshape ndims"),
            TF_TensorData(state_->tensor),
            TF_TensorByteSize(state_->tensor),
            [](void*, std::size_t, void*) {},
            nullptr);

        if (!view) {
            throw std::bad_alloc();
        }

        t.state_->tensor = view;
        t.state_->shape = std::move(shape_vec);
        t.state_->keepalive = state_;
        return t;
    }

    // ────────────────────────────────────────
    // Conversions
    // ────────────────────────────────────────

    template<TensorScalar T>
    [[nodiscard]] std::vector<T> ToVector() const {
        ensure_tensor_("ToVector");
        ensure_dtype_<T>("ToVector");
        auto view = read<T>();
        return std::vector<T>(view.begin(), view.end());
    }

    template<TensorScalar T>
    [[nodiscard]] T ToScalar() const {
        ensure_tensor_("ToScalar");
        ensure_dtype_<T>("ToScalar");
        auto view = read<T>();
        if (view.size() != 1) {
            throw std::runtime_error(tf_wrap::detail::format(
                "ToScalar expected 1 element, got {}", view.size()));
        }
        return view[0];
    }

    // ────────────────────────────────────────
    // Typed views
    // ────────────────────────────────────────

    template<TensorScalar T>
    [[nodiscard]] ReadView<T> read() const {
        ensure_tensor_("read");
        ensure_dtype_<T>("read");
        const auto n = num_elements();
        const T* ptr = static_cast<const T*>(TF_TensorData(state_->tensor));
        return ReadView<T>(state_, std::span<const T>(ptr, n));
    }

    template<TensorScalar T>
    [[nodiscard]] WriteView<T> write() {
        ensure_tensor_("write");
        ensure_dtype_<T>("write");
        const auto n = num_elements();
        T* ptr = static_cast<T*>(TF_TensorData(state_->tensor));
        return WriteView<T>(state_, std::span<T>(ptr, n));
    }

    template<TensorScalar T>
    void fill(const T& value) {
        auto v = write<T>();
        std::fill(v.begin(), v.end(), value);
    }

    template<TensorScalar T, class Fn>
    void with_read(Fn&& fn) const {
        auto v = read<T>();
        std::forward<Fn>(fn)(v.span());
    }

    template<TensorScalar T, class Fn>
    void with_write(Fn&& fn) {
        auto v = write<T>();
        std::forward<Fn>(fn)(v.span());
    }

    template<TensorScalar T>
    [[nodiscard]] T* data() {
        ensure_tensor_("data");
        ensure_dtype_<T>("data");
        return static_cast<T*>(TF_TensorData(state_->tensor));
    }

    template<TensorScalar T>
    [[nodiscard]] const T* data() const {
        ensure_tensor_("data");
        ensure_dtype_<T>("data");
        return static_cast<const T*>(TF_TensorData(state_->tensor));
    }

    // ────────────────────────────────────────
    // Construction helpers
    // ────────────────────────────────────────

    template<TensorScalar T>
    [[nodiscard]] static Tensor FromVector(std::span<const std::int64_t> dims,
                                           const std::vector<T>& values)
    {
        return FromVector<T>(dims, std::span<const T>(values.data(), values.size()));
    }

    template<TensorScalar T>
    [[nodiscard]] static Tensor FromVector(std::span<const std::int64_t> dims,
                                           std::initializer_list<T> values)
    {
        return FromVector<T>(dims, std::span<const T>(values.begin(), values.size()));
    }

    template<TensorScalar T>
    [[nodiscard]] static Tensor FromVector(std::span<const std::int64_t> dims,
                                           std::span<const T> values)
    {
        const std::size_t n = detail::checked_product(dims, "Tensor::FromVector");
        if (values.size() != n) {
            throw std::invalid_argument(tf_wrap::detail::format(
                "Tensor::FromVector: value count mismatch (expected {}, got {})",
                n, values.size()));
        }

        Tensor t;
        t.state_->shape.assign(dims.begin(), dims.end());

        TF_Tensor* raw = TF_AllocateTensor(
            dtype_of_<T>(),
            t.state_->shape.data(),
            detail::checked_int(t.state_->shape.size(), "Tensor::FromVector ndims"),
            n * sizeof(T));

        if (!raw) throw std::bad_alloc();

        t.state_->tensor = raw;
        std::memcpy(TF_TensorData(raw), values.data(), n * sizeof(T));
        return t;
    }

    [[nodiscard]] static Tensor FromRaw(TF_Tensor* tensor) {
        if (!tensor) return Tensor{};
        Tensor t;
        t.state_->tensor = tensor;

        const int ndims = TF_NumDims(tensor);
        t.state_->shape.reserve(static_cast<std::size_t>(ndims));
        for (int i = 0; i < ndims; ++i) {
            t.state_->shape.push_back(TF_Dim(tensor, i));
        }
        return t;
    }

private:
    static std::shared_ptr<detail::TensorState> moved_from_state() noexcept {
        return {};
    }

    void ensure_tensor_(const char* fn) const {
        if (!state_ || !state_->tensor) {
            throw std::runtime_error(tf_wrap::detail::format(
                "Tensor::{}(): tensor is null/empty", fn));
        }
    }

    template<TensorScalar T>
    void ensure_dtype_(const char* context) const {
        const TF_DataType got = dtype();
        const TF_DataType want = dtype_of_<T>();
        if (got != want) {
            throw std::runtime_error(tf_wrap::detail::format(
                "Tensor dtype mismatch in {}: got {}, want {}",
                context, TF_DataTypeString(got), TF_DataTypeString(want)));
        }
    }

    template<TensorScalar T>
    [[nodiscard]] static TF_DataType dtype_of_() {
        if constexpr (std::same_as<T, float>) return TF_FLOAT;
        else if constexpr (std::same_as<T, double>) return TF_DOUBLE;
        else if constexpr (std::same_as<T, std::int8_t>) return TF_INT8;
        else if constexpr (std::same_as<T, std::int16_t>) return TF_INT16;
        else if constexpr (std::same_as<T, std::int32_t>) return TF_INT32;
        else if constexpr (std::same_as<T, std::int64_t>) return TF_INT64;
        else if constexpr (std::same_as<T, std::uint8_t>) return TF_UINT8;
        else if constexpr (std::same_as<T, std::uint16_t>) return TF_UINT16;
        else if constexpr (std::same_as<T, std::uint32_t>) return TF_UINT32;
        else if constexpr (std::same_as<T, std::uint64_t>) return TF_UINT64;
        else if constexpr (std::same_as<T, bool>) return TF_BOOL;
        else if constexpr (std::same_as<T, std::complex<float>>) return TF_COMPLEX64;
        else if constexpr (std::same_as<T, std::complex<double>>) return TF_COMPLEX128;
        else static_assert(detail::always_false_v<T>, "Unsupported tensor scalar type");
    }

private:
    std::shared_ptr<detail::TensorState> state_;
};

} // namespace tf_wrap
