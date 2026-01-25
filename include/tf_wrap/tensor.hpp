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
}

// ============================================================================
// TensorScalar Concept
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
// Type mapping: C++ type -> TF_DataType
// ============================================================================

template<TensorScalar T>
[[nodiscard]] constexpr TF_DataType tf_dtype_of() noexcept {
    if constexpr (std::same_as<T, float>)              return TF_FLOAT;
    else if constexpr (std::same_as<T, double>)        return TF_DOUBLE;
    else if constexpr (std::same_as<T, std::int8_t>)   return TF_INT8;
    else if constexpr (std::same_as<T, std::int16_t>)  return TF_INT16;
    else if constexpr (std::same_as<T, std::int32_t>)  return TF_INT32;
    else if constexpr (std::same_as<T, std::int64_t>)  return TF_INT64;
    else if constexpr (std::same_as<T, std::uint8_t>)  return TF_UINT8;
    else if constexpr (std::same_as<T, std::uint16_t>) return TF_UINT16;
    else if constexpr (std::same_as<T, std::uint32_t>) return TF_UINT32;
    else if constexpr (std::same_as<T, std::uint64_t>) return TF_UINT64;
    else if constexpr (std::same_as<T, bool>)          return TF_BOOL;
    else if constexpr (std::same_as<T, std::complex<float>>)  return TF_COMPLEX64;
    else if constexpr (std::same_as<T, std::complex<double>>) return TF_COMPLEX128;
    else static_assert(detail::always_false_v<T>, "Unsupported scalar type");
}

template<TensorScalar T>
inline constexpr TF_DataType tf_dtype_v = tf_dtype_of<T>();

[[nodiscard]] constexpr const char* dtype_name(TF_DataType dtype) noexcept {
    switch (dtype) {
        case TF_FLOAT:      return "float32";
        case TF_DOUBLE:     return "float64";
        case TF_INT8:       return "int8";
        case TF_INT16:      return "int16";
        case TF_INT32:      return "int32";
        case TF_INT64:      return "int64";
        case TF_UINT8:      return "uint8";
        case TF_UINT16:     return "uint16";
        case TF_UINT32:     return "uint32";
        case TF_UINT64:     return "uint64";
        case TF_BOOL:       return "bool";
        case TF_STRING:     return "string";
        case TF_COMPLEX64:  return "complex64";
        case TF_COMPLEX128: return "complex128";
        case TF_BFLOAT16:   return "bfloat16";
        case TF_HALF:       return "float16";
        default:            return "unknown";
    }
}

// ============================================================================
// TensorState - Internal state (shared_ptr enables safe view lifetime)
// ============================================================================

struct TensorState {
    TF_Tensor* tensor{nullptr};
    std::vector<std::int64_t> shape;
    
    TensorState() = default;
    
    TensorState(TF_Tensor* t, std::vector<std::int64_t> s)
        : tensor(t), shape(std::move(s)) {}
    
    ~TensorState() {
        if (tensor) TF_DeleteTensor(tensor);
    }
    
    TensorState(const TensorState&) = delete;
    TensorState& operator=(const TensorState&) = delete;
};

// ============================================================================
// TensorView - View that keeps tensor alive via shared_ptr
// ============================================================================

template<class T>
    requires TensorScalar<std::remove_cv_t<T>>
class TensorView {
public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = pointer;
    using const_iterator = const_pointer;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    TensorView(std::shared_ptr<TensorState> state, std::span<T> data) noexcept
        : state_(std::move(state)), span_(data) {}

    TensorView(const TensorView&) noexcept = default;
    TensorView& operator=(const TensorView&) noexcept = default;
    TensorView(TensorView&&) noexcept = default;
    TensorView& operator=(TensorView&&) noexcept = default;
    ~TensorView() = default;

    [[nodiscard]] constexpr reference operator[](size_type i) const noexcept { return span_[i]; }
    
    [[nodiscard]] constexpr reference at(size_type i) const {
        if (i >= size()) throw std::out_of_range("TensorView::at: index out of range");
        return span_[i];
    }

    [[nodiscard]] constexpr reference front() const noexcept { return span_.front(); }
    [[nodiscard]] constexpr reference back() const noexcept { return span_.back(); }
    [[nodiscard]] constexpr pointer data() const noexcept { return span_.data(); }
    [[nodiscard]] constexpr std::span<T> span() const noexcept { return span_; }
    // NOTE: No implicit conversion to std::span to avoid accidental dangling spans.
    // Use .span() explicitly when you really want a span that does NOT keep the tensor alive.

    [[nodiscard]] constexpr iterator begin() const noexcept { return span_.data(); }
    [[nodiscard]] constexpr iterator end() const noexcept { return span_.data() + span_.size(); }
    [[nodiscard]] constexpr const_iterator cbegin() const noexcept { return begin(); }
    [[nodiscard]] constexpr const_iterator cend() const noexcept { return end(); }
    [[nodiscard]] constexpr reverse_iterator rbegin() const noexcept { return reverse_iterator(end()); }
    [[nodiscard]] constexpr reverse_iterator rend() const noexcept { return reverse_iterator(begin()); }
    [[nodiscard]] constexpr const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }
    [[nodiscard]] constexpr const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }

    [[nodiscard]] constexpr size_type size() const noexcept { return span_.size(); }
    [[nodiscard]] constexpr size_type size_bytes() const noexcept { return span_.size_bytes(); }
    [[nodiscard]] constexpr bool empty() const noexcept { return span_.empty(); }

private:
    std::shared_ptr<TensorState> state_;
    std::span<T> span_;
};

// ============================================================================
// Tensor - RAII wrapper for TF_Tensor
// ============================================================================

class Tensor {
public:
    template<TensorScalar T>
    using ReadView = TensorView<const T>;

    template<TensorScalar T>
    using WriteView = TensorView<T>;

    // ─────────────────────────────────────────────────────────────────
    // Constructors
    // ─────────────────────────────────────────────────────────────────

    Tensor() : state_(std::make_shared<TensorState>()) {}
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

    // ─────────────────────────────────────────────────────────────────
    // Queries
    // ─────────────────────────────────────────────────────────────────

    [[nodiscard]] const std::vector<std::int64_t>& shape() const noexcept { return state_->shape; }
    [[nodiscard]] int rank() const noexcept { return static_cast<int>(state_->shape.size()); }
    
    [[nodiscard]] TF_DataType dtype() const { 
        if (!state_->tensor) {
            throw std::runtime_error("dtype(): tensor is null (moved-from or empty)");
        }
        return TF_TensorType(state_->tensor); 
    }
    [[nodiscard]] const char* dtype_name() const { return tf_wrap::dtype_name(dtype()); }
    [[nodiscard]] std::size_t byte_size() const noexcept { return state_->tensor ? TF_TensorByteSize(state_->tensor) : 0; }
    
    [[nodiscard]] std::size_t num_elements() const {
        if (!state_->tensor) return 0;
        const std::int64_t n = TF_TensorElementCount(state_->tensor);
        if (n < 0) throw std::runtime_error("TF_TensorElementCount returned negative value");
        return static_cast<std::size_t>(n);
    }

    [[nodiscard]] bool empty() const noexcept {
        if (!state_->tensor) return true;
        const std::int64_t n = TF_TensorElementCount(state_->tensor);
        return n <= 0;
    }
    [[nodiscard]] bool valid() const noexcept { return state_->tensor != nullptr; }
    [[nodiscard]] explicit operator bool() const noexcept { return state_->tensor != nullptr; }
    [[nodiscard]] TF_Tensor* handle() const noexcept { return state_->tensor; }

    // ─────────────────────────────────────────────────────────────────
    // Data extraction
    // ─────────────────────────────────────────────────────────────────

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
        if (num_elements() != 1) {
            throw std::runtime_error(tf_wrap::detail::format(
                "Tensor::ToScalar(): expected 1 element, got {}", num_elements()));
        }
        return read<T>()[0];
    }

    [[nodiscard]] Tensor Clone() const {
        if (!state_->tensor) return Tensor{};
        
        const auto dt = dtype();
        const auto& sh = shape();

        if (dt == TF_STRING) {
            const std::size_t n = num_elements();
            const TF_TString* src = static_cast<const TF_TString*>(TF_TensorData(state_->tensor));
            const std::size_t bytes = detail::checked_mul(n, sizeof(TF_TString), "Tensor::Clone");

            return create_tensor_alloc_(TF_STRING, sh, bytes,
                [src, n](void* dst, std::size_t) {
                    auto* out = static_cast<TF_TString*>(dst);
                    for (std::size_t i = 0; i < n; ++i) {
                        TF_TString_Init(&out[i]);
                        const char* p = TF_TString_GetDataPointer(&src[i]);
                        const std::size_t sz = p ? TF_TString_GetSize(&src[i]) : 0;
                        TF_TString_Copy(&out[i], p ? p : "", sz);
                    }
                });
        }

        const auto bytes = byte_size();
        const void* src = TF_TensorData(state_->tensor);
        
        return create_tensor_alloc_(dt, sh, bytes,
            [src](void* dst, std::size_t len) {
                if (len != 0) std::memcpy(dst, src, len);
            });
    }

    // ─────────────────────────────────────────────────────────────────
    // Data access (views keep tensor alive)
    // ─────────────────────────────────────────────────────────────────

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

    template<TensorScalar T, class Fn>
    decltype(auto) with_read(Fn&& fn) const {
        return std::forward<Fn>(fn)(read<T>().span());
    }

    template<TensorScalar T, class Fn>
    decltype(auto) with_write(Fn&& fn) {
        return std::forward<Fn>(fn)(write<T>().span());
    }

    // ─────────────────────────────────────────────────────────────────
    // UNSAFE: Raw pointer access (lifetime footgun!)
    // ─────────────────────────────────────────────────────────────────
    // WARNING: The returned pointer is only valid while this Tensor exists.
    // Prefer read<T>()/write<T>() which return views that keep the tensor alive.
    // Use these only when interfacing with C APIs that require raw pointers.

    // Alias for unsafe_data<T>() to match common container conventions.
    // WARNING: The returned pointer is only valid while this Tensor exists.
    template<TensorScalar T>
    [[nodiscard]] T* data() { return unsafe_data<T>(); }

    template<TensorScalar T>
    [[nodiscard]] const T* data() const { return unsafe_data<T>(); }

    template<TensorScalar T>
    [[nodiscard]] T* unsafe_data() {
        ensure_tensor_("unsafe_data");
        ensure_dtype_<T>("unsafe_data");
        return static_cast<T*>(TF_TensorData(state_->tensor));
    }

    template<TensorScalar T>
    [[nodiscard]] const T* unsafe_data() const {
        ensure_tensor_("unsafe_data");
        ensure_dtype_<T>("unsafe_data");
        return static_cast<const T*>(TF_TensorData(state_->tensor));
    }

    // ─────────────────────────────────────────────────────────────────
    // Factory: FromVector
    // ─────────────────────────────────────────────────────────────────

    template<TensorScalar T>
    [[nodiscard]] static Tensor FromVector(
        std::span<const std::int64_t> dims,
        const std::vector<T>& values,
        std::source_location loc = std::source_location::current())
    {
        const std::size_t expected = detail::checked_product(dims, "Tensor::FromVector");
        if (expected != values.size()) {
            throw std::invalid_argument(tf_wrap::detail::format(
                "Tensor::FromVector at {}:{}: shape requires {} elements, got {}",
                loc.file_name(), loc.line(), expected, values.size()));
        }

        const std::size_t bytes = detail::checked_mul(expected, sizeof(T), "Tensor::FromVector");
        return create_tensor_alloc_(tf_dtype_v<T>, dims, bytes,
            [&](void* dst, std::size_t len) {
                if (len != 0) {
                    if constexpr (std::same_as<T, bool>) {
                        bool* out = static_cast<bool*>(dst);
                        for (std::size_t i = 0; i < values.size(); ++i) out[i] = values[i];
                    } else {
                        std::memcpy(dst, values.data(), len);
                    }
                }
            });
    }

    template<TensorScalar T>
    [[nodiscard]] static Tensor FromVector(
        std::initializer_list<std::int64_t> dims,
        std::initializer_list<T> values,
        std::source_location loc = std::source_location::current())
    {
        return FromVector<T>(std::vector<std::int64_t>(dims), std::vector<T>(values), loc);
    }

    template<TensorScalar T>
    [[nodiscard]] static Tensor FromVector(
        std::initializer_list<std::int64_t> dims,
        const std::vector<T>& values,
        std::source_location loc = std::source_location::current())
    {
        return FromVector<T>(std::vector<std::int64_t>(dims), values, loc);
    }

    // ─────────────────────────────────────────────────────────────────
    // Factory: FromScalar
    // ─────────────────────────────────────────────────────────────────

    template<TensorScalar T>
    [[nodiscard]] static Tensor FromScalar(T value) {
        return create_tensor_alloc_(tf_dtype_v<T>, std::span<const std::int64_t>{}, sizeof(T),
            [&](void* dst, std::size_t len) {
                if (len != 0) std::memcpy(dst, &value, len);
            });
    }

    // ─────────────────────────────────────────────────────────────────
    // Factory: FromString / ToString
    // ─────────────────────────────────────────────────────────────────

    [[nodiscard]] static Tensor FromString(const std::string& str) {
        TF_Tensor* tensor = TF_AllocateTensor(TF_STRING, nullptr, 0, sizeof(TF_TString));
        if (!tensor) throw std::runtime_error("Tensor::FromString: TF_AllocateTensor failed");
        
        TF_TString* tstr = static_cast<TF_TString*>(TF_TensorData(tensor));
        TF_TString_Init(tstr);
        TF_TString_Copy(tstr, str.data(), str.size());
        return FromRaw(tensor);
    }

    [[nodiscard]] std::string ToString() const {
        ensure_tensor_("ToString");
        if (dtype() != TF_STRING) {
            throw std::runtime_error(tf_wrap::detail::format(
                "Tensor::ToString(): expected TF_STRING dtype, got {}", dtype_name()));
        }
        if (num_elements() != 1) {
            throw std::runtime_error(tf_wrap::detail::format(
                "Tensor::ToString(): expected scalar string tensor, got {} elements", num_elements()));
        }
        const TF_TString* tstr = static_cast<const TF_TString*>(TF_TensorData(state_->tensor));
        return std::string(TF_TString_GetDataPointer(tstr), TF_TString_GetSize(tstr));
    }

    // ─────────────────────────────────────────────────────────────────
    // Factory: FromRaw
    // ─────────────────────────────────────────────────────────────────

    [[nodiscard]] static Tensor FromRaw(TF_Tensor* raw) {
        if (!raw) throw std::invalid_argument("Tensor::FromRaw: null TF_Tensor*");

        struct RawTensorGuard {
            TF_Tensor* t{nullptr};
            ~RawTensorGuard() {
                if (t) {
                    TF_DeleteTensor(t);
                }
            }
            TF_Tensor* release() noexcept {
                TF_Tensor* out = t;
                t = nullptr;
                return out;
            }
        } guard{raw};

        Tensor t;
        const int ndims = TF_NumDims(raw);
        if (ndims < 0) {
            throw std::runtime_error("Tensor::FromRaw: TF_NumDims returned negative");
        }
        t.state_->shape.reserve(static_cast<std::size_t>(ndims));
        for (int i = 0; i < ndims; ++i) {
            t.state_->shape.push_back(TF_Dim(raw, i));
        }
        t.state_->tensor = guard.release();
        return t;
    }

    // ─────────────────────────────────────────────────────────────────
    // Factory: Allocate / Zeros
    // ─────────────────────────────────────────────────────────────────

    template<TensorScalar T>
    [[nodiscard]] static Tensor Allocate(std::span<const std::int64_t> dims) {
        const std::size_t num_elems = detail::checked_product(dims, "Tensor::Allocate");
        const std::size_t bytes = detail::checked_mul(num_elems, sizeof(T), "Tensor::Allocate");
        return create_tensor_alloc_(tf_dtype_v<T>, dims, bytes, [](void*, std::size_t) {});
    }

    template<TensorScalar T>
    [[nodiscard]] static Tensor Allocate(std::initializer_list<std::int64_t> dims) {
        return Allocate<T>(std::vector<std::int64_t>(dims));
    }

    template<TensorScalar T>
    [[nodiscard]] static Tensor Zeros(std::span<const std::int64_t> dims) {
        const std::size_t num_elems = detail::checked_product(dims, "Tensor::Zeros");
        const std::size_t bytes = detail::checked_mul(num_elems, sizeof(T), "Tensor::Zeros");
        return create_tensor_alloc_(tf_dtype_v<T>, dims, bytes,
            [](void* dst, std::size_t len) { if (len != 0) std::memset(dst, 0, len); });
    }

    template<TensorScalar T>
    [[nodiscard]] static Tensor Zeros(std::initializer_list<std::int64_t> dims) {
        return Zeros<T>(std::vector<std::int64_t>(dims));
    }

    // ─────────────────────────────────────────────────────────────────
    // Factory: AdoptMalloc / Adopt
    // ─────────────────────────────────────────────────────────────────

    template<TensorScalar T>
    [[nodiscard]] static Tensor AdoptMalloc(
        std::span<const std::int64_t> dims,
        void* data_ptr,
        std::size_t byte_len)
    {
        if (!data_ptr && byte_len > 0) {
            throw std::invalid_argument("Tensor::AdoptMalloc: null data with non-zero byte_len");
        }
        const std::size_t num_elems = detail::checked_product(dims, "Tensor::AdoptMalloc");
        const std::size_t expected_bytes = detail::checked_mul(num_elems, sizeof(T), "Tensor::AdoptMalloc");
        if (expected_bytes != byte_len) {
            throw std::invalid_argument(tf_wrap::detail::format(
                "Tensor::AdoptMalloc: byte_len mismatch - expected {} but got {}",
                expected_bytes, byte_len));
        }
        return create_tensor_adopt_(tf_dtype_v<T>, dims, data_ptr, byte_len, &default_deallocator, nullptr);
    }

    using Deallocator = void (*)(void* data, std::size_t len, void* arg);

    [[nodiscard]] static Tensor Adopt(
        TF_DataType dtype,
        std::span<const std::int64_t> dims,
        void* data_ptr,
        std::size_t byte_len,
        Deallocator deallocator,
        void* deallocator_arg = nullptr)
    {
        if (!deallocator) {
            throw std::invalid_argument("Tensor::Adopt: deallocator is required");
        }
        if (!data_ptr && byte_len > 0) {
            throw std::invalid_argument("Tensor::Adopt: null data with non-zero byte_len");
        }

        const std::size_t dtype_size = TF_DataTypeSize(dtype);
        if (dtype_size == 0) {
            throw std::invalid_argument("Tensor::Adopt: variable-length dtype not supported");
        }

        const std::size_t num_elems = detail::checked_product(dims, "Tensor::Adopt");
        const std::size_t expected_bytes = detail::checked_mul(num_elems, dtype_size, "Tensor::Adopt");
        if (expected_bytes != byte_len) {
            throw std::invalid_argument(tf_wrap::detail::format(
                "Tensor::Adopt: byte_len mismatch - expected {} but got {}",
                expected_bytes, byte_len));
        }
        return create_tensor_adopt_(dtype, dims, data_ptr, byte_len, deallocator, deallocator_arg);
    }

private:
    std::shared_ptr<TensorState> state_;
    
    // Returns a shared empty state for moved-from tensors.
    // Initialized once, shared by all moved-from instances.
    // This avoids allocation in noexcept move operations.
    static std::shared_ptr<TensorState> moved_from_state() noexcept {
        static auto instance = std::make_shared<TensorState>();
        return instance;
    }

    template<class InitFn>
    [[nodiscard]] static Tensor create_tensor_alloc_(
        TF_DataType dtype,
        std::span<const std::int64_t> dims,
        std::size_t byte_len,
        InitFn&& init)
    {
        Tensor t;
        std::vector<std::int64_t> shape_vec(dims.begin(), dims.end());
        const int64_t* dims_ptr = shape_vec.empty() ? nullptr : shape_vec.data();
        const int num_dims = detail::checked_int(shape_vec.size(), "Tensor::Allocate num_dims");

        TF_Tensor* tensor = TF_AllocateTensor(dtype, dims_ptr, num_dims, byte_len);
        if (!tensor) throw std::runtime_error("TF_AllocateTensor: failed to create tensor");

        if (byte_len != 0) {
            void* dst = TF_TensorData(tensor);
            try {
                init(dst, byte_len);
            } catch (...) {
                TF_DeleteTensor(tensor);
                throw;
            }
        }

        t.state_->tensor = tensor;
        t.state_->shape = std::move(shape_vec);
        return t;
    }

    [[nodiscard]] static Tensor create_tensor_adopt_(
        TF_DataType dtype,
        std::span<const std::int64_t> dims,
        void* data_ptr,
        std::size_t byte_len,
        Deallocator deallocator,
        void* deallocator_arg)
    {
        struct DataGuard {
            void* data; std::size_t len; Deallocator dealloc; void* arg; bool released{false};
            ~DataGuard() { if (!released && dealloc && data) dealloc(data, len, arg); }
            void release() noexcept { released = true; }
        };
        
        DataGuard guard{data_ptr, byte_len, deallocator, deallocator_arg};
        
        Tensor t;
        std::vector<std::int64_t> shape_vec(dims.begin(), dims.end());
        const int64_t* dims_ptr = shape_vec.empty() ? nullptr : shape_vec.data();
        const int num_dims = detail::checked_int(shape_vec.size(), "Tensor::Adopt num_dims");

        TF_Tensor* tensor = TF_NewTensor(dtype, dims_ptr, num_dims, data_ptr, byte_len,
                                          deallocator, deallocator_arg);
        if (!tensor) throw std::runtime_error("TF_NewTensor: failed to create tensor");

        guard.release();
        t.state_->tensor = tensor;
        t.state_->shape = std::move(shape_vec);
        return t;
    }

    void ensure_tensor_(const char* fn) const {
        if (!state_->tensor) {
            throw std::runtime_error(tf_wrap::detail::format("Tensor::{}(): tensor is null/empty", fn));
        }
    }

    template<TensorScalar T>
    void ensure_dtype_(const char* fn) const {
        const TF_DataType expected = tf_dtype_v<T>;
        const TF_DataType actual = dtype();
        if (actual != expected) {
            throw std::runtime_error(tf_wrap::detail::format(
                "Tensor::{}(): dtype mismatch - requested {} but tensor is {}",
                fn, tf_wrap::dtype_name(expected), tf_wrap::dtype_name(actual)));
        }
    }

    static void default_deallocator(void* data, std::size_t, void*) noexcept {
        std::free(data);
    }
};

// Backward compatibility alias

} // namespace tf_wrap
