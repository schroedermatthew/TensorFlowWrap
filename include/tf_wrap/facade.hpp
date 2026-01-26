// tf_wrap/facade.hpp
// Ergonomic layer for TensorFlowWrap - Production Inference Edition
//
// Goals:
// - Make the "right" production path the easiest path.
// - Resolve names once at startup; run with handles in the hot path.
// - Provide a compiled, allocation-minimizing Runner (signature) for serving.
//
// Key types:
// - RunnerBuilder: fluent builder used at startup to describe feeds/fetches/targets.
// - Runner: compiled signature (resolved TF_Output/TF_Operation* + reusable Context).
// - Model: high-level SavedModel facade.

#pragma once

#include <array>
#include <algorithm>
#include <cstddef>
#include <memory>
#include <span>
#include <source_location>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "tf_wrap/error.hpp"
#include "tf_wrap/format.hpp"
#include "tf_wrap/graph.hpp"
#include "tf_wrap/session.hpp"
#include "tf_wrap/tensor.hpp"
#include "tf_wrap/detail/raw_tensor_ptr.hpp"

namespace tf_wrap {
namespace facade {

// ============================================================================
// Runner (compiled signature)
// ============================================================================

class Runner {
public:
    /// Explicitly borrowed tensor handle (for interop).
    ///
    /// Use case (production inference): zero-copy integration with code that already
    /// produces TF_Tensor* (custom batching, foreign runtimes, legacy TF C API code).
    /// Keepalive can hold any shared state required to keep `tensor` valid for the
    /// duration of the call.
    struct BorrowedTensor {
        TF_Tensor* tensor{nullptr};
        std::shared_ptr<const void> keepalive{};

        BorrowedTensor() = default;
        explicit BorrowedTensor(TF_Tensor* t, std::shared_ptr<const void> ka = {})
            : tensor(t), keepalive(std::move(ka)) {}
    };

    /// Reusable buffers for zero/low-allocation serving loops.
    ///
    /// Thread safety: Context is NOT thread-safe. Use one Context per thread.
    class Context {
    public:
        Context() = default;

    private:
        friend class Runner;

        std::vector<TF_Tensor*> input_vals_;
        std::vector<std::shared_ptr<const void>> keepalives_;

        std::vector<TF_Tensor*> output_vals_;

        // RAII ownership for raw outputs until they are successfully wrapped.
        std::vector<detail::RawTensorPtr> owned_;
        std::vector<Tensor> outputs_;
    };

    Runner() = default;

    [[nodiscard]] std::size_t num_feeds() const noexcept { return feeds_.size(); }
    [[nodiscard]] std::size_t num_fetches() const noexcept { return fetches_.size(); }
    [[nodiscard]] std::size_t num_targets() const noexcept { return targets_.size(); }

    [[nodiscard]] Context make_context() const {
        Context ctx;
        ctx.input_vals_.reserve(feeds_.size());
        ctx.keepalives_.reserve(feeds_.size());

        ctx.output_vals_.assign(fetches_.size(), nullptr);
        ctx.owned_.reserve(fetches_.size());
        ctx.outputs_.reserve(fetches_.size());
        return ctx;
    }

    // ─────────────────────────────────────────────────────────────────
    // Hot path (recommended): reuse a Context
    // ─────────────────────────────────────────────────────────────────

    /// Run with Tensor inputs (safe path).
    ///
    /// Returns a view of outputs owned by the Context (valid until next run/reset
    /// on that same Context).
    [[nodiscard]] std::span<const Tensor> run(
        Context& ctx,
        std::span<const Tensor> inputs,
        std::source_location loc = std::source_location::current()) const
    {
        ensure_compiled_("Runner::run", loc);

        if (inputs.size() != feeds_.size()) {
            throw Error::Wrapper(
                TF_INVALID_ARGUMENT,
                "Runner::run",
                detail::format("expected {} inputs, got {}", feeds_.size(), inputs.size()),
                "",
                -1,
                loc);
        }

        ctx.input_vals_.clear();
        ctx.keepalives_.clear();

        ctx.input_vals_.reserve(feeds_.size());
        ctx.keepalives_.reserve(feeds_.size());

        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const Tensor& t = inputs[i];
            if (!t.handle()) {
                throw Error::Wrapper(
                    TF_INVALID_ARGUMENT,
                    "Runner::run",
                    "null tensor handle",
                    feed_op_name_(i),
                    static_cast<int>(feeds_[i].index),
                    loc);
            }
            ctx.input_vals_.push_back(t.handle());
        }

        return run_raw_(ctx, loc);
    }

    /// Variadic hot-path convenience: run(ctx, t1, t2, ..., tN)
    ///
    /// This avoids creating a temporary array/span at the call site while still
    /// using the same optimized internal path.
    template <typename... Ts>
        requires (sizeof...(Ts) >= 1 && (std::is_same_v<std::remove_cvref_t<Ts>, Tensor> && ...))
    [[nodiscard]] std::span<const Tensor> run(Context& ctx, const Ts&... inputs) const {
        return run(ctx, std::source_location::current(), inputs...);
    }

    /// Variadic hot-path with explicit source location.
    template <typename... Ts>
        requires (sizeof...(Ts) >= 1 && (std::is_same_v<std::remove_cvref_t<Ts>, Tensor> && ...))
    [[nodiscard]] std::span<const Tensor> run(
        Context& ctx,
        std::source_location loc,
        const Ts&... inputs) const
    {
        ensure_compiled_("Runner::run", loc);

        constexpr std::size_t N = sizeof...(Ts);
        if (N != feeds_.size()) {
            throw Error::Wrapper(
                TF_INVALID_ARGUMENT,
                "Runner::run",
                detail::format("expected {} inputs, got {}", feeds_.size(), N),
                "",
                -1,
                loc);
        }

        ctx.input_vals_.clear();
        ctx.keepalives_.clear();

        ctx.input_vals_.reserve(feeds_.size());
        ctx.keepalives_.reserve(feeds_.size());

        std::size_t i = 0;
        auto push = [&](const Tensor& t) {
            if (!t.handle()) {
                throw Error::Wrapper(
                    TF_INVALID_ARGUMENT,
                    "Runner::run",
                    "null tensor handle",
                    feed_op_name_(i),
                    static_cast<int>(feeds_[i].index),
                    loc);
            }
            ctx.input_vals_.push_back(t.handle());
            ++i;
        };
        (push(inputs), ...);

        return run_raw_(ctx, loc);
    }

    /// Run with explicitly borrowed TF_Tensor* inputs (interop path).
    [[nodiscard]] std::span<const Tensor> run(
        Context& ctx,
        std::span<const BorrowedTensor> inputs,
        std::source_location loc = std::source_location::current()) const
    {
        ensure_compiled_("Runner::run", loc);

        if (inputs.size() != feeds_.size()) {
            throw Error::Wrapper(
                TF_INVALID_ARGUMENT,
                "Runner::run",
                detail::format("expected {} inputs, got {}", feeds_.size(), inputs.size()),
                "",
                -1,
                loc);
        }

        ctx.input_vals_.clear();
        ctx.keepalives_.clear();

        ctx.input_vals_.reserve(feeds_.size());
        ctx.keepalives_.reserve(feeds_.size());

        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const BorrowedTensor& bt = inputs[i];
            if (!bt.tensor) {
                throw Error::Wrapper(
                    TF_INVALID_ARGUMENT,
                    "Runner::run",
                    "null borrowed tensor handle",
                    feed_op_name_(i),
                    static_cast<int>(feeds_[i].index),
                    loc);
            }
            ctx.input_vals_.push_back(bt.tensor);
            if (bt.keepalive) {
                ctx.keepalives_.push_back(bt.keepalive);
            }
        }

        return run_raw_(ctx, loc);
    }

    /// Variadic hot-path convenience for borrowed tensors:
    /// run(ctx, bt1, bt2, ..., btN)
    template <typename... Ts>
        requires (sizeof...(Ts) >= 1 && (std::is_same_v<std::remove_cvref_t<Ts>, BorrowedTensor> && ...))
    [[nodiscard]] std::span<const Tensor> run(Context& ctx, const Ts&... inputs) const {
        return run(ctx, std::source_location::current(), inputs...);
    }

    /// Variadic hot-path (borrowed) with explicit source location.
    template <typename... Ts>
        requires (sizeof...(Ts) >= 1 && (std::is_same_v<std::remove_cvref_t<Ts>, BorrowedTensor> && ...))
    [[nodiscard]] std::span<const Tensor> run(
        Context& ctx,
        std::source_location loc,
        const Ts&... inputs) const
    {
        ensure_compiled_("Runner::run", loc);

        constexpr std::size_t N = sizeof...(Ts);
        if (N != feeds_.size()) {
            throw Error::Wrapper(
                TF_INVALID_ARGUMENT,
                "Runner::run",
                detail::format("expected {} inputs, got {}", feeds_.size(), N),
                "",
                -1,
                loc);
        }

        ctx.input_vals_.clear();
        ctx.keepalives_.clear();

        ctx.input_vals_.reserve(feeds_.size());
        ctx.keepalives_.reserve(feeds_.size());

        std::size_t i = 0;
        auto push = [&](const BorrowedTensor& bt) {
            if (!bt.tensor) {
                throw Error::Wrapper(
                    TF_INVALID_ARGUMENT,
                    "Runner::run",
                    "null borrowed tensor handle",
                    feed_op_name_(i),
                    static_cast<int>(feeds_[i].index),
                    loc);
            }
            ctx.input_vals_.push_back(bt.tensor);
            if (bt.keepalive) {
                ctx.keepalives_.push_back(bt.keepalive);
            }
            ++i;
        };
        (push(inputs), ...);

        return run_raw_(ctx, loc);
    }

    // ─────────────────────────────────────────────────────────────────
    // Convenience APIs (ergonomic, safe, slightly more allocation)
    // ─────────────────────────────────────────────────────────────────

    /// Convenience: run and return all outputs by value.
    [[nodiscard]] std::vector<Tensor> run(
        std::span<const Tensor> inputs,
        std::source_location loc = std::source_location::current()) const
    {
        Context ctx = make_context();
        (void)run(ctx, inputs, loc);
        return std::move(ctx.outputs_);
    }

    /// Variadic convenience: run(t1, t2, ..., tN) -> vector of outputs.
    template <typename... Ts>
        requires (sizeof...(Ts) >= 1 && (std::is_same_v<std::remove_cvref_t<Ts>, Tensor> && ...))
    [[nodiscard]] std::vector<Tensor> run(const Ts&... inputs) const {
        return run(std::source_location::current(), inputs...);
    }

    /// Variadic convenience with explicit source location.
    template <typename... Ts>
        requires (sizeof...(Ts) >= 1 && (std::is_same_v<std::remove_cvref_t<Ts>, Tensor> && ...))
    [[nodiscard]] std::vector<Tensor> run(
        std::source_location loc,
        const Ts&... inputs) const
    {
        Context ctx = make_context();
        (void)run(ctx, loc, inputs...);
        return std::move(ctx.outputs_);
    }

    /// Variadic convenience for borrowed tensors.
    template <typename... Ts>
        requires (sizeof...(Ts) >= 1 && (std::is_same_v<std::remove_cvref_t<Ts>, BorrowedTensor> && ...))
    [[nodiscard]] std::vector<Tensor> run(const Ts&... inputs) const {
        return run(std::source_location::current(), inputs...);
    }

    /// Variadic convenience (borrowed) with explicit source location.
    template <typename... Ts>
        requires (sizeof...(Ts) >= 1 && (std::is_same_v<std::remove_cvref_t<Ts>, BorrowedTensor> && ...))
    [[nodiscard]] std::vector<Tensor> run(
        std::source_location loc,
        const Ts&... inputs) const
    {
        Context ctx = make_context();
        (void)run(ctx, loc, inputs...);
        return std::move(ctx.outputs_);
    }

    // ─────────────────────────────────────────────────────────────────
    // Structured tuple outputs ("wow easy" multi-fetch)
    // ─────────────────────────────────────────────────────────────────

    /// Convenience: run and return K outputs as a std::tuple, enabling structured bindings.
    ///
    /// Example:
    ///   auto [sum, diff] = run.run_tuple<2>(x, y);
    ///
    /// Requirements:
    ///   - The signature must have exactly K fetches.
    ///   - The call must provide exactly num_feeds() inputs.
    template <std::size_t K>
    [[nodiscard]] auto run_tuple(
        std::source_location loc = std::source_location::current()) const
    {
        static_assert(K >= 1, "run_tuple<K>: K must be >= 1");
        ensure_compiled_("Runner::run_tuple", loc);

        if (fetches_.size() != K) {
            throw Error::Wrapper(
                TF_INVALID_ARGUMENT,
                "Runner::run_tuple",
                detail::format("expected exactly {} fetches, got {}", K, fetches_.size()),
                "",
                -1,
                loc);
        }
        if (!feeds_.empty()) {
            throw Error::Wrapper(
                TF_INVALID_ARGUMENT,
                "Runner::run_tuple",
                detail::format("expected {} inputs, got 0", feeds_.size()),
                "",
                -1,
                loc);
        }

        std::array<TF_Tensor*, K> out_vals{};

        detail::session_run_checked(
            session_->handle(),
            run_options_,
            nullptr,
            nullptr,
            0,
            fetches_.data(),
            out_vals.data(),
            detail::checked_int(K, "Runner fetches"),
            targets_.empty() ? nullptr : targets_.data(),
            detail::checked_int(targets_.size(), "Runner targets"),
            run_metadata_,
            "Runner::run_tuple",
            loc);

        std::array<detail::RawTensorPtr, K> owned{};
        for (std::size_t i = 0; i < K; ++i) {
            owned[i].reset(out_vals[i]);
        }

        return make_tuple_from_owned_<K>(owned, std::make_index_sequence<K>{});
    }

    /// Variadic convenience: run_tuple<K>(t1, t2, ..., tN)
    template <std::size_t K, typename... Ts>
        requires (sizeof...(Ts) >= 1 && (std::is_same_v<std::remove_cvref_t<Ts>, Tensor> && ...))
    [[nodiscard]] auto run_tuple(const Ts&... inputs) const {
        return run_tuple<K>(std::source_location::current(), inputs...);
    }

    /// Variadic convenience with explicit source location.
    template <std::size_t K, typename... Ts>
        requires (sizeof...(Ts) >= 1 && (std::is_same_v<std::remove_cvref_t<Ts>, Tensor> && ...))
    [[nodiscard]] auto run_tuple(
        std::source_location loc,
        const Ts&... inputs) const
    {
        static_assert(K >= 1, "run_tuple<K>: K must be >= 1");
        ensure_compiled_("Runner::run_tuple", loc);

        constexpr std::size_t N = sizeof...(Ts);

        if (fetches_.size() != K) {
            throw Error::Wrapper(
                TF_INVALID_ARGUMENT,
                "Runner::run_tuple",
                detail::format("expected exactly {} fetches, got {}", K, fetches_.size()),
                "",
                -1,
                loc);
        }
        if (feeds_.size() != N) {
            throw Error::Wrapper(
                TF_INVALID_ARGUMENT,
                "Runner::run_tuple",
                detail::format("expected {} inputs, got {}", feeds_.size(), N),
                "",
                -1,
                loc);
        }

        std::array<TF_Tensor*, N> in_vals{};
        std::size_t i = 0;
        auto fill = [&](const Tensor& t) {
            if (!t.handle()) {
                throw Error::Wrapper(
                    TF_INVALID_ARGUMENT,
                    "Runner::run_tuple",
                    "null tensor handle",
                    feed_op_name_(i),
                    static_cast<int>(feeds_[i].index),
                    loc);
            }
            in_vals[i] = t.handle();
            ++i;
        };
        (fill(inputs), ...);

        std::array<TF_Tensor*, K> out_vals{};

        detail::session_run_checked(
            session_->handle(),
            run_options_,
            feeds_.data(),
            in_vals.data(),
            detail::checked_int(N, "Runner feeds"),
            fetches_.data(),
            out_vals.data(),
            detail::checked_int(K, "Runner fetches"),
            targets_.empty() ? nullptr : targets_.data(),
            detail::checked_int(targets_.size(), "Runner targets"),
            run_metadata_,
            "Runner::run_tuple",
            loc);

        std::array<detail::RawTensorPtr, K> owned{};
        for (std::size_t j = 0; j < K; ++j) {
            owned[j].reset(out_vals[j]);
        }

        return make_tuple_from_owned_<K>(owned, std::make_index_sequence<K>{});
    }

    /// Variadic convenience for borrowed tensors: run_tuple<K>(bt1, ..., btN)
    template <std::size_t K, typename... Ts>
        requires (sizeof...(Ts) >= 1 && (std::is_same_v<std::remove_cvref_t<Ts>, BorrowedTensor> && ...))
    [[nodiscard]] auto run_tuple(const Ts&... inputs) const {
        return run_tuple<K>(std::source_location::current(), inputs...);
    }

    /// Variadic convenience (borrowed) with explicit source location.
    template <std::size_t K, typename... Ts>
        requires (sizeof...(Ts) >= 1 && (std::is_same_v<std::remove_cvref_t<Ts>, BorrowedTensor> && ...))
    [[nodiscard]] auto run_tuple(
        std::source_location loc,
        const Ts&... inputs) const
    {
        static_assert(K >= 1, "run_tuple<K>: K must be >= 1");
        ensure_compiled_("Runner::run_tuple", loc);

        constexpr std::size_t N = sizeof...(Ts);

        if (fetches_.size() != K) {
            throw Error::Wrapper(
                TF_INVALID_ARGUMENT,
                "Runner::run_tuple",
                detail::format("expected exactly {} fetches, got {}", K, fetches_.size()),
                "",
                -1,
                loc);
        }
        if (feeds_.size() != N) {
            throw Error::Wrapper(
                TF_INVALID_ARGUMENT,
                "Runner::run_tuple",
                detail::format("expected {} inputs, got {}", feeds_.size(), N),
                "",
                -1,
                loc);
        }

        std::array<TF_Tensor*, N> in_vals{};
        std::array<std::shared_ptr<const void>, N> keepalives{};
        std::size_t i = 0;
        auto fill = [&](const BorrowedTensor& bt) {
            if (!bt.tensor) {
                throw Error::Wrapper(
                    TF_INVALID_ARGUMENT,
                    "Runner::run_tuple",
                    "null borrowed tensor handle",
                    feed_op_name_(i),
                    static_cast<int>(feeds_[i].index),
                    loc);
            }
            in_vals[i] = bt.tensor;
            keepalives[i] = bt.keepalive;
            ++i;
        };
        (fill(inputs), ...);

        std::array<TF_Tensor*, K> out_vals{};

        detail::session_run_checked(
            session_->handle(),
            run_options_,
            feeds_.data(),
            in_vals.data(),
            detail::checked_int(N, "Runner feeds"),
            fetches_.data(),
            out_vals.data(),
            detail::checked_int(K, "Runner fetches"),
            targets_.empty() ? nullptr : targets_.data(),
            detail::checked_int(targets_.size(), "Runner targets"),
            run_metadata_,
            "Runner::run_tuple",
            loc);

        std::array<detail::RawTensorPtr, K> owned{};
        for (std::size_t j = 0; j < K; ++j) {
            owned[j].reset(out_vals[j]);
        }

        return make_tuple_from_owned_<K>(owned, std::make_index_sequence<K>{});
    }

    /// Convenience: run and return a single output (requires exactly 1 fetch).
    [[nodiscard]] Tensor run_one(
        std::span<const Tensor> inputs,
        std::source_location loc = std::source_location::current()) const
    {
        if (fetches_.size() != 1) {
            throw Error::Wrapper(
                TF_INVALID_ARGUMENT,
                "Runner::run_one",
                detail::format("expected exactly 1 fetch, got {}", fetches_.size()),
                "",
                -1,
                loc);
        }

        // Fast path for the common case: 1 feed, 1 fetch.
        if (feeds_.size() == 1 && inputs.size() == 1) {
            return (*this)(inputs[0], loc);
        }

        Context ctx = make_context();
        (void)run(ctx, inputs, loc);
        return std::move(ctx.outputs_[0]);
    }

    /// Variadic convenience: run_one(t1, t2, ..., tN) -> single output.
    /// Requires exactly 1 fetch.
    template <typename... Ts>
        requires (sizeof...(Ts) >= 1 && (std::is_same_v<std::remove_cvref_t<Ts>, Tensor> && ...))
    [[nodiscard]] Tensor run_one(const Ts&... inputs) const {
        return run_one(std::source_location::current(), inputs...);
    }

    /// Variadic convenience with explicit source location.
    template <typename... Ts>
        requires (sizeof...(Ts) >= 1 && (std::is_same_v<std::remove_cvref_t<Ts>, Tensor> && ...))
    [[nodiscard]] Tensor run_one(
        std::source_location loc,
        const Ts&... inputs) const
    {
        if (fetches_.size() != 1) {
            throw Error::Wrapper(
                TF_INVALID_ARGUMENT,
                "Runner::run_one",
                detail::format("expected exactly 1 fetch, got {}", fetches_.size()),
                "",
                -1,
                loc);
        }
        return (*this)(loc, inputs...);
    }

    /// Variadic convenience for borrowed tensors: run_one(bt1, ..., btN).
    template <typename... Ts>
        requires (sizeof...(Ts) >= 1 && (std::is_same_v<std::remove_cvref_t<Ts>, BorrowedTensor> && ...))
    [[nodiscard]] Tensor run_one(const Ts&... inputs) const {
        return run_one(std::source_location::current(), inputs...);
    }

    /// Variadic convenience (borrowed) with explicit source location.
    template <typename... Ts>
        requires (sizeof...(Ts) >= 1 && (std::is_same_v<std::remove_cvref_t<Ts>, BorrowedTensor> && ...))
    [[nodiscard]] Tensor run_one(
        std::source_location loc,
        const Ts&... inputs) const
    {
        if (fetches_.size() != 1) {
            throw Error::Wrapper(
                TF_INVALID_ARGUMENT,
                "Runner::run_one",
                detail::format("expected exactly 1 fetch, got {}", fetches_.size()),
                "",
                -1,
                loc);
        }
        return (*this)(loc, inputs...);
    }

    /// "Wow this is easy" path: treat the compiled signature like a function.
    ///
    /// Requires exactly 1 feed and 1 fetch.
    [[nodiscard]] Tensor operator()(
        const Tensor& input,
        std::source_location loc = std::source_location::current()) const
    {
        ensure_compiled_("Runner::operator()", loc);

        if (feeds_.size() != 1 || fetches_.size() != 1) {
            throw Error::Wrapper(
                TF_INVALID_ARGUMENT,
                "Runner::operator()",
                detail::format(
                    "operator() requires exactly 1 feed and 1 fetch (got {} feeds, {} fetches)",
                    feeds_.size(), fetches_.size()),
                "",
                -1,
                loc);
        }

        if (!input.handle()) {
            throw Error::Wrapper(
                TF_INVALID_ARGUMENT,
                "Runner::operator()",
                "null tensor handle",
                feed_op_name_(0),
                feeds_[0].index,
                loc);
        }

        TF_Tensor* in_val = input.handle();
        TF_Tensor* out_val = nullptr;


        detail::session_run_checked(
            session_->handle(),
            run_options_,
            feeds_.data(),
            &in_val,
            1,
            fetches_.data(),
            &out_val,
            1,
            targets_.empty() ? nullptr : targets_.data(),
            detail::checked_int(targets_.size(), "Runner targets"),
            run_metadata_,
            "Runner::operator()",
            loc);

        return Tensor::FromRaw(out_val);
    }

    /// Variadic operator() for the common serving case where the signature has
    /// multiple feeds but exactly one fetch.
    ///
    /// Example:
    ///   auto run = model.runner().feed("ids:0").feed("mask:0").fetch("logits:0").compile();
    ///   Tensor logits = run(ids, mask);
    template <typename... Ts>
        requires (sizeof...(Ts) >= 1 && (std::is_same_v<std::remove_cvref_t<Ts>, Tensor> && ...))
    [[nodiscard]] Tensor operator()(const Ts&... inputs) const {
        return (*this)(std::source_location::current(), inputs...);
    }

    /// Variadic operator() with explicit source location.
    template <typename... Ts>
        requires (sizeof...(Ts) >= 1 && (std::is_same_v<std::remove_cvref_t<Ts>, Tensor> && ...))
    [[nodiscard]] Tensor operator()(
        std::source_location loc,
        const Ts&... inputs) const
    {
        ensure_compiled_("Runner::operator()", loc);

        constexpr std::size_t N = sizeof...(Ts);

        if (fetches_.size() != 1) {
            throw Error::Wrapper(
                TF_INVALID_ARGUMENT,
                "Runner::operator()",
                detail::format("operator() requires exactly 1 fetch (got {})", fetches_.size()),
                "",
                -1,
                loc);
        }
        if (feeds_.size() != N) {
            throw Error::Wrapper(
                TF_INVALID_ARGUMENT,
                "Runner::operator()",
                detail::format("expected {} inputs, got {}", feeds_.size(), N),
                "",
                -1,
                loc);
        }

        std::array<TF_Tensor*, N> in_vals{};
        std::size_t i = 0;
        auto fill = [&](const Tensor& t) {
            if (!t.handle()) {
                throw Error::Wrapper(
                    TF_INVALID_ARGUMENT,
                    "Runner::operator()",
                    "null tensor handle",
                    feed_op_name_(i),
                    static_cast<int>(feeds_[i].index),
                    loc);
            }
            in_vals[i] = t.handle();
            ++i;
        };
        (fill(inputs), ...);

        TF_Tensor* out_val = nullptr;

        detail::session_run_checked(
            session_->handle(),
            run_options_,
            feeds_.data(),
            in_vals.data(),
            detail::checked_int(N, "Runner feeds"),
            fetches_.data(),
            &out_val,
            1,
            targets_.empty() ? nullptr : targets_.data(),
            detail::checked_int(targets_.size(), "Runner targets"),
            run_metadata_,
            "Runner::operator()",
            loc);

        return Tensor::FromRaw(out_val);
    }

    /// Interop operator() for borrowed tensor input (1 feed/1 fetch).
    [[nodiscard]] Tensor operator()(
        const BorrowedTensor& input,
        std::source_location loc = std::source_location::current()) const
    {
        ensure_compiled_("Runner::operator()", loc);

        if (feeds_.size() != 1 || fetches_.size() != 1) {
            throw Error::Wrapper(
                TF_INVALID_ARGUMENT,
                "Runner::operator()",
                detail::format(
                    "operator() requires exactly 1 feed and 1 fetch (got {} feeds, {} fetches)",
                    feeds_.size(), fetches_.size()),
                "",
                -1,
                loc);
        }

        if (!input.tensor) {
            throw Error::Wrapper(
                TF_INVALID_ARGUMENT,
                "Runner::operator()",
                "null borrowed tensor handle",
                feed_op_name_(0),
                feeds_[0].index,
                loc);
        }

        TF_Tensor* in_val = input.tensor;
        TF_Tensor* out_val = nullptr;

        // Keep external lifetime token alive for duration of the call.
        const auto ka = input.keepalive;

        detail::session_run_checked(
            session_->handle(),
            run_options_,
            feeds_.data(),
            &in_val,
            1,
            fetches_.data(),
            &out_val,
            1,
            targets_.empty() ? nullptr : targets_.data(),
            detail::checked_int(targets_.size(), "Runner targets"),
            run_metadata_,
            "Runner::operator()",
            loc);

        return Tensor::FromRaw(out_val);
    }

    /// Variadic borrowed operator() for multiple feeds + single fetch.
    template <typename... Ts>
        requires (sizeof...(Ts) >= 1 && (std::is_same_v<std::remove_cvref_t<Ts>, BorrowedTensor> && ...))
    [[nodiscard]] Tensor operator()(const Ts&... inputs) const {
        return (*this)(std::source_location::current(), inputs...);
    }

    /// Variadic borrowed operator() with explicit source location.
    template <typename... Ts>
        requires (sizeof...(Ts) >= 1 && (std::is_same_v<std::remove_cvref_t<Ts>, BorrowedTensor> && ...))
    [[nodiscard]] Tensor operator()(
        std::source_location loc,
        const Ts&... inputs) const
    {
        ensure_compiled_("Runner::operator()", loc);

        constexpr std::size_t N = sizeof...(Ts);

        if (fetches_.size() != 1) {
            throw Error::Wrapper(
                TF_INVALID_ARGUMENT,
                "Runner::operator()",
                detail::format("operator() requires exactly 1 fetch (got {})", fetches_.size()),
                "",
                -1,
                loc);
        }
        if (feeds_.size() != N) {
            throw Error::Wrapper(
                TF_INVALID_ARGUMENT,
                "Runner::operator()",
                detail::format("expected {} inputs, got {}", feeds_.size(), N),
                "",
                -1,
                loc);
        }

        std::array<TF_Tensor*, N> in_vals{};
        std::array<std::shared_ptr<const void>, N> keepalives{};
        std::size_t i = 0;
        auto fill = [&](const BorrowedTensor& bt) {
            if (!bt.tensor) {
                throw Error::Wrapper(
                    TF_INVALID_ARGUMENT,
                    "Runner::operator()",
                    "null borrowed tensor handle",
                    feed_op_name_(i),
                    static_cast<int>(feeds_[i].index),
                    loc);
            }
            in_vals[i] = bt.tensor;
            keepalives[i] = bt.keepalive;
            ++i;
        };
        (fill(inputs), ...);

        TF_Tensor* out_val = nullptr;

        detail::session_run_checked(
            session_->handle(),
            run_options_,
            feeds_.data(),
            in_vals.data(),
            detail::checked_int(N, "Runner feeds"),
            fetches_.data(),
            &out_val,
            1,
            targets_.empty() ? nullptr : targets_.data(),
            detail::checked_int(targets_.size(), "Runner targets"),
            run_metadata_,
            "Runner::operator()",
            loc);

        return Tensor::FromRaw(out_val);
    }

private:
    friend class RunnerBuilder;

    std::shared_ptr<const Session> session_{};
    std::vector<TF_Output> feeds_{};
    std::vector<TF_Output> fetches_{};
    std::vector<TF_Operation*> targets_{};

    const TF_Buffer* run_options_{nullptr};
    TF_Buffer* run_metadata_{nullptr};

    explicit Runner(std::shared_ptr<const Session> session)
        : session_(std::move(session)) {}

    [[nodiscard]] const char* feed_op_name_(std::size_t i) const noexcept {
        if (i >= feeds_.size()) return "";
        const TF_Output out = feeds_[i];
        return out.oper ? TF_OperationName(out.oper) : "";
    }

    template <std::size_t K, std::size_t... Is>
    static auto make_tuple_from_owned_(
        std::array<detail::RawTensorPtr, K>& owned,
        std::index_sequence<Is...>)
    {
        return std::make_tuple(Tensor::FromRaw(owned[Is].release())...);
    }

    void ensure_compiled_(std::string_view where, std::source_location loc) const {
        if (!session_ || !session_->handle()) {
            throw Error::Wrapper(
                TF_FAILED_PRECONDITION,
                where,
                "runner is not compiled (no session)",
                "",
                -1,
                loc);
        }
    }

    [[nodiscard]] std::span<const Tensor> run_raw_(
        Context& ctx,
        std::source_location loc) const
    {
        // Ensure output_vals_ is the correct size and cleared to null.
        if (ctx.output_vals_.size() != fetches_.size()) {
            ctx.output_vals_.assign(fetches_.size(), nullptr);
        } else {
            std::fill(ctx.output_vals_.begin(), ctx.output_vals_.end(), nullptr);
        }

        const int num_inputs = detail::checked_int(feeds_.size(), "Runner feeds");
        const int num_outputs = detail::checked_int(fetches_.size(), "Runner fetches");
        const int num_targets = detail::checked_int(targets_.size(), "Runner targets");

        detail::session_run_checked(
            session_->handle(),
            run_options_,
            feeds_.data(),
            ctx.input_vals_.data(),
            num_inputs,
            fetches_.data(),
            ctx.output_vals_.data(),
            num_outputs,
            targets_.empty() ? nullptr : targets_.data(),
            num_targets,
            run_metadata_,
            "Runner::run",
            loc);

        // Inputs are no longer needed after TF_SessionRun returns.
        ctx.keepalives_.clear();

        // Take ownership of outputs for exception safety.
        ctx.owned_.clear();
        ctx.owned_.reserve(fetches_.size());
        for (TF_Tensor* t : ctx.output_vals_) {
            ctx.owned_.emplace_back(t);
        }

        ctx.outputs_.clear();
        ctx.outputs_.reserve(fetches_.size());
        for (auto& p : ctx.owned_) {
            ctx.outputs_.push_back(Tensor::FromRaw(p.release()));
        }

        return std::span<const Tensor>(ctx.outputs_);
    }
};

// ============================================================================
// RunnerBuilder (startup-time fluent builder)
// ============================================================================

class RunnerBuilder {
public:
    explicit RunnerBuilder(std::shared_ptr<const Session> session)
        : session_(std::move(session)) {}

    /// Provide serialized RunOptions (TensorFlow protobuf) to TF_SessionRun.
    RunnerBuilder& with_options(const Buffer& options) & {
        run_options_ = options.handle();
        return *this;
    }

    RunnerBuilder&& with_options(const Buffer& options) && {
        return std::move(with_options(options));
    }

    /// Provide a Buffer to receive RunMetadata from TF_SessionRun.
    RunnerBuilder& with_metadata(Buffer& metadata) & {
        run_metadata_ = metadata.handle();
        return *this;
    }

    RunnerBuilder&& with_metadata(Buffer& metadata) && {
        return std::move(with_metadata(metadata));
    }

    // ─────────────────────────────────────────────────────────────────
    // Feeds
    // ─────────────────────────────────────────────────────────────────

    RunnerBuilder& feed(TF_Output output,
                        std::source_location loc = std::source_location::current()) &
    {
        ensure_session_("RunnerBuilder::feed", loc);
        session_->validate_output(output, loc);
        feeds_.push_back(output);
        return *this;
    }

    RunnerBuilder&& feed(TF_Output output,
                         std::source_location loc = std::source_location::current()) &&
    {
        return std::move(feed(output, loc));
    }

    RunnerBuilder& feed(std::string_view name,
                        std::source_location loc = std::source_location::current()) &
    {
        ensure_session_("RunnerBuilder::feed", loc);
        feeds_.push_back(session_->resolve(name, loc));
        return *this;
    }

    RunnerBuilder&& feed(std::string_view name,
                         std::source_location loc = std::source_location::current()) &&
    {
        return std::move(feed(name, loc));
    }

    // ─────────────────────────────────────────────────────────────────
    // Fetches
    // ─────────────────────────────────────────────────────────────────

    RunnerBuilder& fetch(TF_Output output,
                         std::source_location loc = std::source_location::current()) &
    {
        ensure_session_("RunnerBuilder::fetch", loc);
        session_->validate_output(output, loc);
        fetches_.push_back(output);
        return *this;
    }

    RunnerBuilder&& fetch(TF_Output output,
                          std::source_location loc = std::source_location::current()) &&
    {
        return std::move(fetch(output, loc));
    }

    RunnerBuilder& fetch(std::string_view name,
                         std::source_location loc = std::source_location::current()) &
    {
        ensure_session_("RunnerBuilder::fetch", loc);
        fetches_.push_back(session_->resolve(name, loc));
        return *this;
    }

    RunnerBuilder&& fetch(std::string_view name,
                          std::source_location loc = std::source_location::current()) &&
    {
        return std::move(fetch(name, loc));
    }

    // ─────────────────────────────────────────────────────────────────
    // Targets
    // ─────────────────────────────────────────────────────────────────

    RunnerBuilder& target(TF_Operation* op,
                          std::source_location loc = std::source_location::current()) &
    {
        ensure_session_("RunnerBuilder::target", loc);
        session_->validate_operation(op, loc);
        targets_.push_back(op);
        return *this;
    }

    RunnerBuilder&& target(TF_Operation* op,
                           std::source_location loc = std::source_location::current()) &&
    {
        return std::move(target(op, loc));
    }

    /// Target by name (accepts either "OpName" or "OpName:0"; index is ignored).
    RunnerBuilder& target(std::string_view op_name,
                          std::source_location loc = std::source_location::current()) &
    {
        ensure_session_("RunnerBuilder::target", loc);
        const TF_Output out = session_->resolve(op_name, loc);
        session_->validate_operation(out.oper, loc);
        targets_.push_back(out.oper);
        return *this;
    }

    RunnerBuilder&& target(std::string_view op_name,
                           std::source_location loc = std::source_location::current()) &&
    {
        return std::move(target(op_name, loc));
    }

    // ─────────────────────────────────────────────────────────────────
    // Compile
    // ─────────────────────────────────────────────────────────────────

    /// Compile the signature (resolve once; store handles).
    [[nodiscard]] Runner compile(
        std::source_location loc = std::source_location::current()) const
    {
        ensure_session_("RunnerBuilder::compile", loc);

        Runner r(session_);
        r.feeds_ = feeds_;
        r.fetches_ = fetches_;
        r.targets_ = targets_;
        r.run_options_ = run_options_;
        r.run_metadata_ = run_metadata_;

        // Fail fast at startup: validate all handles belong to this session.
        for (const TF_Output& out : r.feeds_) {
            r.session_->validate_output(out, loc);
        }
        for (const TF_Output& out : r.fetches_) {
            r.session_->validate_output(out, loc);
        }
        for (TF_Operation* op : r.targets_) {
            r.session_->validate_operation(op, loc);
        }

        return r;
    }

private:
    std::shared_ptr<const Session> session_{};

    std::vector<TF_Output> feeds_{};
    std::vector<TF_Output> fetches_{};
    std::vector<TF_Operation*> targets_{};

    const TF_Buffer* run_options_{nullptr};
    TF_Buffer* run_metadata_{nullptr};

    void ensure_session_(std::string_view where, std::source_location loc) const {
        if (!session_ || !session_->handle()) {
            throw Error::Wrapper(
                TF_FAILED_PRECONDITION,
                where,
                "no session",
                "",
                -1,
                loc);
        }
    }
};

// ============================================================================
// Model - High-level facade for SavedModel with production features
// ============================================================================

class Model {
public:
    Model() = default;

    // Moveable
    Model(Model&&) = default;
    Model& operator=(Model&&) = default;

    // Non-copyable
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    /// Load a SavedModel from disk
    [[nodiscard]] static Model Load(
        const std::string& export_dir,
        const std::vector<std::string>& tags = {"serve"})
    {
        Model m;
        auto [session, graph] = Session::LoadSavedModel(export_dir, tags);
        m.session_ = std::make_shared<Session>(std::move(session));
        m.graph_ = std::make_shared<Graph>(std::move(graph));
        return m;
    }

    // ─────────────────────────────────────────────────────────────────
    // Endpoint resolution (call once at startup, cache the result)
    // ─────────────────────────────────────────────────────────────────

    /// Resolve an operation name to TF_Output
    [[nodiscard]] TF_Output resolve(
        std::string_view name,
        std::source_location loc = std::source_location::current()) const
    {
        ensure_loaded_("resolve", loc);
        return session_->resolve(name, loc);
    }

    /// Resolve input and output endpoints together (convenience)
    struct Endpoints {
        TF_Output input;
        TF_Output output;
    };

    [[nodiscard]] Endpoints resolve(
        std::string_view input_name,
        std::string_view output_name,
        std::source_location loc = std::source_location::current()) const
    {
        ensure_loaded_("resolve", loc);
        return {
            session_->resolve(input_name, loc),
            session_->resolve(output_name, loc)
        };
    }

    // ─────────────────────────────────────────────────────────────────
    // "Wow this is easy" compilation helpers
    // ─────────────────────────────────────────────────────────────────

    /// Compile a 1-in / 1-out inference signature by name.
    [[nodiscard]] Runner compile(
        std::string_view input_name,
        std::string_view output_name,
        std::source_location loc = std::source_location::current()) const
    {
        ensure_loaded_("compile", loc);
        return runner()
            .feed(input_name, loc)
            .fetch(output_name, loc)
            .compile(loc);
    }

    /// Compile a 1-in / 1-out inference signature by handle.
    [[nodiscard]] Runner compile(
        TF_Output input,
        TF_Output output,
        std::source_location loc = std::source_location::current()) const
    {
        ensure_loaded_("compile", loc);
        return runner()
            .feed(input, loc)
            .fetch(output, loc)
            .compile(loc);
    }

    // ─────────────────────────────────────────────────────────────────
    // Production features
    // ─────────────────────────────────────────────────────────────────

    /// Warmup: run inference once to trigger TF internal allocations/JIT.
    /// Call during startup, not on first request.
    void warmup(
        TF_Output input,
        const Tensor& dummy_input,
        TF_Output output,
        std::source_location loc = std::source_location::current()) const
    {
        ensure_loaded_("warmup", loc);
        auto run = compile(input, output, loc);
        (void)run(dummy_input, loc);
    }

    /// Warmup with multiple inputs/outputs
    void warmup(
        std::span<const Feed> feeds,
        std::span<const Fetch> fetches,
        std::source_location loc = std::source_location::current()) const
    {
        ensure_loaded_("warmup", loc);
        (void)session_->Run(feeds, fetches, {}, nullptr, nullptr, loc);
    }

    /// Validate that a tensor matches expected input dtype.
    /// Returns empty string if valid, error description otherwise.
    [[nodiscard]] std::string validate_input(
        TF_Output input,
        const Tensor& tensor) const noexcept
    {
        if (!session_) {
            return "model not loaded";
        }
        if (!tensor.handle()) {
            return "tensor is null";
        }

        TF_DataType expected_dtype = TF_OperationOutputType(input);
        if (tensor.dtype() != expected_dtype) {
            return detail::format(
                "dtype mismatch: expected {}, got {}",
                dtype_name(expected_dtype), dtype_name(tensor.dtype()));
        }

        return {};
    }

    /// Validate and throw if invalid
    void require_valid_input(
        TF_Output input,
        const Tensor& tensor,
        std::source_location loc = std::source_location::current()) const
    {
        auto error = validate_input(input, tensor);
        if (!error.empty()) {
            const char* op_name = input.oper ? TF_OperationName(input.oper) : "";
            throw Error::Wrapper(
                TF_INVALID_ARGUMENT,
                "Model::require_valid_input",
                error,
                op_name ? op_name : "",
                input.index,
                loc);
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // Runner access
    // ─────────────────────────────────────────────────────────────────

    /// Get a builder for compiling production inference signatures.
    [[nodiscard]] RunnerBuilder runner() const {
        ensure_loaded_("runner", std::source_location::current());
        return RunnerBuilder(session_);
    }

    // ─────────────────────────────────────────────────────────────────
    // Batch operations
    // ─────────────────────────────────────────────────────────────────

    /// Run same input->output mapping for multiple inputs
    [[nodiscard]] std::vector<Tensor> BatchRun(
        TF_Output input,
        std::span<const Tensor> inputs,
        TF_Output output,
        std::source_location loc = std::source_location::current()) const
    {
        ensure_loaded_("BatchRun", loc);
        return session_->BatchRun(input, inputs, output, loc);
    }

    [[nodiscard]] std::vector<Tensor> BatchRun(
        TF_Output input,
        const std::vector<Tensor>& inputs,
        TF_Output output,
        std::source_location loc = std::source_location::current()) const
    {
        return BatchRun(input, std::span<const Tensor>(inputs), output, loc);
    }

    /// True batching: stack inputs, single TF call, split outputs
    [[nodiscard]] std::vector<Tensor> BatchRunStacked(
        TF_Output input,
        std::span<const Tensor> inputs,
        TF_Output output,
        std::source_location loc = std::source_location::current()) const
    {
        ensure_loaded_("BatchRunStacked", loc);
        return session_->BatchRunStacked(input, inputs, output, loc);
    }

    [[nodiscard]] std::vector<Tensor> BatchRunStacked(
        TF_Output input,
        const std::vector<Tensor>& inputs,
        TF_Output output,
        std::source_location loc = std::source_location::current()) const
    {
        return BatchRunStacked(input, std::span<const Tensor>(inputs), output, loc);
    }

    // ─────────────────────────────────────────────────────────────────
    // Accessors
    // ─────────────────────────────────────────────────────────────────

    [[nodiscard]] const Session& session() const {
        ensure_loaded_("session", std::source_location::current());
        return *session_;
    }

    /// Access underlying graph (read-only)
    [[nodiscard]] const Graph& graph() const {
        ensure_loaded_("graph", std::source_location::current());
        return *graph_;
    }

    [[nodiscard]] bool loaded() const noexcept { return session_ && graph_; }

private:
    std::shared_ptr<Session> session_{};
    std::shared_ptr<Graph> graph_{};

    void ensure_loaded_(
        const char* method,
        std::source_location loc = std::source_location::current()) const
    {
        if (!session_ || !graph_) {
            throw Error::Wrapper(
                TF_FAILED_PRECONDITION,
                detail::format("Model::{}", method),
                "model not loaded",
                "",
                -1,
                loc);
        }
    }
};

} // namespace facade

// Bring common items into tf_wrap namespace for convenience
using facade::Runner;
using facade::RunnerBuilder;
using facade::Model;

} // namespace tf_wrap
