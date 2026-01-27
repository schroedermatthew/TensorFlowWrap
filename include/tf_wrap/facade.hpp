// tf_wrap/facade.hpp
// Ergonomic layer for TensorFlowWrap - Production Inference Edition
//
// Provides:
// - Runner: Fluent API for session execution (handle-based only)
// - Model: High-level facade for SavedModel with production features

#pragma once

#include <memory>
#include <string>
#include <string_view>
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
// Runner - Fluent API for session execution (handle-based only)
// ============================================================================

class Runner {
public:
    explicit Runner(const Session& session)
        : session_(&session)
        , graph_(session.graph_handle()) {}

    /// Provide serialized RunOptions (TensorFlow protobuf) to TF_SessionRun.
    Runner& with_options(const Buffer& options) & {
        run_options_ = &options;
        return *this;
    }

    Runner&& with_options(const Buffer& options) && {
        return std::move(with_options(options));
    }

    /// Provide a Buffer to receive RunMetadata from TF_SessionRun.
    Runner& with_metadata(Buffer& metadata) & {
        run_metadata_ = &metadata;
        return *this;
    }

    Runner&& with_metadata(Buffer& metadata) && {
        return std::move(with_metadata(metadata));
    }
    
    /// Add a feed (input tensor) - TF_Output version
    Runner& feed(TF_Output output, const Tensor& tensor) & {
        if (!tensor.handle()) {
            throw Error::Wrapper(TF_INVALID_ARGUMENT, "Runner::feed",
                "null tensor handle", "", output.index);
        }
        feeds_.push_back(FeedEntry{output, tensor.handle(), tensor.keepalive()});
        return *this;
    }
    
    Runner&& feed(TF_Output output, const Tensor& tensor) && {
        return std::move(feed(output, tensor));
    }
    
    /// Add a feed from raw TF_Tensor*
    Runner& feed(TF_Output output, TF_Tensor* tensor) & {
        if (!tensor) {
            throw Error::Wrapper(TF_INVALID_ARGUMENT, "Runner::feed",
                "null TF_Tensor*", "", output.index);
        }
        feeds_.push_back(FeedEntry{output, tensor, {}});
        return *this;
    }
    
    Runner&& feed(TF_Output output, TF_Tensor* tensor) && {
        return std::move(feed(output, tensor));
    }
    
    /// Add a fetch (output to retrieve)
    Runner& fetch(TF_Output output) & {
        fetches_.push_back(output);
        return *this;
    }
    
    Runner&& fetch(TF_Output output) && {
        return std::move(fetch(output));
    }
    
    /// Add a target operation (run but don't fetch)
    Runner& target(TF_Operation* op) & {
        targets_.push_back(op);
        return *this;
    }
    
    Runner&& target(TF_Operation* op) && {
        return std::move(target(op));
    }
    
    /// Execute and return all fetched outputs
    [[nodiscard]] std::vector<Tensor> run() const {
        if (!session_) {
            throw Error::Wrapper(TF_FAILED_PRECONDITION, "Runner::run",
                "no session", "", -1);
        }
        
        std::vector<TF_Output> input_ops;
        std::vector<TF_Tensor*> input_vals;
        input_ops.reserve(feeds_.size());
        input_vals.reserve(feeds_.size());
        
        for (const auto& f : feeds_) {
            input_ops.push_back(f.output);
            input_vals.push_back(f.tensor);
        }
        
        std::vector<TF_Tensor*> output_vals(fetches_.size(), nullptr);
        
        Status st;
        TF_SessionRun(
            session_->handle(),
            run_options_ ? run_options_->handle() : nullptr,
            input_ops.data(), input_vals.data(), 
            detail::checked_int(feeds_.size(), "Runner::run feeds"),
            fetches_.data(), output_vals.data(), 
            detail::checked_int(fetches_.size(), "Runner::run fetches"),
            targets_.data(), 
            detail::checked_int(targets_.size(), "Runner::run targets"),
            run_metadata_ ? run_metadata_->handle() : nullptr,
            st.get());
            
        // Take ownership of output tensors for exception safety
        using RawTensorPtr = detail::RawTensorPtr;
        std::vector<RawTensorPtr> owned;
        owned.reserve(output_vals.size());
        for (auto* t : output_vals) {
            owned.emplace_back(t);
        }
        
        st.throw_if_error("Runner::run");

        for (std::size_t i = 0; i < owned.size(); ++i) {
            if (!owned[i]) {
                const TF_Output out = fetches_[i];
                const char* op_name = out.oper ? TF_OperationName(out.oper) : "";
                throw Error::Wrapper(TF_INTERNAL, "Runner::run",
                    "fetch returned null tensor",
                    op_name ? op_name : "", out.index);
            }
        }

        std::vector<Tensor> results;
        results.reserve(owned.size());
        for (auto& p : owned) {
            results.push_back(Tensor::FromRaw(p.release()));
        }
        
        return results;
    }
    
    /// Execute and return single output (convenience for single fetch)
    [[nodiscard]] Tensor run_one() const {
        if (fetches_.size() != 1) {
            throw Error::Wrapper(TF_INVALID_ARGUMENT, "Runner::run_one",
                detail::format("expected exactly 1 fetch, got {}", fetches_.size()),
                "", -1);
        }
        return std::move(run()[0]);
    }
    
    /// Clear all feeds, fetches, and targets
    void clear() {
        feeds_.clear();
        fetches_.clear();
        targets_.clear();
        run_options_ = nullptr;
        run_metadata_ = nullptr;
    }

private:
    const Session* session_;
    TF_Graph* graph_;

    const Buffer* run_options_{nullptr};
    Buffer* run_metadata_{nullptr};
    
    struct FeedEntry {
        TF_Output output{nullptr, 0};
        TF_Tensor* tensor{nullptr};
        std::shared_ptr<const void> keepalive{};
    };

    std::vector<FeedEntry> feeds_;
    std::vector<TF_Output> fetches_;
    std::vector<TF_Operation*> targets_;
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
        m.session_ = std::make_unique<Session>(std::move(session));
        m.graph_ = std::make_unique<Graph>(std::move(graph));
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
    // Production features
    // ─────────────────────────────────────────────────────────────────
    
    /// Warmup: run inference once to trigger JIT compilation and memory allocation.
    /// Call during startup, not on first request.
    void warmup(
        TF_Output input, 
        const Tensor& dummy_input,
        TF_Output output,
        std::source_location loc = std::source_location::current()) const
    {
        ensure_loaded_("warmup", loc);
        // Run inference, discard result
        (void)runner().feed(input, dummy_input).fetch(output).run();
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
            throw Error::Wrapper(TF_INVALID_ARGUMENT, "Model::require_valid_input",
                error, op_name ? op_name : "", input.index, loc);
        }
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Runner access
    // ─────────────────────────────────────────────────────────────────
    
    /// Get a runner for this model
    [[nodiscard]] Runner runner() const {
        if (!session_) {
            throw Error::Wrapper(TF_FAILED_PRECONDITION, "Model::runner",
                "model not loaded", "", -1);
        }
        return Runner(*session_);
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
    
    // ─────────────────────────────────────────────────────────────────
    // State access
    // ─────────────────────────────────────────────────────────────────
    
    /// Check if model is loaded
    [[nodiscard]] bool valid() const noexcept {
        return session_ != nullptr && graph_ != nullptr;
    }
    
    [[nodiscard]] explicit operator bool() const noexcept {
        return valid();
    }
    
    /// Access underlying session
    [[nodiscard]] const Session& session() const {
        if (!session_) {
            throw Error::Wrapper(TF_FAILED_PRECONDITION, "Model::session",
                "model not loaded", "", -1);
        }
        return *session_;
    }
    
    /// Access underlying graph
    [[nodiscard]] const Graph& graph() const {
        if (!graph_) {
            throw Error::Wrapper(TF_FAILED_PRECONDITION, "Model::graph",
                "model not loaded", "", -1);
        }
        return *graph_;
    }

private:
    std::unique_ptr<Session> session_;
    std::unique_ptr<Graph> graph_;
    
    void ensure_loaded_(
        const char* method,
        std::source_location loc = std::source_location::current()) const
    {
        if (!session_ || !graph_) {
            throw Error::Wrapper(TF_FAILED_PRECONDITION, 
                detail::format("Model::{}", method),
                "model not loaded", "", -1, loc);
        }
    }
};

} // namespace facade

// Bring common items into tf_wrap namespace for convenience
using facade::Runner;
using facade::Model;

} // namespace tf_wrap
