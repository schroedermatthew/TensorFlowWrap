// tf_wrap/facade.hpp
// Ergonomic layer for TensorFlowWrap
//
// Provides:
// - TensorName: Parse "op:index" strings
// - Endpoint: Unified way to refer to tensor outputs
// - Runner: Fluent API for session execution
// - Model: High-level facade for SavedModel
//
// Graph-building helpers (dtype-inferred ops) live in tf_wrap/facade_ops.hpp.

#pragma once

#include <algorithm>
#include <cctype>
#include <charconv>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "tf_wrap/format.hpp"
#include "tf_wrap/graph.hpp"
#include "tf_wrap/session.hpp"
#include "tf_wrap/tensor.hpp"
#include "tf_wrap/detail/raw_tensor_ptr.hpp"
#include "tf_wrap/tensor_name.hpp"

namespace tf_wrap {
namespace facade {

// ============================================================================
// TensorName - Parse "op:index" strings
// ============================================================================

struct TensorName {
    std::string op;
    int index{0};
    bool had_explicit_index{false};

    /// Parse a tensor name string like "op_name:0" or "op_name"
    /// Throws std::invalid_argument on parse errors
    [[nodiscard]] static TensorName parse(std::string_view s) {
        auto parsed = tf_wrap::detail::parse_tensor_name(s);
        TensorName result;
        result.op = std::move(parsed.op);
        result.index = parsed.index;
        result.had_explicit_index = parsed.had_explicit_index;
        return result;
    }
    
    /// Check if a string looks like a tensor name
    [[nodiscard]] static bool looks_like_tensor_name(std::string_view s) noexcept {
        if (s.empty()) return false;
        
        // Trim
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
            s.remove_prefix(1);
        }
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
            s.remove_suffix(1);
        }
        
        if (s.empty()) return false;
        
        // Contains only valid TF name characters: alphanumeric, underscore, slash, colon
        return std::all_of(s.begin(), s.end(), [](unsigned char c) {
            return std::isalnum(c) || c == '_' || c == '/' || c == ':' || c == '-' || c == '.';
        });
    }
    
    /// Convert back to string representation
    [[nodiscard]] std::string to_string() const {
        if (had_explicit_index || index != 0) {
            return op + ":" + std::to_string(index);
        }
        return op;
    }
    
    /// Convert to TF_Output given a graph
    [[nodiscard]] TF_Output to_output(TF_Graph* graph) const {
        TF_Operation* operation = TF_GraphOperationByName(graph, op.c_str());
        if (!operation) {
            throw std::runtime_error(tf_wrap::detail::format(
                "TensorName::to_output: operation '{}' not found in graph", op));
        }
        
        // Bounds check
        const int num_outputs = TF_OperationNumOutputs(operation);
        if (index < 0 || index >= num_outputs) {
            throw std::out_of_range(tf_wrap::detail::format(
                "TensorName::to_output: output index {} out of range for operation '{}' "
                "(has {} outputs, valid indices are 0-{})",
                index, op, num_outputs, num_outputs > 0 ? num_outputs - 1 : 0));
        }
        
        return TF_Output{operation, index};
    }
};

// ============================================================================
// Endpoint - Unified tensor output reference
// ============================================================================

class Endpoint {
public:
    /// Construct from resolved TF_Output
    Endpoint(TF_Output output) : data_(output) {}
    
    /// Construct from string (will be parsed as TensorName)
    Endpoint(const std::string& name) : data_(TensorName::parse(name)) {}
    
    /// Construct from string_view (will be parsed as TensorName)
    Endpoint(std::string_view name) : data_(TensorName::parse(name)) {}
    
    /// Construct from C string
    Endpoint(const char* name) : Endpoint(std::string_view(name)) {}
    
    /// Check if already resolved to TF_Output
    [[nodiscard]] bool is_resolved() const noexcept {
        return std::holds_alternative<TF_Output>(data_);
    }
    
    /// Get as TF_Output (throws if unresolved)
    [[nodiscard]] TF_Output as_output() const {
        if (auto* p = std::get_if<TF_Output>(&data_)) {
            return *p;
        }
        throw std::logic_error("Endpoint::as_output: endpoint is not resolved");
    }
    
    /// Get as TensorName (throws if already resolved)
    [[nodiscard]] const TensorName& as_name() const {
        if (auto* p = std::get_if<TensorName>(&data_)) {
            return *p;
        }
        throw std::logic_error("Endpoint::as_name: endpoint is already resolved");
    }
    
    /// Resolve to TF_Output given a graph
    [[nodiscard]] TF_Output resolve(TF_Graph* graph) const {
        if (auto* p = std::get_if<TF_Output>(&data_)) {
            return *p;
        }
        return std::get<TensorName>(data_).to_output(graph);
    }

private:
    std::variant<TF_Output, TensorName> data_;
};

// ============================================================================
// Runner - Fluent API for session execution
// ============================================================================

class Runner {
public:
    explicit Runner(const Session& session)
        : session_(&session)
        , graph_(session.graph_handle()) {}

    /// Provide serialized RunOptions (TensorFlow protobuf) to TF_SessionRun.
    /// Lifetime: options must outlive run().
    Runner& with_options(const Buffer& options) & {
        run_options_ = &options;
        return *this;
    }

    Runner&& with_options(const Buffer& options) && {
        return std::move(with_options(options));
    }

    /// Provide a Buffer to receive RunMetadata from TF_SessionRun.
    /// Lifetime: metadata must outlive run(). Contents are overwritten by TF.
    Runner& with_metadata(Buffer& metadata) & {
        run_metadata_ = &metadata;
        return *this;
    }

    Runner&& with_metadata(Buffer& metadata) && {
        return std::move(with_metadata(metadata));
    }
    
    /// Add a feed (input tensor)
    Runner& feed(Endpoint endpoint, const Tensor& tensor) & {
        TF_Output output = resolve(endpoint);
        if (!tensor.handle()) {
            throw std::invalid_argument("Runner::feed: null tensor handle");
        }
        feeds_.push_back(FeedEntry{output, tensor.handle(), tensor.keepalive()});
        return *this;
    }
    
    Runner&& feed(Endpoint endpoint, const Tensor& tensor) && {
        return std::move(feed(endpoint, tensor));
    }
    
    /// Add a feed from raw TF_Tensor*
    Runner& feed(Endpoint endpoint, TF_Tensor* tensor) & {
        if (!tensor) {
            throw std::invalid_argument("Runner::feed: null TF_Tensor*");
        }
        TF_Output output = resolve(endpoint);
        feeds_.push_back(FeedEntry{output, tensor, {}});
        return *this;
    }
    
    Runner&& feed(Endpoint endpoint, TF_Tensor* tensor) && {
        return std::move(feed(endpoint, tensor));
    }
    
    /// Add a fetch (output to retrieve)
    Runner& fetch(Endpoint endpoint) & {
        TF_Output output = resolve(endpoint);
        fetches_.push_back(output);
        return *this;
    }
    
    Runner&& fetch(Endpoint endpoint) && {
        return std::move(fetch(endpoint));
    }
    
    /// Add a target operation (run but don't fetch)
    Runner& target(Endpoint endpoint) & {
        TF_Output output = resolve(endpoint);
        targets_.push_back(output.oper);
        return *this;
    }
    
    Runner&& target(Endpoint endpoint) && {
        return std::move(target(endpoint));
    }
    
    /// Execute and return all fetched outputs
    [[nodiscard]] std::vector<Tensor> run() const {
        if (!session_) {
            throw std::runtime_error("Runner::run: no session");
        }
        
        // Build arrays for TF_SessionRun
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
            run_options_ ? run_options_->handle() : nullptr,  // run_options
            input_ops.data(), input_vals.data(), detail::checked_int(feeds_.size(), "Runner::run feeds"),
            fetches_.data(), output_vals.data(), detail::checked_int(fetches_.size(), "Runner::run fetches"),
            targets_.data(), detail::checked_int(targets_.size(), "Runner::run targets"),
            run_metadata_ ? run_metadata_->handle() : nullptr,  // run_metadata
            st.handle());
        // Take ownership of output tensors for exception safety
        using RawTensorPtr = tf_wrap::detail::RawTensorPtr;
        std::vector<RawTensorPtr> owned;
        owned.reserve(output_vals.size());
        for (auto* t : output_vals) {
            owned.emplace_back(t);
        }
        
        st.throw_if_error("Runner::run");

        for (std::size_t i = 0; i < owned.size(); ++i) {
            if (!owned[i]) {
                const TF_Output out = fetches_[i];
                const char* op_name = out.oper ? TF_OperationName(out.oper) : nullptr;
                throw tf_wrap::Error::Wrapper(
                    TF_INTERNAL,
                    "Runner::run",
                    "fetch returned null tensor",
                    op_name ? op_name : "",
                    out.index);
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
            throw std::runtime_error(tf_wrap::detail::format(
                "Runner::run_one: expected exactly 1 fetch, got {}", fetches_.size()));
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
    
    // Per-runner resolution cache (Runner is single-use, not thread-safe)
    struct CacheKey {
        std::string op;
        int index{0};

        friend bool operator==(const CacheKey& a, const CacheKey& b) noexcept {
            return a.index == b.index && a.op == b.op;
        }
    };

    struct CacheKeyHash {
        std::size_t operator()(const CacheKey& k) const noexcept {
            std::size_t h1 = std::hash<std::string>{}(k.op);
            std::size_t h2 = std::hash<int>{}(k.index);
            // Hash combine (boost-style)
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
        }
    };

    mutable std::unordered_map<CacheKey, TF_Output, CacheKeyHash> cache_;
    
    [[nodiscard]] static TF_Output validate_output_(TF_Output out, const char* context) {
        if (!out.oper) {
            throw std::invalid_argument(tf_wrap::detail::format(
                "{}: null TF_Operation*", context));
        }
        const int n = TF_OperationNumOutputs(out.oper);
        if (out.index < 0 || out.index >= n) {
            const char* name = TF_OperationName(out.oper);
            throw std::out_of_range(tf_wrap::detail::format(
                "{}: output index {} out of range for operation '{}' (has {} outputs)",
                context, out.index, name ? name : "", n));
        }
        return out;
    }

    TF_Output resolve(const Endpoint& endpoint) const {
        if (endpoint.is_resolved()) {
            return validate_output_(endpoint.as_output(), "Runner::resolve");
        }

        const TensorName& name = endpoint.as_name();
        CacheKey key{name.op, name.index};

        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;
        }

        TF_Output output = name.to_output(graph_);
        cache_.emplace(std::move(key), output);
        return output;
    }

};

// ============================================================================
// Model - High-level facade for SavedModel
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
        try {
            auto [session, graph] = Session::LoadSavedModel(export_dir, tags);
            m.session_ = std::make_unique<Session>(std::move(session));
            m.graph_ = std::make_unique<Graph>(std::move(graph));
            return m;
        } catch (const tf_wrap::Error&) {
            throw;
        } catch (const std::exception& e) {
            throw std::runtime_error(tf_wrap::detail::format(
                "Model::Load: failed to load SavedModel from '{}': {} (check the directory exists and contains saved_model.pb)",
                export_dir, e.what()));
        }
    }
    
    /// Get a runner for this model
    [[nodiscard]] Runner runner() const {
        if (!session_) {
            throw std::runtime_error("Model::runner: model not loaded");
        }
        return Runner(*session_);
    }
    
    /// Convenience: run with single input/output
    [[nodiscard]] Tensor operator()(
        const std::string& input_name,
        const Tensor& input,
        const std::string& output_name) const
    {
        return runner()
            .feed(Endpoint(input_name), input)
            .fetch(Endpoint(output_name))
            .run_one();
    }


    /// Convenience: batch run with single input/output.
    [[nodiscard]] std::vector<Tensor> BatchRun(
        const std::string& input_name,
        const std::vector<Tensor>& inputs,
        const std::string& output_name) const
    {
        return session().BatchRun(input_name, inputs, output_name);
    }

    /// Convenience: batch run with single input/output (span overload).
    [[nodiscard]] std::vector<Tensor> BatchRun(
        const std::string& input_name,
        std::span<const Tensor> inputs,
        const std::string& output_name) const
    {
        return session().BatchRun(input_name, inputs, output_name);
    }
    
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
            throw std::runtime_error("Model::session: model not loaded");
        }
        return *session_;
    }
    
    /// Access underlying graph
    [[nodiscard]] const Graph& graph() const {
        if (!graph_) {
            throw std::runtime_error("Model::graph: model not loaded");
        }
        return *graph_;
    }

private:
    std::unique_ptr<Session> session_;
    std::unique_ptr<Graph> graph_;
};

} // namespace facade

// Bring common items into tf_wrap namespace for convenience
using facade::TensorName;
using facade::Endpoint;
using facade::Runner;
using facade::Model;

} // namespace tf_wrap
