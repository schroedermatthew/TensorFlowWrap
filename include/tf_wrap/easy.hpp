// tf_wrap/easy.hpp
// Ergonomic layer for TensorFlowWrap
//
// Provides:
// - TensorName: Parse "op:index" strings
// - Endpoint: Unified way to refer to tensor outputs
// - Runner: Fluent API for session execution
// - Model: High-level facade for SavedModel
// - Easy helpers: Scalar, Const, Placeholder, Identity

#pragma once

#include <algorithm>
#include <cctype>
#include <charconv>
#include <memory>
#include <mutex>
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
#include "tf_wrap/ops.hpp"
#include "tf_wrap/session.hpp"
#include "tf_wrap/tensor.hpp"

namespace tf_wrap {
namespace easy {

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
        TensorName result;
        
        // Trim whitespace
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
            s.remove_prefix(1);
        }
        while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
            s.remove_suffix(1);
        }
        
        if (s.empty()) {
            throw std::invalid_argument("TensorName::parse: empty string");
        }
        
        // Find the last colon
        const auto colon_pos = s.rfind(':');
        
        if (colon_pos == std::string_view::npos) {
            // No colon - just op name
            result.op = std::string(s);
            result.index = 0;
            result.had_explicit_index = false;
        } else if (colon_pos == 0) {
            // Colon at start - invalid
            throw std::invalid_argument("TensorName::parse: empty operation name");
        } else if (colon_pos == s.size() - 1) {
            // Colon at end with nothing after
            throw std::invalid_argument("TensorName::parse: missing index after colon");
        } else {
            // Has colon - check if everything after is a valid non-negative integer
            std::string_view index_part = s.substr(colon_pos + 1);
            
            // Check all digits
            bool all_digits = !index_part.empty() && 
                std::all_of(index_part.begin(), index_part.end(), 
                    [](unsigned char c) { return std::isdigit(c); });
            
            if (all_digits) {
                result.op = std::string(s.substr(0, colon_pos));
                
                // Parse the index
                int idx = 0;
                auto [ptr, ec] = std::from_chars(
                    index_part.data(), 
                    index_part.data() + index_part.size(), 
                    idx);
                
                if (ec != std::errc{} || ptr != index_part.data() + index_part.size()) {
                    throw std::invalid_argument(tf_wrap::detail::format(
                        "TensorName::parse: invalid index '{}'", index_part));
                }
                
                if (idx < 0) {
                    throw std::invalid_argument(tf_wrap::detail::format(
                        "TensorName::parse: negative index {}", idx));
                }
                
                result.index = idx;
                result.had_explicit_index = true;
            } else {
                // Colon is part of the op name (e.g. "scope/op:name")
                result.op = std::string(s);
                result.index = 0;
                result.had_explicit_index = false;
            }
        }
        
        if (result.op.empty()) {
            throw std::invalid_argument("TensorName::parse: empty operation name");
        }
        
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
    
    /// Construct from OpResult
    Endpoint(const ops::OpResult& result) : data_(result.output(0)) {}
    
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
    
    /// Add a feed (input tensor)
    Runner& feed(Endpoint endpoint, const Tensor& tensor) & {
        TF_Output output = resolve(endpoint);
        feeds_.push_back({output, tensor.handle()});
        return *this;
    }
    
    Runner&& feed(Endpoint endpoint, const Tensor& tensor) && {
        return std::move(feed(endpoint, tensor));
    }
    
    /// Add a feed from raw TF_Tensor*
    Runner& feed(Endpoint endpoint, TF_Tensor* tensor) & {
        TF_Output output = resolve(endpoint);
        feeds_.push_back({output, tensor});
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
        
        for (const auto& [output, tensor] : feeds_) {
            input_ops.push_back(output);
            input_vals.push_back(tensor);
        }
        
        std::vector<TF_Tensor*> output_vals(fetches_.size(), nullptr);
        
        Status st;
        TF_SessionRun(
            session_->handle(),
            nullptr,  // run_options
            input_ops.data(), input_vals.data(), static_cast<int>(feeds_.size()),
            fetches_.data(), output_vals.data(), static_cast<int>(fetches_.size()),
            targets_.data(), static_cast<int>(targets_.size()),
            nullptr,  // run_metadata
            st.get());
        
        // Take ownership of output tensors for exception safety
        struct TensorDeleter {
            void operator()(TF_Tensor* t) const noexcept {
                if (t) TF_DeleteTensor(t);
            }
        };
        std::vector<std::unique_ptr<TF_Tensor, TensorDeleter>> owned;
        owned.reserve(output_vals.size());
        for (auto* t : output_vals) {
            owned.emplace_back(t);
        }
        
        st.throw_if_error("Runner::run");
        
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
    }

private:
    const Session* session_;
    TF_Graph* graph_;
    
    std::vector<std::pair<TF_Output, TF_Tensor*>> feeds_;
    std::vector<TF_Output> fetches_;
    std::vector<TF_Operation*> targets_;
    
    // Per-runner resolution cache (no mutex needed - Runner is single-threaded)
    mutable std::unordered_map<std::string, TF_Output> cache_;
    
    TF_Output resolve(const Endpoint& endpoint) const {
        if (endpoint.is_resolved()) {
            return endpoint.as_output();
        }
        
        const TensorName& name = endpoint.as_name();
        const std::string key = name.to_string();
        
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            return it->second;
        }
        
        TF_Output output = name.to_output(graph_);
        cache_[key] = output;
        return output;
    }
};

// ============================================================================
// Model - High-level facade for SavedModel
// ============================================================================

class Model {
public:
    Model() = default;
    
    // Make Model moveable by using unique_ptr for mutex
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
        m.cache_mutex_ = std::make_unique<std::mutex>();
        return m;
    }
    
    /// Get a runner for this model
    [[nodiscard]] Runner runner() const {
        if (!session_) {
            throw std::runtime_error("Model::runner: model not loaded");
        }
        return Runner(*session_);
    }
    
    /// Resolve an output name to TF_Output (thread-safe, cached)
    [[nodiscard]] TF_Output resolve_output(const std::string& name) const {
        if (!graph_) {
            throw std::runtime_error("Model::resolve_output: model not loaded");
        }
        
        if (cache_mutex_) {
            std::lock_guard<std::mutex> lock(*cache_mutex_);
            
            auto it = output_cache_.find(name);
            if (it != output_cache_.end()) {
                return it->second;
            }
            
            TensorName tn = TensorName::parse(name);
            TF_Output output = tn.to_output(graph_->handle());
            output_cache_[name] = output;
            return output;
        }
        
        // No mutex (shouldn't happen after Load)
        TensorName tn = TensorName::parse(name);
        return tn.to_output(graph_->handle());
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
    
    std::unique_ptr<std::mutex> cache_mutex_;
    mutable std::unordered_map<std::string, TF_Output> output_cache_;
};

// ============================================================================
// Easy helpers - Create common ops with dtype inference
// ============================================================================

/// Create a scalar constant
template<TensorScalar T>
[[nodiscard]] inline ops::OpResult Scalar(
    Graph& graph,
    std::string_view name,
    T value)
{
    auto tensor = Tensor::FromScalar<T>(value);
    return ops::Const(graph, name, tensor.handle(), tf_dtype_v<T>);
}

/// Create a constant from tensor
template<TensorScalar T>
[[nodiscard]] inline ops::OpResult Const(
    Graph& graph,
    std::string_view name,
    const Tensor& tensor)
{
    return ops::Const(graph, name, tensor.handle(), tf_dtype_v<T>);
}

/// Create a placeholder
template<TensorScalar T>
[[nodiscard]] inline ops::OpResult Placeholder(
    Graph& graph,
    std::string_view name,
    std::span<const std::int64_t> shape = {})
{
    return ops::Placeholder(graph, name, tf_dtype_v<T>, shape);
}

/// Create an identity (pass-through)
template<TensorScalar T>
[[nodiscard]] inline ops::OpResult Identity(
    Graph& graph,
    std::string_view name,
    TF_Output input)
{
    return ops::Identity(graph, name, input, tf_dtype_v<T>);
}

// ============================================================================
// Dtype-inferred binary operations
// ============================================================================

namespace detail {

/// Get dtype from TF_Output and validate both inputs match
inline TF_DataType get_binary_dtype(TF_Output x, TF_Output y, const char* op_name) {
    const TF_DataType dx = TF_OperationOutputType(x);
    const TF_DataType dy = TF_OperationOutputType(y);
    
    if (dx != dy) {
        throw std::invalid_argument(tf_wrap::detail::format(
            "{}: input dtypes must match, got {} and {}",
            op_name, dtype_name(dx), dtype_name(dy)));
    }
    
    return dx;
}

} // namespace detail

/// Add with dtype inference
[[nodiscard]] inline ops::OpResult Add(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y)
{
    return ops::Add(graph, name, x, y, detail::get_binary_dtype(x, y, "Add"));
}

/// AddV2 with dtype inference
[[nodiscard]] inline ops::OpResult AddV2(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y)
{
    return ops::AddV2(graph, name, x, y, detail::get_binary_dtype(x, y, "AddV2"));
}

/// Sub with dtype inference
[[nodiscard]] inline ops::OpResult Sub(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y)
{
    return ops::Sub(graph, name, x, y, detail::get_binary_dtype(x, y, "Sub"));
}

/// Mul with dtype inference
[[nodiscard]] inline ops::OpResult Mul(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y)
{
    return ops::Mul(graph, name, x, y, detail::get_binary_dtype(x, y, "Mul"));
}

/// Div with dtype inference
[[nodiscard]] inline ops::OpResult Div(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output y)
{
    return ops::Div(graph, name, x, y, detail::get_binary_dtype(x, y, "Div"));
}

/// MatMul with dtype inference
[[nodiscard]] inline ops::OpResult MatMul(
    Graph& graph,
    std::string_view name,
    TF_Output a,
    TF_Output b)
{
    return ops::MatMul(graph, name, a, b, detail::get_binary_dtype(a, b, "MatMul"));
}

} // namespace easy

// Bring common items into tf_wrap namespace for convenience
using easy::TensorName;
using easy::Endpoint;
using easy::Runner;
using easy::Model;

} // namespace tf_wrap
