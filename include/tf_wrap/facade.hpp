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

namespace tf_wrap {
namespace facade {

// ============================================================================
// TensorName
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
            // Has colon - parse only if everything after is [0-9]+
            std::string_view index_part = s.substr(colon_pos + 1);

            bool all_digits = !index_part.empty() &&
                std::all_of(index_part.begin(), index_part.end(),
                    [](unsigned char c) { return std::isdigit(c); });

            if (all_digits) {
                result.op = std::string(s.substr(0, colon_pos));

                int idx = 0;
                auto [ptr, ec] = std::from_chars(
                    index_part.data(),
                    index_part.data() + index_part.size(),
                    idx);

                if (ec != std::errc{} || ptr != index_part.data() + index_part.size()) {
                    throw std::invalid_argument(tf_wrap::detail::format(
                        "TensorName::parse: invalid index '{}'", index_part));
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
};

// ============================================================================
// Endpoint
// ============================================================================

class Endpoint {
public:
    Endpoint() = default;

    Endpoint(const char* s) : value_(std::string_view(s)) {}
    Endpoint(std::string s) : value_(std::move(s)) {}
    Endpoint(std::string_view s) : value_(s) {}

    Endpoint(Operation op, int idx = 0) : value_(TF_Output{op.handle(), idx}) {}
    Endpoint(TF_Output out) : value_(out) {}

    [[nodiscard]] bool is_resolved() const noexcept {
        return std::holds_alternative<TF_Output>(value_);
    }

    [[nodiscard]] TF_Output as_output() const {
        return std::get<TF_Output>(value_);
    }

    [[nodiscard]] std::string_view as_string() const {
        if (const auto* sv = std::get_if<std::string_view>(&value_)) return *sv;
        if (const auto* str = std::get_if<std::string>(&value_)) return *str;
        throw std::runtime_error("Endpoint: not a string");
    }

private:
    std::variant<std::string_view, std::string, TF_Output> value_;
};

// ============================================================================
// Runner
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
            run_options_ ? run_options_->handle() : nullptr,  // run_options
            input_ops.data(), input_vals.data(), detail::checked_int(feeds_.size(), "Runner::run feeds"),
            fetches_.data(), output_vals.data(), detail::checked_int(fetches_.size(), "Runner::run fetches"),
            targets_.data(), detail::checked_int(targets_.size(), "Runner::run targets"),
            run_metadata_ ? run_metadata_->handle() : nullptr,  // run_metadata
            st.handle());

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
        run_options_ = nullptr;
        run_metadata_ = nullptr;
    }

private:
    const Session* session_;
    TF_Graph* graph_;
    const Buffer* run_options_{nullptr};
    Buffer* run_metadata_{nullptr};

    std::vector<std::pair<TF_Output, TF_Tensor*>> feeds_;
    std::vector<TF_Output> fetches_;
    std::vector<TF_Operation*> targets_;

    // Per-runner resolution cache (Runner is not thread-safe; do not share across threads)
    mutable std::unordered_map<std::string, TF_Output> cache_;

    TF_Output resolve(const Endpoint& endpoint) const {
        if (endpoint.is_resolved()) {
            return endpoint.as_output();
        }

        const auto key = std::string(endpoint.as_string());

        if (auto it = cache_.find(key); it != cache_.end()) {
            return it->second;
        }

        TensorName tn = TensorName::parse(endpoint.as_string());
        Operation op = Operation::ByName(graph_, tn.op);
        TF_Output out{op.handle(), tn.index};

        cache_.emplace(key, out);
        return out;
    }
};

// ============================================================================
// Model
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
        } catch (const std::exception& e) {
            throw std::runtime_error(tf_wrap::detail::format(
                "Model::Load: failed to load SavedModel from '{}': {} (check directory exists and contains saved_model.pb)",
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

    /// Call operator for simple 1-in / 1-out inference
    [[nodiscard]] Tensor operator()(Endpoint in, const Tensor& input, Endpoint out) const {
        return runner()
            .feed(in, input)
            .fetch(out)
            .run_one();
    }

    [[nodiscard]] const Session& session() const {
        if (!session_) throw std::runtime_error("Model::session: not loaded");
        return *session_;
    }

    [[nodiscard]] const Graph& graph() const {
        if (!graph_) throw std::runtime_error("Model::graph: not loaded");
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
