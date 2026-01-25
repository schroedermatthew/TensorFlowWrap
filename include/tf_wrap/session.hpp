// include/tf_wrap/session.hpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <tensorflow/c/c_api.h>

#include "tf_wrap/error.hpp"
#include "tf_wrap/graph.hpp"
#include "tf_wrap/status.hpp"
#include "tf_wrap/tensor.hpp"

namespace tf_wrap {

// ============================================================================
// Feed / Fetch / Target
// ============================================================================

struct Feed
{
    std::string op_name;
    int index = 0;

    TF_Output output{};

    Tensor tensor;

    Feed(std::string name, Tensor t, int i = 0)
        : op_name(std::move(name))
        , index(i)
        , output{}
        , tensor(std::move(t))
    {
    }

    Feed(TF_Output out, Tensor t)
        : op_name{}
        , index(0)
        , output(out)
        , tensor(std::move(t))
    {
    }

    Feed(TF_Operation* oper, int i, Tensor t)
        : op_name{}
        , index(0)
        , output{oper, i}
        , tensor(std::move(t))
    {
    }
};

struct Fetch
{
    std::string op_name;
    int index = 0;

    TF_Output output{};

    Fetch(std::string name, int i = 0)
        : op_name(std::move(name))
        , index(i)
        , output{}
    {
    }

    Fetch(TF_Output out)
        : op_name{}
        , index(0)
        , output(out)
    {
    }

    Fetch(TF_Operation* oper, int i)
        : op_name{}
        , index(0)
        , output{oper, i}
    {
    }
};

struct Target
{
    std::string op_name;
    TF_Operation* oper = nullptr;

    Target(std::string name)
        : op_name(std::move(name))
        , oper(nullptr)
    {
    }

    Target(TF_Operation* op)
        : op_name{}
        , oper(op)
    {
    }
};

// ============================================================================
// Session
// ============================================================================

class Session
{
public:
    explicit Session(Graph& graph);

    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    Session(Session&&) noexcept = default;
    Session& operator=(Session&&) noexcept = default;

    ~Session();

    // ------------------------------------------------------------------------
    // Name resolution helpers
    // ------------------------------------------------------------------------
    [[nodiscard]] TF_Operation* resolve_operation(std::string_view name) const;
    [[nodiscard]] TF_Output resolve_output(std::string_view name, int index) const;

    // ------------------------------------------------------------------------
    // Primary Run overload: span<const Target>
    // ------------------------------------------------------------------------
    [[nodiscard]] std::vector<Tensor> Run(const std::vector<Feed>& feeds,
                                         const std::vector<Fetch>& fetches,
                                         std::span<const Target> targets,
                                         TF_Buffer* run_options = nullptr,
                                         TF_Buffer* run_metadata = nullptr) const;

    // ------------------------------------------------------------------------
    // Convenience overload: initializer_list<Target>
    //
    // This is the critical overload that eliminates ambiguity for calls like:
    //   Run(feeds, fetches, {})
    //   Run(feeds, fetches, {"WriteFile"})
    //   Run(feeds, fetches, {Target{op_ptr}})
    // ------------------------------------------------------------------------
    [[nodiscard]] std::vector<Tensor> Run(const std::vector<Feed>& feeds,
                                         const std::vector<Fetch>& fetches,
                                         std::initializer_list<Target> targets,
                                         TF_Buffer* run_options = nullptr,
                                         TF_Buffer* run_metadata = nullptr) const
    {
        return Run(feeds,
                   fetches,
                   std::span<const Target>(targets.begin(), targets.size()),
                   run_options,
                   run_metadata);
    }

    // ------------------------------------------------------------------------
    // Back-compat overload: vector<string> targets
    // ------------------------------------------------------------------------
    [[nodiscard]] std::vector<Tensor> Run(const std::vector<Feed>& feeds,
                                         const std::vector<Fetch>& fetches,
                                         const std::vector<std::string>& targets,
                                         TF_Buffer* run_options = nullptr,
                                         TF_Buffer* run_metadata = nullptr) const
    {
        std::vector<Target> t;
        t.reserve(targets.size());
        for (const auto& n : targets) {
            t.emplace_back(n);
        }
        return Run(feeds, fetches, std::span<const Target>(t.data(), t.size()), run_options, run_metadata);
    }

    // ------------------------------------------------------------------------
    // Back-compat overload: vector<Target> (kept, but avoids brace ambiguity)
    // ------------------------------------------------------------------------
    [[nodiscard]] std::vector<Tensor> Run(const std::vector<Feed>& feeds,
                                         const std::vector<Fetch>& fetches,
                                         const std::vector<Target>& targets,
                                         TF_Buffer* run_options = nullptr,
                                         TF_Buffer* run_metadata = nullptr) const
    {
        return Run(feeds,
                   fetches,
                   std::span<const Target>(targets.data(), targets.size()),
                   run_options,
                   run_metadata);
    }

    // ------------------------------------------------------------------------
    // Batch helpers
    // ------------------------------------------------------------------------
    [[nodiscard]] std::vector<Tensor> BatchRun(std::string_view input_op,
                                               std::span<const Tensor> inputs,
                                               std::string_view output_op,
                                               int input_index = 0,
                                               int output_index = 0) const;

    [[nodiscard]] std::vector<Tensor> BatchRun(std::string_view input_op,
                                               const std::vector<Tensor>& inputs,
                                               std::string_view output_op,
                                               int input_index = 0,
                                               int output_index = 0) const
    {
        return BatchRun(input_op,
                        std::span<const Tensor>(inputs.data(), inputs.size()),
                        output_op,
                        input_index,
                        output_index);
    }

    // True batching: stack inputs into one run and split outputs.
    [[nodiscard]] std::vector<Tensor> BatchRunStacked(std::string_view input_op,
                                                      std::span<const Tensor> inputs,
                                                      std::string_view output_op,
                                                      int input_index = 0,
                                                      int output_index = 0) const;

    // ------------------------------------------------------------------------
    // Partial run
    // ------------------------------------------------------------------------
    [[nodiscard]] std::string PartialRunSetup(const std::vector<Feed>& feeds,
                                              const std::vector<Fetch>& fetches,
                                              std::span<const Target> targets) const;

    [[nodiscard]] std::string PartialRunSetup(const std::vector<Feed>& feeds,
                                              const std::vector<Fetch>& fetches,
                                              std::initializer_list<Target> targets) const
    {
        return PartialRunSetup(feeds, fetches, std::span<const Target>(targets.begin(), targets.size()));
    }

    [[nodiscard]] std::string PartialRunSetup(const std::vector<Feed>& feeds,
                                              const std::vector<Fetch>& fetches,
                                              const std::vector<std::string>& targets) const
    {
        std::vector<Target> t;
        t.reserve(targets.size());
        for (const auto& n : targets) {
            t.emplace_back(n);
        }
        return PartialRunSetup(feeds, fetches, std::span<const Target>(t.data(), t.size()));
    }

    [[nodiscard]] std::string PartialRunSetup(const std::vector<Feed>& feeds,
                                              const std::vector<Fetch>& fetches,
                                              const std::vector<Target>& targets) const
    {
        return PartialRunSetup(feeds, fetches, std::span<const Target>(targets.data(), targets.size()));
    }

    [[nodiscard]] std::vector<Tensor> PartialRun(std::string_view handle,
                                                 const std::vector<Feed>& feeds,
                                                 const std::vector<Fetch>& fetches,
                                                 std::span<const Target> targets) const;

    [[nodiscard]] std::vector<Tensor> PartialRun(std::string_view handle,
                                                 const std::vector<Feed>& feeds,
                                                 const std::vector<Fetch>& fetches,
                                                 std::initializer_list<Target> targets) const
    {
        return PartialRun(handle, feeds, fetches, std::span<const Target>(targets.begin(), targets.size()));
    }

    [[nodiscard]] std::vector<Tensor> PartialRun(std::string_view handle,
                                                 const std::vector<Feed>& feeds,
                                                 const std::vector<Fetch>& fetches,
                                                 const std::vector<std::string>& targets) const
    {
        std::vector<Target> t;
        t.reserve(targets.size());
        for (const auto& n : targets) {
            t.emplace_back(n);
        }
        return PartialRun(handle, feeds, fetches, std::span<const Target>(t.data(), t.size()));
    }

    [[nodiscard]] std::vector<Tensor> PartialRun(std::string_view handle,
                                                 const std::vector<Feed>& feeds,
                                                 const std::vector<Fetch>& fetches,
                                                 const std::vector<Target>& targets) const
    {
        return PartialRun(handle, feeds, fetches, std::span<const Target>(targets.data(), targets.size()));
    }

private:
    std::shared_ptr<detail::GraphState> graph_state_;
    TF_Session* session_ = nullptr;

    [[nodiscard]] TF_Output resolve_output_or_throw(const Fetch& f) const;
    [[nodiscard]] TF_Output resolve_output_or_throw(const Feed& f) const;
    [[nodiscard]] TF_Operation* resolve_target_or_throw(const Target& t) const;

    void validate_output_index_or_throw(TF_Operation* oper, int index, std::string_view what) const;
};

} // namespace tf_wrap
