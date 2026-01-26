// tf_wrap/session.hpp
// RAII wrapper for TF_Session - Production Inference Edition
//
// Thread safety contract:
// - Session::Run() is thread-safe (TensorFlow's guarantee)
// - Graph is frozen after Session creation
// - For multi-threaded serving, each request should have its own input tensors

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

extern "C" {
#include <tensorflow/c/c_api.h>
}

#include "tf_wrap/error.hpp"
#include "tf_wrap/format.hpp"
#include "tf_wrap/graph.hpp"
#include "tf_wrap/scope_guard.hpp"
#include "tf_wrap/small_vector.hpp"
#include "tf_wrap/status.hpp"
#include "tf_wrap/tensor.hpp"
#include "tf_wrap/detail/raw_tensor_ptr.hpp"

namespace tf_wrap {
namespace detail {

/// Internal helper: call TF_SessionRun and throw on error.
/// Also validates that all fetched outputs are non-null.
///
/// This exists so Session and higher-level runners share a single, consistent
/// TF_SessionRun call path (avoids drift).
inline void session_run_checked(
    TF_Session* session,
    const TF_Buffer* run_options,
    const TF_Output* input_ops,
    TF_Tensor* const* input_vals,
    int num_inputs,
    const TF_Output* output_ops,
    TF_Tensor** output_vals,
    int num_outputs,
    TF_Operation* const* target_ops,
    int num_targets,
    TF_Buffer* run_metadata,
    std::string_view context,
    std::source_location loc)
{
    Status st;
    TF_SessionRun(
        session,
        run_options,
        input_ops, input_vals, num_inputs,
        output_ops, output_vals, num_outputs,
        target_ops, num_targets,
        run_metadata,
        st.get());

    // If TF_SessionRun failed, TensorFlow may still have allocated some output
    // tensors. Since we throw on error, ensure we don't leak them.
    if (!st.ok()) {
        for (int i = 0; i < num_outputs; ++i) {
            if (output_vals[i]) {
                TF_DeleteTensor(output_vals[i]);
                output_vals[i] = nullptr;
            }
        }
        st.throw_if_error(context, loc);
    }

    // Verify all requested outputs are non-null.
    for (int i = 0; i < num_outputs; ++i) {
        if (output_vals[i] != nullptr) continue;

        // Defensive cleanup: if any requested fetch is null, free all outputs
        // before throwing.
        for (int j = 0; j < num_outputs; ++j) {
            if (output_vals[j]) {
                TF_DeleteTensor(output_vals[j]);
                output_vals[j] = nullptr;
            }
        }

        const TF_Output out = output_ops[i];
        const char* op_name = out.oper ? TF_OperationName(out.oper) : "";
        throw Error::Wrapper(
            TF_INTERNAL,
            context,
            "fetch returned null tensor",
            op_name ? op_name : "",
            out.index,
            loc);
    }
}

} // namespace detail


// ============================================================================
// SessionOptions - RAII wrapper for TF_SessionOptions
// ============================================================================

class SessionOptions {
public:
    SessionOptions() : opts_(TF_NewSessionOptions()) {
        if (!opts_) {
            throw Error::Wrapper(TF_INTERNAL, "SessionOptions", 
                "TF_NewSessionOptions failed", "", -1);
        }
    }
    
    ~SessionOptions() { if (opts_) TF_DeleteSessionOptions(opts_); }
    
    SessionOptions(const SessionOptions&) = delete;
    SessionOptions& operator=(const SessionOptions&) = delete;
    
    SessionOptions(SessionOptions&& other) noexcept : opts_(other.opts_) {
        other.opts_ = nullptr;
    }
    
    SessionOptions& operator=(SessionOptions&& other) noexcept {
        if (this != &other) {
            if (opts_) TF_DeleteSessionOptions(opts_);
            opts_ = other.opts_;
            other.opts_ = nullptr;
        }
        return *this;
    }
    
    SessionOptions& SetConfig(const void* proto, std::size_t len) {
        Status st;
        TF_SetConfig(opts_, proto, len, st.get());
        st.throw_if_error("TF_SetConfig");
        return *this;
    }
    
    SessionOptions& SetTarget(const char* target) {
        TF_SetTarget(opts_, target);
        return *this;
    }
    
    [[nodiscard]] TF_SessionOptions* handle() const noexcept { return opts_; }

private:
    TF_SessionOptions* opts_;
};

// ============================================================================
// Feed/Fetch/Target - Simplified handle-only structures
// ============================================================================

/// Input feed for Session::Run (handle-based only)
struct Feed {
    TF_Output output;
    TF_Tensor* tensor{nullptr};
    std::shared_ptr<const void> keepalive{};

    Feed(TF_Output out, const Tensor& t)
        : output(out), tensor(t.handle()), keepalive(t.keepalive()) {}
    
    Feed(TF_Output out, TF_Tensor* t)
        : output(out), tensor(t) {}

    Feed(TF_Operation* op, int idx, const Tensor& t)
        : Feed(TF_Output{op, idx}, t) {}

    Feed(TF_Operation* op, const Tensor& t)
        : Feed(TF_Output{op, 0}, t) {}
};

/// Output fetch for Session::Run (handle-based only)
struct Fetch {
    TF_Output output;

    Fetch(TF_Output out) : output(out) {}
    Fetch(TF_Operation* op, int idx = 0) : output{op, idx} {}
};

/// Target operation for Session::Run (handle-based only)
struct Target {
    TF_Operation* oper;

    Target(TF_Operation* op) : oper(op) {}
};

// ============================================================================
// Buffer - RAII wrapper for TF_Buffer
// ============================================================================

class Buffer {
public:
    Buffer() : buf_(TF_NewBuffer()) {
        if (!buf_) {
            throw Error::Wrapper(TF_INTERNAL, "Buffer", 
                "TF_NewBuffer failed", "", -1);
        }
    }
    
    explicit Buffer(const void* data, std::size_t len)
        : buf_(TF_NewBufferFromString(data, len))
    {
        if (!buf_) {
            throw Error::Wrapper(TF_INTERNAL, "Buffer", 
                "TF_NewBufferFromString failed", "", -1);
        }
    }
    
    ~Buffer() { if (buf_) TF_DeleteBuffer(buf_); }
    
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    
    Buffer(Buffer&& other) noexcept : buf_(other.buf_) { other.buf_ = nullptr; }
    
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            if (buf_) TF_DeleteBuffer(buf_);
            buf_ = other.buf_;
            other.buf_ = nullptr;
        }
        return *this;
    }
    
    [[nodiscard]] TF_Buffer* handle() const noexcept { return buf_; }
    [[nodiscard]] const void* data() const noexcept { return buf_ ? buf_->data : nullptr; }
    [[nodiscard]] std::size_t length() const noexcept { return buf_ ? buf_->length : 0; }
    [[nodiscard]] bool empty() const noexcept { return !buf_ || buf_->length == 0; }
    
    [[nodiscard]] std::vector<std::uint8_t> to_bytes() const {
        if (!buf_ || !buf_->data || buf_->length == 0) return {};
        const auto* p = static_cast<const std::uint8_t*>(buf_->data);
        return std::vector<std::uint8_t>(p, p + buf_->length);
    }

private:
    TF_Buffer* buf_;
};

// ============================================================================
// Device - Information about a compute device
// ============================================================================

struct Device {
    std::string name;
    std::string type;
    std::int64_t memory_bytes{0};
    
    [[nodiscard]] bool is_gpu() const noexcept { return type == "GPU"; }
    [[nodiscard]] bool is_cpu() const noexcept { return type == "CPU"; }
};

// ============================================================================
// DeviceList - RAII wrapper for TF_DeviceList
// ============================================================================

class DeviceList {
public:
    DeviceList() = default;
    explicit DeviceList(TF_DeviceList* list) : list_(list) {}
    
    ~DeviceList() { if (list_) TF_DeleteDeviceList(list_); }
    
    DeviceList(const DeviceList&) = delete;
    DeviceList& operator=(const DeviceList&) = delete;
    
    DeviceList(DeviceList&& other) noexcept : list_(other.list_) { other.list_ = nullptr; }
    
    DeviceList& operator=(DeviceList&& other) noexcept {
        if (this != &other) {
            if (list_) TF_DeleteDeviceList(list_);
            list_ = other.list_;
            other.list_ = nullptr;
        }
        return *this;
    }
    
    [[nodiscard]] int count() const noexcept { return list_ ? TF_DeviceListCount(list_) : 0; }
    
    [[nodiscard]] Device at(int index) const {
        if (!list_ || index < 0 || index >= count()) {
            throw std::out_of_range("DeviceList::at: index out of range");
        }
        
        Device dev;
        Status st;
        
        const char* name = TF_DeviceListName(list_, index, st.get());
        st.throw_if_error("TF_DeviceListName");
        dev.name = name ? name : "";
        
        const char* type = TF_DeviceListType(list_, index, st.get());
        st.throw_if_error("TF_DeviceListType");
        dev.type = type ? type : "";
        
        dev.memory_bytes = TF_DeviceListMemoryBytes(list_, index, st.get());
        st.throw_if_error("TF_DeviceListMemoryBytes");
        
        return dev;
    }
    
    [[nodiscard]] std::vector<Device> all() const {
        std::vector<Device> devices;
        const int n = count();
        devices.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            devices.push_back(at(i));
        }
        return devices;
    }
    
    [[nodiscard]] TF_DeviceList* handle() const noexcept { return list_; }

private:
    TF_DeviceList* list_{nullptr};
};

// ============================================================================
// RunContext - Reusable buffers for zero-allocation hot path
// ============================================================================

class RunContext {
public:
    explicit RunContext(std::size_t max_feeds = 8, std::size_t max_fetches = 4) {
        input_ops_.reserve(max_feeds);
        input_vals_.reserve(max_feeds);
        output_ops_.reserve(max_fetches);
        output_vals_.reserve(max_fetches);
    }
    
    void reset() noexcept {
        input_ops_.clear();
        input_vals_.clear();
        output_ops_.clear();
        output_vals_.clear();
        target_ops_.clear();
        keepalives_.clear();
    }
    
    void add_feed(TF_Output output, const Tensor& tensor) {
        input_ops_.push_back(output);
        input_vals_.push_back(tensor.handle());
        if (auto ka = tensor.keepalive()) {
            keepalives_.push_back(std::move(ka));
        }
    }
    
    void add_feed(TF_Output output, TF_Tensor* tensor) {
        input_ops_.push_back(output);
        input_vals_.push_back(tensor);
    }
    
    void add_fetch(TF_Output output) {
        output_ops_.push_back(output);
    }
    
    void add_target(TF_Operation* op) {
        target_ops_.push_back(op);
    }

private:
    friend class Session;
    std::vector<TF_Output> input_ops_;
    std::vector<TF_Tensor*> input_vals_;
    std::vector<TF_Output> output_ops_;
    std::vector<TF_Tensor*> output_vals_;
    std::vector<TF_Operation*> target_ops_;
    std::vector<std::shared_ptr<const void>> keepalives_;
};

// ============================================================================
// Session - RAII wrapper for TF_Session
// ============================================================================

class Session {
public:
    explicit Session(Graph& graph, const SessionOptions& opts = SessionOptions())
        : graph_state_(graph.share_state())
    {
        if (!graph_state_ || !graph_state_->graph) {
            throw Error::Wrapper(TF_FAILED_PRECONDITION, "Session",
                "cannot create session from moved-from graph", "", -1);
        }

        Status st;
        session_ = TF_NewSession(graph_state_->graph, opts.handle(), st.get());
        st.throw_if_error("TF_NewSession");

        graph_state_->frozen = true;
    }
    
    explicit Session(Graph& graph, TF_SessionOptions* opts)
        : graph_state_(graph.share_state())
    {
        if (!graph_state_ || !graph_state_->graph) {
            throw Error::Wrapper(TF_FAILED_PRECONDITION, "Session",
                "cannot create session from moved-from graph", "", -1);
        }

        Status st;
        session_ = TF_NewSession(graph_state_->graph, opts, st.get());
        st.throw_if_error("TF_NewSession");

        graph_state_->frozen = true;
    }
    
    ~Session() noexcept { Cleanup(); }
    
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    
    Session(Session&& other) noexcept
        : session_(other.session_)
        , graph_state_(std::move(other.graph_state_))
    {
        other.session_ = nullptr;
    }
    
    Session& operator=(Session&& other) noexcept {
        if (this != &other) {
            Cleanup();
            session_ = other.session_;
            graph_state_ = std::move(other.graph_state_);
            other.session_ = nullptr;
        }
        return *this;
    }

    // ─────────────────────────────────────────────────────────────────
    // Resolve - Convert name to TF_Output (call once at startup)
    // ─────────────────────────────────────────────────────────────────
    
    /// Resolve an operation name to TF_Output. Call once at startup, cache the result.
    [[nodiscard]] TF_Output resolve(
        std::string_view name,
        std::source_location loc = std::source_location::current()) const
    {
        if (!graph_state_ || !graph_state_->graph) {
            throw Error::Wrapper(TF_FAILED_PRECONDITION, "Session::resolve",
                "session has no graph", std::string(name), -1, loc);
        }

        // Parse "op_name:index" format
        std::string op_name;
        int index = 0;
        
        auto colon_pos = name.rfind(':');
        if (colon_pos != std::string_view::npos && colon_pos + 1 < name.size()) {
            // Check if everything after colon is digits
            auto suffix = name.substr(colon_pos + 1);
            bool all_digits = !suffix.empty() && 
                std::all_of(suffix.begin(), suffix.end(), 
                    [](unsigned char c) { return std::isdigit(c); });
            
            if (all_digits) {
                op_name = std::string(name.substr(0, colon_pos));
                index = 0;
                for (char c : suffix) {
                    index = index * 10 + (c - '0');
                }
            } else {
                op_name = std::string(name);
            }
        } else {
            op_name = std::string(name);
        }
        
        TF_Operation* op = TF_GraphOperationByName(graph_state_->graph, op_name.c_str());
        if (!op) {
            throw Error::Wrapper(TF_NOT_FOUND, "Session::resolve",
                "operation not found in graph", op_name, index, loc);
        }
        
        const int num_outputs = TF_OperationNumOutputs(op);
        if (index < 0 || index >= num_outputs) {
            throw Error::Wrapper(TF_OUT_OF_RANGE, "Session::resolve",
                detail::format("output index {} out of range (operation has {} outputs)",
                    index, num_outputs),
                op_name, index, loc);
        }
        
        return TF_Output{op, index};
    }

    // ─────────────────────────────────────────────────────────────────
    // Validation helpers (production safety)
    // ─────────────────────────────────────────────────────────────────

    /// Validate that a TF_Output is usable with this session (belongs to this graph,
    /// index is in range, and oper is non-null). Useful to fail fast at startup.
    void validate_output(
        TF_Output out,
        std::source_location loc = std::source_location::current()) const
    {
        if (!graph_state_ || !graph_state_->graph) {
            throw Error::Wrapper(
                TF_FAILED_PRECONDITION,
                "Session::validate_output",
                "session has no graph",
                "",
                -1,
                loc);
        }
        if (!out.oper) {
            throw Error::Wrapper(
                TF_INVALID_ARGUMENT,
                "Session::validate_output",
                "null TF_Output.oper",
                "",
                out.index,
                loc);
        }

        const int n = TF_OperationNumOutputs(out.oper);
        if (out.index < 0 || out.index >= n) {
            const char* op_name = TF_OperationName(out.oper);
            throw Error::Wrapper(
                TF_OUT_OF_RANGE,
                "Session::validate_output",
                detail::format("output index {} out of range (operation has {} outputs)", out.index, n),
                op_name ? op_name : "",
                out.index,
                loc);
        }

        // Defensive: ensure the operation pointer belongs to this graph.
        const char* op_name = TF_OperationName(out.oper);
        TF_Operation* by_name = TF_GraphOperationByName(graph_state_->graph, op_name ? op_name : "");
        if (by_name != out.oper) {
            throw Error::Wrapper(
                TF_INVALID_ARGUMENT,
                "Session::validate_output",
                "TF_Output does not belong to this session's graph",
                op_name ? op_name : "",
                out.index,
                loc);
        }
    }

    /// Validate that a TF_Operation belongs to this session's graph.
    void validate_operation(
        TF_Operation* op,
        std::source_location loc = std::source_location::current()) const
    {
        if (!graph_state_ || !graph_state_->graph) {
            throw Error::Wrapper(
                TF_FAILED_PRECONDITION,
                "Session::validate_operation",
                "session has no graph",
                "",
                -1,
                loc);
        }
        if (!op) {
            throw Error::Wrapper(
                TF_INVALID_ARGUMENT,
                "Session::validate_operation",
                "null TF_Operation*",
                "",
                -1,
                loc);
        }

        const char* op_name = TF_OperationName(op);
        TF_Operation* by_name = TF_GraphOperationByName(graph_state_->graph, op_name ? op_name : "");
        if (by_name != op) {
            throw Error::Wrapper(
                TF_INVALID_ARGUMENT,
                "Session::validate_operation",
                "TF_Operation does not belong to this session's graph",
                op_name ? op_name : "",
                -1,
                loc);
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // Run - Execute the graph (handle-based, production fast path)
    // ─────────────────────────────────────────────────────────────────
    
    /// Run with spans (primary implementation)
    [[nodiscard]] std::vector<Tensor> Run(
        std::span<const Feed> feeds,
        std::span<const Fetch> fetches,
        std::span<const Target> targets = {},
        TF_Buffer* run_options = nullptr,
        TF_Buffer* run_metadata = nullptr,
        std::source_location loc = std::source_location::current()) const
    {
        if (!session_) {
            throw Error::Wrapper(TF_FAILED_PRECONDITION, "Session::Run",
                "session is null", "", -1, loc);
        }

        // Use SmallVector for stack allocation in common case
        SmallVector<TF_Output, 8> input_ops;
        SmallVector<TF_Tensor*, 8> input_vals;
        SmallVector<TF_Output, 8> output_ops;
        SmallVector<TF_Operation*, 4> target_ops;
        
        input_ops.reserve(feeds.size());
        input_vals.reserve(feeds.size());
        output_ops.reserve(fetches.size());
        target_ops.reserve(targets.size());

        for (const auto& f : feeds) {
            input_ops.push_back(f.output);
            input_vals.push_back(f.tensor);
        }
        
        for (const auto& f : fetches) {
            output_ops.push_back(f.output);
        }
        
        for (const auto& t : targets) {
            target_ops.push_back(t.oper);
        }
        
        std::vector<TF_Tensor*> output_vals(fetches.size(), nullptr);

        const int num_inputs = detail::checked_int(feeds.size(), "Session::Run feeds");
        const int num_outputs = detail::checked_int(fetches.size(), "Session::Run fetches");
        const int num_targets = detail::checked_int(targets.size(), "Session::Run targets");
        
        detail::session_run_checked(
            session_,
            run_options,
            input_ops.data(),
            input_vals.data(),
            num_inputs,
            output_ops.data(),
            output_vals.data(),
            num_outputs,
            target_ops.data(),
            num_targets,
            run_metadata,
            "Session::Run",
            loc);

        // Take ownership immediately for exception safety
        using RawTensorPtr = detail::RawTensorPtr;
        std::vector<RawTensorPtr> owned;
        owned.reserve(output_vals.size());
        for (auto* t : output_vals) {
            owned.emplace_back(t);
        }

        std::vector<Tensor> results;
        results.reserve(owned.size());
        for (auto& p : owned) {
            results.push_back(Tensor::FromRaw(p.release()));
        }

        return results;
    }
    
    /// Run with vectors (convenience overload)
    [[nodiscard]] std::vector<Tensor> Run(
        const std::vector<Feed>& feeds,
        const std::vector<Fetch>& fetches,
        const std::vector<Target>& targets = {},
        TF_Buffer* run_options = nullptr,
        TF_Buffer* run_metadata = nullptr,
        std::source_location loc = std::source_location::current()) const
    {
        return Run(
            std::span<const Feed>(feeds),
            std::span<const Fetch>(fetches),
            std::span<const Target>(targets),
            run_options, run_metadata, loc);
    }
    
    /// Run with pre-allocated context (zero allocations in hot path)
    [[nodiscard]] std::vector<Tensor> Run(
        RunContext& ctx,
        TF_Buffer* run_options = nullptr,
        TF_Buffer* run_metadata = nullptr,
        std::source_location loc = std::source_location::current()) const
    {
        if (!session_) {
            throw Error::Wrapper(TF_FAILED_PRECONDITION, "Session::Run",
                "session is null", "", -1, loc);
        }

        ctx.output_vals_.resize(ctx.output_ops_.size(), nullptr);

        const int num_inputs = detail::checked_int(ctx.input_ops_.size(), "feeds");
        const int num_outputs = detail::checked_int(ctx.output_ops_.size(), "fetches");
        const int num_targets = detail::checked_int(ctx.target_ops_.size(), "targets");
        
        detail::session_run_checked(
            session_,
            run_options,
            ctx.input_ops_.data(),
            ctx.input_vals_.data(),
            num_inputs,
            ctx.output_ops_.data(),
            ctx.output_vals_.data(),
            num_outputs,
            ctx.target_ops_.data(),
            num_targets,
            run_metadata,
            "Session::Run",
            loc);

        // Take ownership
        using RawTensorPtr = detail::RawTensorPtr;
        std::vector<RawTensorPtr> owned;
        owned.reserve(ctx.output_vals_.size());
        for (auto* t : ctx.output_vals_) {
            owned.emplace_back(t);
        }

        std::vector<Tensor> results;
        results.reserve(owned.size());
        for (auto& p : owned) {
            results.push_back(Tensor::FromRaw(p.release()));
        }

        return results;
    }

    // ─────────────────────────────────────────────────────────────────
    // BatchRun - Run inference on multiple inputs
    // ─────────────────────────────────────────────────────────────────
    
    /// Run same input->output mapping for multiple inputs (one TF call per input)
    [[nodiscard]] std::vector<Tensor> BatchRun(
        TF_Output input,
        std::span<const Tensor> inputs,
        TF_Output output,
        std::source_location loc = std::source_location::current()) const
    {
        if (!session_) {
            throw Error::Wrapper(TF_FAILED_PRECONDITION, "Session::BatchRun",
                "session is null", "", -1, loc);
        }

        std::vector<Tensor> results;
        results.reserve(inputs.size());

        Status st;
        TF_Output input_ops[1] = {input};
        TF_Output output_ops[1] = {output};

        for (const Tensor& t : inputs) {
            if (!t.handle()) {
                throw Error::Wrapper(TF_INVALID_ARGUMENT, "Session::BatchRun",
                    "input tensor is null", "", -1, loc);
            }

            TF_Tensor* input_vals[1] = {t.handle()};
            TF_Tensor* output_vals[1] = {nullptr};

            st.reset();
            TF_SessionRun(
                session_,
                nullptr,
                input_ops, input_vals, 1,
                output_ops, output_vals, 1,
                nullptr, 0,
                nullptr,
                st.get());

            detail::RawTensorPtr owned(output_vals[0]);
                if (!owned) {
                const char* out_name = TF_OperationName(output.oper);
                throw Error::Wrapper(TF_INTERNAL, "TF_SessionRun",
                    "fetch returned null tensor",
                    out_name ? out_name : "", output.index, loc);
            }

            results.push_back(Tensor::FromRaw(owned.release()));
        }

        return results;
    }

    [[nodiscard]] std::vector<Tensor> BatchRun(
        TF_Output input,
        const std::vector<Tensor>& inputs,
        TF_Output output,
        std::source_location loc = std::source_location::current()) const
    {
        return BatchRun(input, std::span<const Tensor>(inputs), output, loc);
    }

    // ─────────────────────────────────────────────────────────────────
    // BatchRunStacked - True batching with single TF call
    // ─────────────────────────────────────────────────────────────────

    /// Stack inputs into batch tensor, run once, split outputs
    [[nodiscard]] std::vector<Tensor> BatchRunStacked(
        TF_Output input,
        std::span<const Tensor> inputs,
        TF_Output output,
        std::source_location loc = std::source_location::current()) const
    {
        if (!session_) {
            throw Error::Wrapper(TF_FAILED_PRECONDITION, "Session::BatchRunStacked",
                "session is null", "", -1, loc);
        }
        if (inputs.empty()) {
            return {};
        }

        const Tensor& first = inputs.front();
        if (!first.handle()) {
            throw Error::Wrapper(TF_INVALID_ARGUMENT, "Session::BatchRunStacked",
                "first input tensor is null", "", -1, loc);
        }
        
        const TF_DataType dtype = first.dtype();
        const auto item_shape = first.shape();
        const std::size_t elem_size = TF_DataTypeSize(dtype);
        
        if (elem_size == 0) {
            throw Error::Wrapper(TF_INVALID_ARGUMENT, "Session::BatchRunStacked",
                "variable-length dtype not supported for stacked batching", "", -1, loc);
        }

        // Compute item size
        std::size_t item_elems = 1;
        for (auto d : item_shape) {
            item_elems *= static_cast<std::size_t>(d);
        }
        const std::size_t item_bytes = item_elems * elem_size;

        // Build batch shape: [N, ...item_shape]
        std::vector<std::int64_t> batch_shape;
        batch_shape.reserve(1 + item_shape.size());
        batch_shape.push_back(static_cast<std::int64_t>(inputs.size()));
        batch_shape.insert(batch_shape.end(), item_shape.begin(), item_shape.end());

        // Allocate and fill batch tensor
        const std::size_t batch_bytes = inputs.size() * item_bytes;
        TF_Tensor* batch_raw = TF_AllocateTensor(dtype, 
            batch_shape.data(), 
            static_cast<int>(batch_shape.size()), 
            batch_bytes);
        
        if (!batch_raw) {
            throw Error::Wrapper(TF_RESOURCE_EXHAUSTED, "Session::BatchRunStacked",
                "failed to allocate batch tensor", "", -1, loc);
        }
        
        Tensor batch_tensor = Tensor::FromRaw(batch_raw);
        std::byte* dst = static_cast<std::byte*>(TF_TensorData(batch_tensor.handle()));

        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const Tensor& t = inputs[i];
            if (!t.handle()) {
                throw Error::Wrapper(TF_INVALID_ARGUMENT, "Session::BatchRunStacked",
                    "input tensor is null", "", static_cast<int>(i), loc);
            }
            if (t.dtype() != dtype || t.shape() != item_shape) {
                throw Error::Wrapper(TF_INVALID_ARGUMENT, "Session::BatchRunStacked",
                    "input tensor has mismatched dtype or shape", "", static_cast<int>(i), loc);
            }
            
            const void* src = TF_TensorData(t.handle());
            std::memcpy(dst + i * item_bytes, src, item_bytes);
        }

        // Run inference
        TF_Output input_ops[1] = {input};
        TF_Output output_ops[1] = {output};
        TF_Tensor* input_vals[1] = {batch_tensor.handle()};
        TF_Tensor* output_vals[1] = {nullptr};

        Status st;
        TF_SessionRun(
            session_,
            nullptr,
            input_ops, input_vals, 1,
            output_ops, output_vals, 1,
            nullptr, 0,
            nullptr,
            st.get());

        detail::RawTensorPtr out_owned(output_vals[0]);
        if (!out_owned) {
            throw Error::Wrapper(TF_INTERNAL, "Session::BatchRunStacked",
                "fetch returned null tensor", "", -1, loc);
        }

        Tensor out_batched = Tensor::FromRaw(out_owned.release());

        // Split output
        const TF_DataType out_dtype = out_batched.dtype();
        const std::size_t out_elem_size = TF_DataTypeSize(out_dtype);
        const auto out_shape = out_batched.shape();
        
        if (out_shape.empty() || out_shape[0] != static_cast<std::int64_t>(inputs.size())) {
            throw Error::Wrapper(TF_INTERNAL, "Session::BatchRunStacked",
                "output batch dimension mismatch", "", -1, loc);
        }

        std::vector<std::int64_t> out_item_shape(out_shape.begin() + 1, out_shape.end());
        std::size_t out_item_elems = 1;
        for (auto d : out_item_shape) {
            out_item_elems *= static_cast<std::size_t>(d);
        }
        const std::size_t out_item_bytes = out_item_elems * out_elem_size;

        const std::byte* out_src = static_cast<const std::byte*>(TF_TensorData(out_batched.handle()));

        std::vector<Tensor> results;
        results.reserve(inputs.size());
        
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            TF_Tensor* item_raw = TF_AllocateTensor(out_dtype,
                out_item_shape.data(),
                static_cast<int>(out_item_shape.size()),
                out_item_bytes);
            
            if (!item_raw) {
                throw Error::Wrapper(TF_RESOURCE_EXHAUSTED, "Session::BatchRunStacked",
                    "failed to allocate output tensor", "", static_cast<int>(i), loc);
            }
            
            void* item_dst = TF_TensorData(item_raw);
            std::memcpy(item_dst, out_src + i * out_item_bytes, out_item_bytes);
            results.push_back(Tensor::FromRaw(item_raw));
        }

        return results;
    }

    // ─────────────────────────────────────────────────────────────────
    // Device enumeration
    // ─────────────────────────────────────────────────────────────────
    
    [[nodiscard]] DeviceList ListDevices() const {
        if (!session_) {
            throw Error::Wrapper(TF_FAILED_PRECONDITION, "Session::ListDevices",
                "session is null", "", -1);
        }
        
        Status st;
        TF_DeviceList* list = TF_SessionListDevices(session_, st.get());
        st.throw_if_error("TF_SessionListDevices");
        
        return DeviceList(list);
    }
    
    [[nodiscard]] bool HasGPU() const {
        auto devices = ListDevices();
        for (int i = 0; i < devices.count(); ++i) {
            if (devices.at(i).is_gpu()) return true;
        }
        return false;
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Handle access
    // ─────────────────────────────────────────────────────────────────
    
    [[nodiscard]] TF_Session* handle() const noexcept { return session_; }
    [[nodiscard]] TF_Graph* graph_handle() const noexcept {
        return graph_state_ ? graph_state_->graph : nullptr;
    }
    [[nodiscard]] bool valid() const noexcept { return session_ != nullptr; }
    
    // ─────────────────────────────────────────────────────────────────
    // SavedModel loading
    // ─────────────────────────────────────────────────────────────────
    
    [[nodiscard]] static std::pair<Session, Graph> LoadSavedModel(
        const std::string& export_dir,
        const std::vector<std::string>& tags = {"serve"},
        const SessionOptions& opts = SessionOptions())
    {
        Graph graph;
        
        std::vector<const char*> tag_ptrs;
        tag_ptrs.reserve(tags.size());
        for (const auto& t : tags) {
            tag_ptrs.push_back(t.c_str());
        }
        
        Status st;
        TF_Session* raw_session = TF_LoadSessionFromSavedModel(
            opts.handle(),
            nullptr,
            export_dir.c_str(),
            tag_ptrs.data(),
            detail::checked_int(tags.size(), "Session::LoadSavedModel tags"),
            graph.handle(),
            nullptr,
            st.get());
        
        st.throw_if_error("TF_LoadSessionFromSavedModel");
        
        graph.freeze();
        
        Session session;
        session.session_ = raw_session;
        session.graph_state_ = graph.share_state();
        
        return {std::move(session), std::move(graph)};
    }

private:
    TF_Session* session_{nullptr};
    std::shared_ptr<detail::GraphState> graph_state_{};
    
    Session() = default;
    
    void Cleanup() noexcept {
        if (!session_) return;
        
        TF_Status* st = TF_NewStatus();
        if (st) {
            TF_SCOPE_EXIT { TF_DeleteStatus(st); };
            TF_CloseSession(session_, st);
            TF_DeleteSession(session_, st);
        }
        
        session_ = nullptr;
        graph_state_.reset();
    }
};

} // namespace tf_wrap
