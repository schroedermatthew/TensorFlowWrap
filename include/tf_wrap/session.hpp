// tf/session.hpp
// RAII wrapper for TF_Session
//
// v5 - Thread safety policies REMOVED (simplification):
// - Single Session class (no policy templates)
// - No mutex/locking machinery
//
// Thread safety contract:
// - Session::Run() is thread-safe (TensorFlow's guarantee)
// - Graph is frozen after Session creation (this wrapper's policy choice)
// - For multi-threaded serving, each request should have its own input tensors

#pragma once

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

extern "C" {
#include <tensorflow/c/c_api.h>
}

#include "tf_wrap/format.hpp"
#include "tf_wrap/graph.hpp"
#include "tf_wrap/scope_guard.hpp"
#include "tf_wrap/status.hpp"
#include "tf_wrap/tensor.hpp"
#include "tf_wrap/tensor_name.hpp"

namespace tf_wrap {

// ============================================================================
// SessionOptions - RAII wrapper for TF_SessionOptions
// ============================================================================

class SessionOptions {
public:
    SessionOptions() : opts_(TF_NewSessionOptions()) {
        if (!opts_) throw std::runtime_error("TF_NewSessionOptions failed");
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
    
    [[nodiscard]] TF_SessionOptions* get() const noexcept { return opts_; }
    [[nodiscard]] TF_SessionOptions* handle() const noexcept { return opts_; }

private:
    TF_SessionOptions* opts_;
};

// ============================================================================
// Feed/Fetch structures for Session::Run
// ============================================================================

struct Feed {
    std::string op_name;
    int index{0};
    TF_Tensor* tensor{nullptr};

    // Optional pre-resolved output handle (avoids string lookup in Session::Run).
    TF_Output output{nullptr, 0};
    bool has_output{false};

    // Name-based feeds (backward compatible)
    Feed(std::string name, int idx, TF_Tensor* t)
        : tensor(t)
    {
        auto parsed = detail::parse_tensor_name(name);
        if (parsed.had_explicit_index && parsed.index != idx) {
            throw std::invalid_argument(detail::format(
                "Feed: conflicting output index for '{}' (name specifies {}, argument specifies {})",
                name, parsed.index, idx));
        }
        op_name = std::move(parsed.op);
        index = parsed.had_explicit_index ? parsed.index : idx;
    }

    Feed(std::string name, TF_Tensor* t)
        : tensor(t)
    {
        auto parsed = detail::parse_tensor_name(name);
        op_name = std::move(parsed.op);
        index = parsed.index;
    }

    Feed(std::string name, int idx, const Tensor& t)
        : Feed(std::move(name), idx, t.handle()) {}

    Feed(std::string name, const Tensor& t)
        : Feed(std::move(name), t.handle()) {}

    // Handle-based feeds (preferred for hot paths)
    Feed(TF_Output out, TF_Tensor* t)
        : op_name(), index(out.index), tensor(t), output(out), has_output(true) {}

    Feed(TF_Output out, const Tensor& t)
        : Feed(out, t.handle()) {}

    Feed(TF_Operation* op, int idx, TF_Tensor* t)
        : Feed(TF_Output{op, idx}, t) {}

    Feed(TF_Operation* op, int idx, const Tensor& t)
        : Feed(TF_Output{op, idx}, t.handle()) {}

    Feed(TF_Operation* op, TF_Tensor* t)
        : Feed(TF_Output{op, 0}, t) {}

    Feed(TF_Operation* op, const Tensor& t)
        : Feed(TF_Output{op, 0}, t.handle()) {}
};

// ============================================================================
// Buffer - RAII wrapper for TF_Buffer
// ============================================================================

class Buffer {
public:
    Buffer() : buf_(TF_NewBuffer()) {
        if (!buf_) throw std::runtime_error("TF_NewBuffer failed");
    }
    
    explicit Buffer(const void* data, std::size_t len)
        : buf_(TF_NewBufferFromString(data, len))
    {
        if (!buf_) throw std::runtime_error("TF_NewBufferFromString failed");
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
    
    [[nodiscard]] TF_Buffer* get() const noexcept { return buf_; }
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

struct Fetch {
    std::string op_name;
    int index{0};




    // Optional pre-resolved output handle (avoids string lookup in Session::Run).
    TF_Output output{nullptr, 0};
    bool has_output{false};

    // Name-based fetches (backward compatible)
    Fetch(std::string name, int idx = 0)
    {
        auto parsed = detail::parse_tensor_name(name);
        op_name = std::move(parsed.op);

        if (parsed.had_explicit_index) {
            // If the caller also passed an explicit non-zero index, treat that as a conflict.
            // (idx == 0 may be a default argument, so we cannot reliably distinguish.)
            if (idx != 0 && idx != parsed.index) {
                throw std::invalid_argument(detail::format(
                    "Fetch: conflicting output index for '{}' (name specifies {}, argument specifies {})",
                    name, parsed.index, idx));
            }
            index = parsed.index;
        } else {
            index = idx;
        }
    }

    // Handle-based fetches
    Fetch(TF_Output out)
        : op_name(), index(out.index), output(out), has_output(true) {}

    Fetch(TF_Operation* op, int idx = 0)
        : Fetch(TF_Output{op, idx}) {}
};

struct Target {
    std::string op_name;

    TF_Operation* oper{nullptr};
    bool has_oper{false};

    Target(std::string name) : op_name(std::move(name)) {}

    Target(TF_Operation* op) : oper(op), has_oper(true) {}

    Target(TF_Operation* op, std::string debug_name)
        : op_name(std::move(debug_name)), oper(op), has_oper(true) {}
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
// PartialRunHandle - RAII wrapper for partial run handle
// ============================================================================

class PartialRunHandle {
public:
    PartialRunHandle() = default;
    explicit PartialRunHandle(const char* h) : handle_(h) {}
    
    ~PartialRunHandle() { if (handle_) TF_DeletePRunHandle(handle_); }
    
    PartialRunHandle(const PartialRunHandle&) = delete;
    PartialRunHandle& operator=(const PartialRunHandle&) = delete;
    
    PartialRunHandle(PartialRunHandle&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }
    
    PartialRunHandle& operator=(PartialRunHandle&& other) noexcept {
        if (this != &other) {
            if (handle_) TF_DeletePRunHandle(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }
    
    [[nodiscard]] bool valid() const noexcept { return handle_ != nullptr; }
    [[nodiscard]] const char* get() const noexcept { return handle_; }

private:
    const char* handle_{nullptr};
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
            throw std::runtime_error("Session: cannot create session from moved-from graph");
        }

        Status st;
        session_ = TF_NewSession(graph_state_->graph, opts.get(), st.get());
        st.throw_if_error("TF_NewSession");

        graph_state_->frozen = true;
    }
    
    explicit Session(Graph& graph, TF_SessionOptions* opts)
        : graph_state_(graph.share_state())
    {
        if (!graph_state_ || !graph_state_->graph) {
            throw std::runtime_error("Session: cannot create session from moved-from graph");
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
        other.graph_state_.reset();
    }
    
    Session& operator=(Session&& other) noexcept {
        if (this != &other) {
            Cleanup();
            session_ = other.session_;
            graph_state_ = std::move(other.graph_state_);
            other.session_ = nullptr;
            other.graph_state_.reset();
        }
        return *this;
    }
    
    // ─────────────────────────────────────────────────────────────────
    // Resolve output with bounds checking
    // ─────────────────────────────────────────────────────────────────
    
    /// Resolve an operation name and index to TF_Output with bounds checking
    [[nodiscard]] TF_Output resolve_output(
        const std::string& op_name,
        int index = 0,
        std::source_location loc = std::source_location::current()) const
    {
        detail::ParsedTensorName parsed;
        try {
            parsed = detail::parse_tensor_name(op_name);
        } catch (const std::invalid_argument& e) {
            throw tf_wrap::Error::Wrapper(TF_INVALID_ARGUMENT,
                "Session::resolve_output",
                e.what(),
                op_name, -1, loc);
        }

        int used_index = parsed.had_explicit_index ? parsed.index : index;
        if (parsed.had_explicit_index && index != 0 && index != parsed.index) {
            throw tf_wrap::Error::Wrapper(TF_INVALID_ARGUMENT,
                "Session::resolve_output",
                tf_wrap::detail::format(
                    "conflicting output indices: name specifies {}, argument specifies {}",
                    parsed.index, index),
                parsed.op, parsed.index, loc);
        }

        if (!graph_state_ || !graph_state_->graph) {
            throw tf_wrap::Error::Wrapper(TF_FAILED_PRECONDITION,
                "Session::resolve_output",
                "session has no graph",
                parsed.op, used_index, loc);
        }
        
        TF_Operation* op = TF_GraphOperationByName(graph_state_->graph, parsed.op.c_str());
        if (!op) {
            throw tf_wrap::Error::Wrapper(TF_NOT_FOUND,
                "Session::resolve_output",
                "operation not found in graph",
                parsed.op, used_index, loc);
        }
        
        const int num_outputs = TF_OperationNumOutputs(op);
        if (used_index < 0 || used_index >= num_outputs) {
            throw tf_wrap::Error::Wrapper(TF_OUT_OF_RANGE,
                "Session::resolve_output",
                tf_wrap::detail::format(
                    "output index {} out of range (has {} outputs, valid indices are 0-{})",
                    used_index, num_outputs, num_outputs > 0 ? num_outputs - 1 : 0),
                parsed.op, used_index, loc);
        }
        
        return TF_Output{op, used_index};
    }
    
    
    [[nodiscard]] TF_Output validate_output_(
        TF_Output out,
        std::string_view context,
        std::source_location loc) const
    {
        if (out.oper == nullptr) {
            throw tf_wrap::Error::Wrapper(TF_INVALID_ARGUMENT,
                context,
                "TF_Output has null operation",
                {}, out.index, loc);
        }

        const int num_outputs = TF_OperationNumOutputs(out.oper);
        if (out.index < 0 || out.index >= num_outputs) {
            const char* name = TF_OperationName(out.oper);
            const std::string opn = name ? name : "";
            throw tf_wrap::Error::Wrapper(TF_OUT_OF_RANGE,
                context,
                tf_wrap::detail::format(
                    "output index {} out of range (has {} outputs, valid indices are 0-{})",
                    out.index, num_outputs, num_outputs > 0 ? num_outputs - 1 : 0),
                opn, out.index, loc);
        }

        return out;
    }

    [[nodiscard]] static Tensor allocate_by_dtype_(
        TF_DataType dtype,
        std::span<const std::int64_t> dims,
        std::source_location loc)
    {
        const std::size_t dtype_size = TF_DataTypeSize(dtype);
        if (dtype_size == 0) {
            throw tf_wrap::Error::Wrapper(TF_INVALID_ARGUMENT,
                "Session::BatchRunStacked",
                tf_wrap::detail::format(
                    "variable-length or unsupported dtype {} ({}) for stacked batching",
                    static_cast<int>(dtype), tf_wrap::dtype_name(dtype)),
                {}, -1, loc);
        }

        const std::size_t num_elems = detail::checked_product(
            dims,
            "Session::BatchRunStacked allocate_by_dtype_");
        const std::size_t bytes = detail::checked_mul(
            num_elems,
            dtype_size,
            "Session::BatchRunStacked allocate_by_dtype_ bytes");

        const int num_dims = detail::checked_int(
            dims.size(),
            "Session::BatchRunStacked allocate_by_dtype_ num_dims");
        const std::int64_t* dims_ptr = dims.empty() ? nullptr : dims.data();

        TF_Tensor* raw = TF_AllocateTensor(dtype, dims_ptr, num_dims, bytes);
        if (!raw) {
            throw tf_wrap::Error::Wrapper(TF_RESOURCE_EXHAUSTED,
                "Session::BatchRunStacked",
                tf_wrap::detail::format(
                    "TF_AllocateTensor failed for dtype {} ({}) with shape {}",
                    static_cast<int>(dtype), tf_wrap::dtype_name(dtype), detail::shape_to_string(dims)),
                {}, -1, loc);
        }

        return Tensor::FromRaw(raw);
    }

// ─────────────────────────────────────────────────────────────────
    // Run - Execute the graph
    // TF_SessionRun is thread-safe (TensorFlow's guarantee)
    // ─────────────────────────────────────────────────────────────────
    
    [[nodiscard]] 
std::vector<Tensor> Run(
    const std::vector<Feed>& feeds,
    const std::vector<Fetch>& fetches,
    const std::vector<std::string>& targets,
    TF_Buffer* run_options = nullptr,
    TF_Buffer* run_metadata = nullptr,
    std::source_location loc = std::source_location::current()) const
{
    std::vector<Target> t;
    t.reserve(targets.size());
    for (const auto& name : targets) {
        t.emplace_back(name);
    }
    return Run(feeds, fetches, std::span<const Target>(t), run_options, run_metadata, loc);
}


[[nodiscard]]
std::vector<Tensor> Run(
    const std::vector<Feed>& feeds,
    const std::vector<Fetch>& fetches,
    std::initializer_list<Target> targets,
    TF_Buffer* run_options = nullptr,
    TF_Buffer* run_metadata = nullptr,
    std::source_location loc = std::source_location::current()) const
{
    return Run(feeds, fetches,
               std::span<const Target>(targets.begin(), targets.size()),
               run_options, run_metadata, loc);
}

std::vector<Tensor> Run(
        const std::vector<Feed>& feeds,
        const std::vector<Fetch>& fetches,
        std::span<const Target> targets,
        TF_Buffer* run_options = nullptr,
        TF_Buffer* run_metadata = nullptr,
        std::source_location loc = std::source_location::current()) const
    {
        if (!session_) {
            throw std::runtime_error("Session::Run(): session is null (moved-from?)");
        }

        std::vector<TF_Output> input_ops;
        std::vector<TF_Tensor*> input_vals;
        input_ops.reserve(feeds.size());
        input_vals.reserve(feeds.size());
        
        for (const auto& f : feeds) {
            TF_Output output = f.has_output
                ? validate_output_(f.output, "Session::Run feed", loc)
                : resolve_output(f.op_name, f.index, loc);
            input_ops.push_back(output);
            input_vals.push_back(f.tensor);
        }
        
        std::vector<TF_Output> output_ops;
        output_ops.reserve(fetches.size());
        
        for (const auto& f : fetches) {
            TF_Output output = f.has_output
                ? validate_output_(f.output, "Session::Run fetch", loc)
                : resolve_output(f.op_name, f.index, loc);
            output_ops.push_back(output);
        }
        
        std::vector<TF_Operation*> target_ops;
        target_ops.reserve(targets.size());
        
        for (const auto& t : targets) {
    TF_Operation* op = t.has_oper ? t.oper
        : TF_GraphOperationByName(graph_state_->graph, t.op_name.c_str());

    if (!op) {
        throw tf_wrap::Error::Wrapper(TF_NOT_FOUND,
            "Session::Run",
            "target operation not found",
            t.op_name, -1, loc);
    }
    target_ops.push_back(op);
}
        
        std::vector<TF_Tensor*> output_vals(fetches.size(), nullptr);

        const int num_inputs  = detail::checked_int(feeds.size(), "Session::Run feeds");
        const int num_outputs = detail::checked_int(fetches.size(), "Session::Run fetches");
        const int num_targets = detail::checked_int(targets.size(), "Session::Run targets");
        
        Status st;
        TF_SessionRun(
            session_,
            run_options,
            input_ops.data(), input_vals.data(), num_inputs,
            output_ops.data(), output_vals.data(), num_outputs,
            target_ops.data(), num_targets,
            run_metadata,
            st.get());

        struct TensorDeleter {
            void operator()(TF_Tensor* t) const noexcept {
                if (t) TF_DeleteTensor(t);
            }
        };
        using RawTensorPtr = std::unique_ptr<TF_Tensor, TensorDeleter>;

        std::vector<RawTensorPtr> owned;
        owned.reserve(output_vals.size());
        for (auto* t : output_vals) {
            owned.emplace_back(t);
        }

        st.throw_if_error("TF_SessionRun", loc);

        for (std::size_t i = 0; i < owned.size(); ++i) {
            if (!owned[i]) {
                const std::string name = fetches[i].has_output && fetches[i].output.oper
                    ? (TF_OperationName(fetches[i].output.oper) ? TF_OperationName(fetches[i].output.oper) : "")
                    : fetches[i].op_name;
                const int idx = fetches[i].has_output ? fetches[i].output.index : fetches[i].index;
                throw tf_wrap::Error::Wrapper(TF_INTERNAL,
                    "TF_SessionRun",
                    "fetch returned null tensor",
                    name, idx, loc);
            }
        }

        std::vector<Tensor> results;
        results.reserve(owned.size());
        for (auto& p : owned) {
            results.push_back(Tensor::FromRaw(p.release()));
        }

        return results;
    }
    
    [[nodiscard]] std::vector<Tensor> Run(
        const std::vector<Feed>& feeds,
        const std::vector<Fetch>& fetches) const
    {
        return Run(feeds, fetches, std::span<const Target>{}, nullptr, nullptr);
    }
    
    [[nodiscard]] std::vector<Tensor> Run(
        const std::vector<Fetch>& fetches) const
    {
        return Run(std::vector<Feed>{}, fetches, std::span<const Target>{}, nullptr, nullptr);
    }
    
    /// Convenience: Run with targets but no fetch outputs
    [[nodiscard]] std::vector<Tensor> Run(
        const std::vector<Feed>& feeds,
        const std::vector<Fetch>& fetches,
        const std::vector<Target>& targets) const
    {
        return Run(feeds, fetches, std::span<const Target>(targets.data(), targets.size()));
    }
    
    /// Convenience: Fetch single output by name and index
    [[nodiscard]] Tensor Run(const std::string& op_name, int index = 0,
        std::source_location loc = std::source_location::current()) const {
        auto results = Run(std::vector<Feed>{}, std::vector<Fetch>{Fetch{op_name, index}}, std::span<const Target>{}, nullptr, nullptr, loc);
        return std::move(results[0]);
    }
    
    /// Convenience: Fetch single output by name (string literal version)
    [[nodiscard]] Tensor Run(const char* op_name, int index = 0,
        std::source_location loc = std::source_location::current()) const {
        return Run(std::string(op_name), index, loc);
    }


    // ─────────────────────────────────────────────────────────────────
    // BatchRun - Convenience helper for running many single-input inferences
    // ─────────────────────────────────────────────────────────────────

    /// Run the same (input -> output) mapping for many inputs.
    ///
    /// This is a convenience wrapper that:
    /// - resolves the input/output TF_Output once (with bounds checking),
    /// - then calls TF_SessionRun repeatedly (one per input).
    ///
    /// The returned vector has the same length as `inputs`.
    [[nodiscard]] std::vector<Tensor> BatchRun(
        const std::string& input_op,
        const std::vector<Tensor>& inputs,
        const std::string& output_op,
        int input_index = 0,
        int output_index = 0,
        std::source_location loc = std::source_location::current()) const
    {
        return BatchRun(input_op, std::span<const Tensor>(inputs.data(), inputs.size()),
                        output_op, input_index, output_index, loc);
    }

    [[nodiscard]] std::vector<Tensor> BatchRun(
        const std::string& input_op,
        std::span<const Tensor> inputs,
        const std::string& output_op,
        int input_index = 0,
        int output_index = 0,
        std::source_location loc = std::source_location::current()) const
    {
        if (!session_) {
            throw std::runtime_error("Session::BatchRun(): session is null (moved-from?)");
        }

        TF_Output in = resolve_output(input_op, input_index, loc);
        TF_Output out = resolve_output(output_op, output_index, loc);

        std::vector<Tensor> results;
        results.reserve(inputs.size());

        struct TensorDeleter {
            void operator()(TF_Tensor* t) const noexcept {
                if (t) TF_DeleteTensor(t);
            }
        };
        using RawTensorPtr = std::unique_ptr<TF_Tensor, TensorDeleter>;

        Status st;
        TF_Output input_ops[1] = {in};
        TF_Output output_ops[1] = {out};

        for (const Tensor& t : inputs) {
            if (!t.handle()) {
                throw std::invalid_argument("Session::BatchRun: input tensor is null");
            }

            TF_Tensor* input_vals[1] = {t.handle()};
            TF_Tensor* output_vals[1] = {nullptr};

            st.reset();
            TF_SessionRun(
                session_,
                nullptr,  // run_options
                input_ops, input_vals, 1,
                output_ops, output_vals, 1,
                nullptr, 0,  // targets
                nullptr,      // run_metadata
                st.get());

            RawTensorPtr owned(output_vals[0]);

            st.throw_if_error("TF_SessionRun (BatchRun)", loc);

            if (!owned) {
                throw std::runtime_error(tf_wrap::detail::format(
                    "TF_SessionRun (BatchRun): fetch '{}' returned null tensor",
                    output_op));
            }

            results.push_back(Tensor::FromRaw(owned.release()));
        }

        return results;
    }
    

    // ─────────────────────────────────────────────────────────────────
    // BatchRun (handle-based overloads)
    // ─────────────────────────────────────────────────────────────────

    [[nodiscard]] std::vector<Tensor> BatchRun(
        TF_Output input,
        const std::vector<Tensor>& inputs,
        TF_Output output,
        std::source_location loc = std::source_location::current()) const
    {
        return BatchRun(input,
                        std::span<const Tensor>(inputs.data(), inputs.size()),
                        output,
                        loc);
    }

    [[nodiscard]] std::vector<Tensor> BatchRun(
        TF_Output input,
        std::span<const Tensor> inputs,
        TF_Output output,
        std::source_location loc = std::source_location::current()) const
    {
        if (!session_) {
            throw tf_wrap::Error::Wrapper(TF_FAILED_PRECONDITION,
                "Session::BatchRun",
                "session is null (moved-from?)",
                {}, -1, loc);
        }

        TF_Output in = validate_output_(input, "Session::BatchRun input", loc);
        TF_Output out = validate_output_(output, "Session::BatchRun output", loc);

        std::vector<Tensor> results;
        results.reserve(inputs.size());

        struct TensorDeleter {
            void operator()(TF_Tensor* t) const noexcept {
                if (t) TF_DeleteTensor(t);
            }
        };
        using RawTensorPtr = std::unique_ptr<TF_Tensor, TensorDeleter>;

        Status st;
        TF_Output input_ops[1] = {in};
        TF_Output output_ops[1] = {out};

        for (const Tensor& t : inputs) {
            if (!t.handle()) {
                throw tf_wrap::Error::Wrapper(TF_INVALID_ARGUMENT,
                    "Session::BatchRun",
                    "input tensor is null",
                    {}, -1, loc);
            }

            TF_Tensor* input_vals[1] = {t.handle()};
            TF_Tensor* output_vals[1] = {nullptr};

            st.reset();
            TF_SessionRun(
                session_,
                nullptr,  // run_options
                input_ops, input_vals, 1,
                output_ops, output_vals, 1,
                nullptr, 0,  // targets
                nullptr,      // run_metadata
                st.get());

            RawTensorPtr owned(output_vals[0]);

            st.throw_if_error("TF_SessionRun (BatchRun)", loc);

            if (!owned) {
                const char* out_name = TF_OperationName(out.oper);
                throw tf_wrap::Error::Wrapper(TF_INTERNAL,
                    "TF_SessionRun (BatchRun)",
                    "fetch returned null tensor",
                    out_name ? out_name : "", out.index, loc);
            }

            results.push_back(Tensor::FromRaw(owned.release()));
        }

        return results;
    }

    // ─────────────────────────────────────────────────────────────────
    // BatchRunStacked - True batching (single TF_SessionRun)
    // ─────────────────────────────────────────────────────────────────

    /// True batching helper: stack inputs into a single batch tensor (N, ...),
    /// run one TF_SessionRun, then split outputs back into N tensors.
    ///
    /// Constraints:
    /// - inputs must be non-empty
    /// - all inputs must have the same dtype and shape
    /// - variable-length dtypes (e.g. TF_STRING) are not supported
    [[nodiscard]] std::vector<Tensor> BatchRunStacked(
        const std::string& input_op,
        const std::vector<Tensor>& inputs,
        const std::string& output_op,
        int input_index = 0,
        int output_index = 0,
        std::source_location loc = std::source_location::current()) const
    {
        TF_Output in = resolve_output(input_op, input_index, loc);
        TF_Output out = resolve_output(output_op, output_index, loc);
        return BatchRunStacked(in,
                               std::span<const Tensor>(inputs.data(), inputs.size()),
                               out,
                               loc);
    }

    [[nodiscard]] std::vector<Tensor> BatchRunStacked(
        TF_Output input,
        std::span<const Tensor> inputs,
        TF_Output output,
        std::source_location loc = std::source_location::current()) const
    {
        if (!session_) {
            throw tf_wrap::Error::Wrapper(TF_FAILED_PRECONDITION,
                "Session::BatchRunStacked",
                "session is null (moved-from?)",
                {}, -1, loc);
        }
        if (inputs.empty()) {
            return {};
        }

        TF_Output in = validate_output_(input, "Session::BatchRunStacked input", loc);
        TF_Output out = validate_output_(output, "Session::BatchRunStacked output", loc);

        const Tensor& first = inputs.front();
        if (!first.handle()) {
            throw tf_wrap::Error::Wrapper(TF_INVALID_ARGUMENT,
                "Session::BatchRunStacked",
                "first input tensor is null",
                {}, -1, loc);
        }
        const TF_DataType dtype = first.dtype();
        const std::size_t elem_size = TF_DataTypeSize(dtype);
        if (elem_size == 0) {
            throw tf_wrap::Error::Wrapper(TF_INVALID_ARGUMENT,
                "Session::BatchRunStacked",
                "variable-length dtype not supported for stacked batching",
                {}, -1, loc);
        }

        const auto base_shape = first.shape();
        std::vector<std::int64_t> base_dims(base_shape.begin(), base_shape.end());

        const std::size_t per_item_elems = detail::checked_product(base_dims, "Session::BatchRunStacked per_item");
        const std::size_t per_item_bytes = detail::checked_mul(per_item_elems, elem_size, "Session::BatchRunStacked per_item_bytes");

        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const Tensor& t = inputs[i];
            if (!t.handle()) {
                throw tf_wrap::Error::Wrapper(TF_INVALID_ARGUMENT,
                    "Session::BatchRunStacked",
                    tf_wrap::detail::format("input tensor {} is null", i),
                    {}, static_cast<int>(i), loc);
            }
            if (t.dtype() != dtype) {
                throw tf_wrap::Error::Wrapper(TF_INVALID_ARGUMENT,
                    "Session::BatchRunStacked",
                    "all inputs must have same dtype",
                    {}, static_cast<int>(i), loc);
            }
            const auto sh = t.shape();
            if (!std::equal(sh.begin(), sh.end(), base_dims.begin(), base_dims.end())) {
                throw tf_wrap::Error::Wrapper(TF_INVALID_ARGUMENT,
                    "Session::BatchRunStacked",
                    "all inputs must have same shape",
                    {}, static_cast<int>(i), loc);
            }
        }

        std::vector<std::int64_t> batched_dims;
        batched_dims.reserve(base_dims.size() + 1);
        batched_dims.push_back(static_cast<std::int64_t>(inputs.size()));
        batched_dims.insert(batched_dims.end(), base_dims.begin(), base_dims.end());

        Tensor batched = allocate_by_dtype_(dtype, batched_dims, loc);
        void* dst = TF_TensorData(batched.handle());
        if (!dst && per_item_bytes != 0) {
            throw tf_wrap::Error::Wrapper(TF_INTERNAL,
                "Session::BatchRunStacked",
                "TF_TensorData returned null for batched tensor",
                {}, -1, loc);
        }

        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const void* src = TF_TensorData(inputs[i].handle());
            if (!src && per_item_bytes != 0) {
                throw tf_wrap::Error::Wrapper(TF_INTERNAL,
                    "Session::BatchRunStacked",
                    "TF_TensorData returned null for input tensor",
                    {}, static_cast<int>(i), loc);
            }
            std::memcpy(static_cast<std::byte*>(dst) + i * per_item_bytes, src, per_item_bytes);
        }

        TF_Output input_ops[1] = {in};
        TF_Tensor* input_vals[1] = {batched.handle()};
        TF_Output output_ops[1] = {out};
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

        struct TensorDeleter {
            void operator()(TF_Tensor* t) const noexcept {
                if (t) TF_DeleteTensor(t);
            }
        };
        using RawTensorPtr = std::unique_ptr<TF_Tensor, TensorDeleter>;
        RawTensorPtr owned(output_vals[0]);

        st.throw_if_error("TF_SessionRun (BatchRunStacked)", loc);
        if (!owned) {
            throw tf_wrap::Error::Wrapper(TF_INTERNAL,
                "TF_SessionRun (BatchRunStacked)",
                "returned null output tensor",
                {}, -1, loc);
        }

        Tensor out_batched = Tensor::FromRaw(owned.release());
        if (out_batched.dtype() != dtype) {
            // still allow, but splitting uses element size from returned dtype
        }

        const TF_DataType out_dtype = out_batched.dtype();
        const std::size_t out_elem_size = TF_DataTypeSize(out_dtype);
        if (out_elem_size == 0) {
            throw tf_wrap::Error::Wrapper(TF_INVALID_ARGUMENT,
                "Session::BatchRunStacked",
                "variable-length output dtype not supported for splitting",
                {}, -1, loc);
        }

        const auto out_shape = out_batched.shape();
        if (out_shape.size() < 1 || out_shape[0] != static_cast<std::int64_t>(inputs.size())) {
            throw tf_wrap::Error::Wrapper(TF_INVALID_ARGUMENT,
                "Session::BatchRunStacked",
                "batched output first dimension does not match batch size",
                {}, -1, loc);
        }

        std::vector<std::int64_t> out_item_dims(out_shape.begin() + 1, out_shape.end());
        const std::size_t out_item_elems = detail::checked_product(out_item_dims, "Session::BatchRunStacked out_item");
        const std::size_t out_item_bytes = detail::checked_mul(out_item_elems, out_elem_size, "Session::BatchRunStacked out_item_bytes");

        const void* out_src = TF_TensorData(out_batched.handle());
        if (!out_src && out_item_bytes != 0) {
            throw tf_wrap::Error::Wrapper(TF_INTERNAL,
                "Session::BatchRunStacked",
                "TF_TensorData returned null for output tensor",
                {}, -1, loc);
        }

        std::vector<Tensor> split;
        split.reserve(inputs.size());
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            Tensor item = allocate_by_dtype_(out_dtype, out_item_dims, loc);
            void* item_dst = TF_TensorData(item.handle());
            if (!item_dst && out_item_bytes != 0) {
                throw tf_wrap::Error::Wrapper(TF_INTERNAL,
                    "Session::BatchRunStacked",
                    "TF_TensorData returned null for split tensor",
                    {}, static_cast<int>(i), loc);
            }
            std::memcpy(item_dst,
                        static_cast<const std::byte*>(out_src) + i * out_item_bytes,
                        out_item_bytes);
            split.push_back(std::move(item));
        }

        return split;
    }


    // ─────────────────────────────────────────────────────────────────
    // Device enumeration
    // ─────────────────────────────────────────────────────────────────
    
    [[nodiscard]] DeviceList ListDevices() const {
        if (!session_) {
            throw std::runtime_error("Session::ListDevices(): session is null");
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
    // Partial runs
    // ─────────────────────────────────────────────────────────────────
    
    
[[nodiscard]] PartialRunHandle PartialRunSetup(
    const std::vector<Fetch>& inputs,
    const std::vector<Fetch>& outputs,
    const std::vector<std::string>& targets,
    std::source_location loc = std::source_location::current()) const
{
    std::vector<Target> t;
    t.reserve(targets.size());
    for (const auto& name : targets) {
        t.emplace_back(name);
    }
    return PartialRunSetup(inputs, outputs, t, loc);
}

[[nodiscard]] PartialRunHandle PartialRunSetup(
        const std::vector<Fetch>& inputs,
        const std::vector<Fetch>& outputs,
        const std::vector<Target>& targets = {},
        std::source_location loc = std::source_location::current()) const
    {
        if (!session_) {
            throw std::runtime_error("Session::PartialRunSetup(): session is null");
        }
        
        std::vector<TF_Output> input_ops;
        input_ops.reserve(inputs.size());
        for (const auto& f : inputs) {
            TF_Output out = f.has_output
                ? validate_output_(f.output, "Session::PartialRunSetup input", loc)
                : resolve_output(f.op_name, f.index, loc);
            input_ops.push_back(out);
        }
        
        std::vector<TF_Output> output_ops;
        output_ops.reserve(outputs.size());
        for (const auto& f : outputs) {
            TF_Output out = f.has_output
                ? validate_output_(f.output, "Session::PartialRunSetup output", loc)
                : resolve_output(f.op_name, f.index, loc);
            output_ops.push_back(out);
        }
        
        std::vector<TF_Operation*> target_ops;
        target_ops.reserve(targets.size());
        for (const auto& t : targets) {
            TF_Operation* op = t.has_oper ? t.oper
                : TF_GraphOperationByName(graph_state_->graph, t.op_name.c_str());
            if (!op) {
                throw tf_wrap::Error::Wrapper(TF_NOT_FOUND,
                    "Session::PartialRunSetup",
                    "target operation not found",
                    t.op_name, -1, loc);
            }
            target_ops.push_back(op);
        }
        
        const char* handle = nullptr;
        Status st;
        
        TF_SessionPRunSetup(
            session_,
            input_ops.data(), detail::checked_int(inputs.size(), "Session::PartialRunSetup inputs"),
            output_ops.data(), detail::checked_int(outputs.size(), "Session::PartialRunSetup outputs"),
            target_ops.data(), detail::checked_int(targets.size(), "Session::PartialRunSetup targets"),
            &handle,
            st.get());
        
        st.throw_if_error("TF_SessionPRunSetup", loc);
        return PartialRunHandle(handle);
    }
    
    [[nodiscard]] 
std::vector<Tensor> PartialRun(
    const PartialRunHandle& handle,
    const std::vector<Feed>& feeds,
    const std::vector<Fetch>& fetches,
    const std::vector<std::string>& targets,
    std::source_location loc = std::source_location::current()) const
{
    std::vector<Target> t;
    t.reserve(targets.size());
    for (const auto& name : targets) {
        t.emplace_back(name);
    }
    return PartialRun(handle, feeds, fetches, t, loc);
}

std::vector<Tensor> PartialRun(
        const PartialRunHandle& handle,
        const std::vector<Feed>& feeds,
        const std::vector<Fetch>& fetches,
        const std::vector<Target>& targets = {},
        std::source_location loc = std::source_location::current()) const
    {
        if (!session_) {
            throw std::runtime_error("Session::PartialRun(): session is null");
        }
        if (!handle.valid()) {
            throw std::runtime_error("Session::PartialRun(): invalid handle");
        }
        
        std::vector<TF_Output> input_ops;
        std::vector<TF_Tensor*> input_vals;
        input_ops.reserve(feeds.size());
        input_vals.reserve(feeds.size());
        
        for (const auto& f : feeds) {
            TF_Output out = f.has_output
                ? validate_output_(f.output, "Session::PartialRun feed", loc)
                : resolve_output(f.op_name, f.index, loc);
            input_ops.push_back(out);
            input_vals.push_back(f.tensor);
        }
        
        std::vector<TF_Output> output_ops;
        output_ops.reserve(fetches.size());
        for (const auto& f : fetches) {
            TF_Output out = f.has_output
                ? validate_output_(f.output, "Session::PartialRun fetch", loc)
                : resolve_output(f.op_name, f.index, loc);
            output_ops.push_back(out);
        }
        
        std::vector<TF_Operation*> target_ops;
        target_ops.reserve(targets.size());
        for (const auto& t : targets) {
            TF_Operation* op = t.has_oper ? t.oper
                : TF_GraphOperationByName(graph_state_->graph, t.op_name.c_str());
            if (!op) {
                throw tf_wrap::Error::Wrapper(TF_NOT_FOUND,
                    "Session::PartialRun",
                    "target operation not found",
                    t.op_name, -1, loc);
            }
            target_ops.push_back(op);
        }
        
        std::vector<TF_Tensor*> output_vals(fetches.size(), nullptr);

        const int num_inputs  = detail::checked_int(feeds.size(), "Session::PartialRun feeds");
        const int num_outputs = detail::checked_int(fetches.size(), "Session::PartialRun fetches");
        const int num_targets = detail::checked_int(targets.size(), "Session::PartialRun targets");
        
        Status st;
        TF_SessionPRun(
            session_,
            handle.get(),
            input_ops.data(), input_vals.data(), num_inputs,
            output_ops.data(), output_vals.data(), num_outputs,
            target_ops.data(), num_targets,
            st.get());

        struct TensorDeleter {
            void operator()(TF_Tensor* t) const noexcept {
                if (t) TF_DeleteTensor(t);
            }
        };
        using RawTensorPtr = std::unique_ptr<TF_Tensor, TensorDeleter>;

        std::vector<RawTensorPtr> owned;
        owned.reserve(output_vals.size());
        for (auto* t : output_vals) {
            owned.emplace_back(t);
        }

        st.throw_if_error("TF_SessionPRun", loc);

        for (std::size_t i = 0; i < owned.size(); ++i) {
            if (!owned[i]) {
                const std::string name = fetches[i].has_output && fetches[i].output.oper
                    ? (TF_OperationName(fetches[i].output.oper) ? TF_OperationName(fetches[i].output.oper) : "")
                    : fetches[i].op_name;
                const int idx = fetches[i].has_output ? fetches[i].output.index : fetches[i].index;
                throw tf_wrap::Error::Wrapper(TF_INTERNAL,
                    "TF_SessionPRun",
                    "fetch returned null tensor",
                    name, idx, loc);
            }
        }

        std::vector<Tensor> results;
        results.reserve(owned.size());
        for (auto& p : owned) {
            results.push_back(Tensor::FromRaw(p.release()));
        }

        return results;
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
            opts.get(),
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

// Backward compatibility alias

} // namespace tf_wrap
