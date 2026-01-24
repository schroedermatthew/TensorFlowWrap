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
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include <tensorflow/c/c_api.h>
}

#include "tf_wrap/format.hpp"
#include "tf_wrap/graph.hpp"
#include "tf_wrap/status.hpp"
#include "tf_wrap/tensor.hpp"

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
    TF_Tensor* tensor;
    
    Feed(std::string name, int idx, TF_Tensor* t)
        : op_name(std::move(name)), index(idx), tensor(t) {}
    
    Feed(std::string name, TF_Tensor* t)
        : op_name(std::move(name)), index(0), tensor(t) {}
    
    Feed(std::string name, int idx, const Tensor& t)
        : op_name(std::move(name)), index(idx), tensor(t.handle()) {}
    
    Feed(std::string name, const Tensor& t)
        : op_name(std::move(name)), index(0), tensor(t.handle()) {}
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
    
    Fetch(std::string name, int idx = 0)
        : op_name(std::move(name)), index(idx) {}
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
    // Run - Execute the graph
    // TF_SessionRun is thread-safe (TensorFlow's guarantee)
    // ─────────────────────────────────────────────────────────────────
    
    [[nodiscard]] std::vector<Tensor> Run(
        const std::vector<Feed>& feeds,
        const std::vector<Fetch>& fetches,
        const std::vector<std::string>& targets,
        TF_Buffer* run_options,
        TF_Buffer* run_metadata) const
    {
        if (!session_) {
            throw std::runtime_error("Session::Run(): session is null (moved-from?)");
        }

        std::vector<TF_Output> input_ops;
        std::vector<TF_Tensor*> input_vals;
        input_ops.reserve(feeds.size());
        input_vals.reserve(feeds.size());
        
        for (const auto& f : feeds) {
            TF_Operation* op = TF_GraphOperationByName(graph_state_->graph, f.op_name.c_str());
            if (!op) {
                throw std::runtime_error(tf_wrap::detail::format(
                    "Feed operation '{}' not found", f.op_name));
            }
            input_ops.push_back(TF_Output{op, f.index});
            input_vals.push_back(f.tensor);
        }
        
        std::vector<TF_Output> output_ops;
        output_ops.reserve(fetches.size());
        
        for (const auto& f : fetches) {
            TF_Operation* op = TF_GraphOperationByName(graph_state_->graph, f.op_name.c_str());
            if (!op) {
                throw std::runtime_error(tf_wrap::detail::format(
                    "Fetch operation '{}' not found", f.op_name));
            }
            output_ops.push_back(TF_Output{op, f.index});
        }
        
        std::vector<TF_Operation*> target_ops;
        target_ops.reserve(targets.size());
        
        for (const auto& t : targets) {
            TF_Operation* op = TF_GraphOperationByName(graph_state_->graph, t.c_str());
            if (!op) {
                throw std::runtime_error(tf_wrap::detail::format(
                    "Target operation '{}' not found", t));
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
<<<<<<< HEAD
        };
        using RawTensorPtr = std::unique_ptr<TF_Tensor, TensorDeleter>;

        std::vector<RawTensorPtr> owned;
        owned.reserve(output_vals.size());
=======
            st.throw_if_error("TF_SessionRun");
        }

        for (std::size_t i = 0; i < output_vals.size(); ++i) {
            if (output_vals[i] == nullptr) {
                for (auto* t : output_vals) {
                    if (t) TF_DeleteTensor(t);
                }
                throw std::runtime_error(tf_wrap::detail::format(
                    "TF_SessionRun: fetch '{}' returned null tensor",
                    fetches[i].op_name));
            }
        }
        
        std::vector<Tensor> results;
        results.reserve(fetches.size());
>>>>>>> 1182454810694b01398ec7aa912d4e2a28324fb8
        for (auto* t : output_vals) {
            owned.emplace_back(t);
        }

        st.throw_if_error("TF_SessionRun");

        for (std::size_t i = 0; i < owned.size(); ++i) {
            if (!owned[i]) {
                throw std::runtime_error(tf_wrap::detail::format(
                    "TF_SessionRun: fetch '{}' returned null tensor",
                    fetches[i].op_name));
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
        return Run(feeds, fetches, {}, nullptr, nullptr);
    }
    
    [[nodiscard]] std::vector<Tensor> Run(
        const std::vector<Fetch>& fetches) const
    {
        return Run({}, fetches, {}, nullptr, nullptr);
    }
    
    /// Convenience: Run with targets but no fetch outputs
    [[nodiscard]] std::vector<Tensor> Run(
        const std::vector<Feed>& feeds,
        const std::vector<Fetch>& fetches,
        const std::vector<std::string>& targets) const
    {
        return Run(feeds, fetches, targets, nullptr, nullptr);
    }
    
    /// Convenience: Fetch single output by name and index
    [[nodiscard]] Tensor Run(const std::string& op_name, int index = 0) const {
        auto results = Run({}, {Fetch{op_name, index}});
        return std::move(results[0]);
    }
    
    /// Convenience: Fetch single output by name (string literal version)
    [[nodiscard]] Tensor Run(const char* op_name, int index = 0) const {
        return Run(std::string(op_name), index);
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
        const std::vector<std::string>& targets = {}) const
    {
        if (!session_) {
            throw std::runtime_error("Session::PartialRunSetup(): session is null");
        }
        
        std::vector<TF_Output> input_ops;
        input_ops.reserve(inputs.size());
        for (const auto& f : inputs) {
            TF_Operation* op = TF_GraphOperationByName(graph_state_->graph, f.op_name.c_str());
            if (!op) {
                throw std::runtime_error(tf_wrap::detail::format(
                    "PartialRunSetup: input '{}' not found", f.op_name));
            }
            input_ops.push_back(TF_Output{op, f.index});
        }
        
        std::vector<TF_Output> output_ops;
        output_ops.reserve(outputs.size());
        for (const auto& f : outputs) {
            TF_Operation* op = TF_GraphOperationByName(graph_state_->graph, f.op_name.c_str());
            if (!op) {
                throw std::runtime_error(tf_wrap::detail::format(
                    "PartialRunSetup: output '{}' not found", f.op_name));
            }
            output_ops.push_back(TF_Output{op, f.index});
        }
        
        std::vector<TF_Operation*> target_ops;
        target_ops.reserve(targets.size());
        for (const auto& t : targets) {
            TF_Operation* op = TF_GraphOperationByName(graph_state_->graph, t.c_str());
            if (!op) {
                throw std::runtime_error(tf_wrap::detail::format(
                    "PartialRunSetup: target '{}' not found", t));
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
        
        st.throw_if_error("TF_SessionPRunSetup");
        return PartialRunHandle(handle);
    }
    
    [[nodiscard]] std::vector<Tensor> PartialRun(
        const PartialRunHandle& handle,
        const std::vector<Feed>& feeds,
        const std::vector<Fetch>& fetches,
        const std::vector<std::string>& targets = {}) const
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
            TF_Operation* op = TF_GraphOperationByName(graph_state_->graph, f.op_name.c_str());
            if (!op) {
                throw std::runtime_error(tf_wrap::detail::format(
                    "PartialRun: feed '{}' not found", f.op_name));
            }
            input_ops.push_back(TF_Output{op, f.index});
            input_vals.push_back(f.tensor);
        }
        
        std::vector<TF_Output> output_ops;
        output_ops.reserve(fetches.size());
        for (const auto& f : fetches) {
            TF_Operation* op = TF_GraphOperationByName(graph_state_->graph, f.op_name.c_str());
            if (!op) {
                throw std::runtime_error(tf_wrap::detail::format(
                    "PartialRun: fetch '{}' not found", f.op_name));
            }
            output_ops.push_back(TF_Output{op, f.index});
        }
        
        std::vector<TF_Operation*> target_ops;
        target_ops.reserve(targets.size());
        for (const auto& t : targets) {
            TF_Operation* op = TF_GraphOperationByName(graph_state_->graph, t.c_str());
            if (!op) {
                throw std::runtime_error(tf_wrap::detail::format(
                    "PartialRun: target '{}' not found", t));
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
<<<<<<< HEAD
        };
        using RawTensorPtr = std::unique_ptr<TF_Tensor, TensorDeleter>;

        std::vector<RawTensorPtr> owned;
        owned.reserve(output_vals.size());
=======
            st.throw_if_error("TF_SessionPRun");
        }

        for (std::size_t i = 0; i < output_vals.size(); ++i) {
            if (output_vals[i] == nullptr) {
                for (auto* t : output_vals) {
                    if (t) TF_DeleteTensor(t);
                }
                throw std::runtime_error(tf_wrap::detail::format(
                    "TF_SessionPRun: fetch '{}' returned null tensor",
                    fetches[i].op_name));
            }
        }
        
        std::vector<Tensor> results;
        results.reserve(fetches.size());
>>>>>>> 1182454810694b01398ec7aa912d4e2a28324fb8
        for (auto* t : output_vals) {
            owned.emplace_back(t);
        }

        st.throw_if_error("TF_SessionPRun");

        for (std::size_t i = 0; i < owned.size(); ++i) {
            if (!owned[i]) {
                throw std::runtime_error(tf_wrap::detail::format(
                    "TF_SessionPRun: fetch '{}' returned null tensor",
                    fetches[i].op_name));
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
            TF_CloseSession(session_, st);
            TF_DeleteSession(session_, st);
            TF_DeleteStatus(st);
        }
        
        session_ = nullptr;
        graph_state_.reset();
    }
};

// Backward compatibility alias

} // namespace tf_wrap
