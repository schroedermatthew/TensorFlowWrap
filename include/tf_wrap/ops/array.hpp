// tf_wrap/ops/array.hpp
// Array operation wrappers (generated)

#pragma once

#include "tf_wrap/ops/common.hpp"

namespace tf_wrap {
namespace ops {

// ============================================================================
// Array Operations
// ============================================================================

/// Constant tensor

[[nodiscard]] inline OpResult Const(
    Graph& graph,
    std::string_view name,
    TF_Tensor* value,
    TF_DataType dtype) {
    return OpResult(
        graph.NewOperation("Const", std::string(name))
        .SetAttrTensor("value", value)
        .SetAttrType("dtype", dtype)
        .Finish());
}

/// Placeholder for feeding data

[[nodiscard]] inline OpResult Placeholder(
    Graph& graph,
    std::string_view name,
    TF_DataType dtype,
    std::span<const int64_t> shape) {
    return OpResult(
        graph.NewOperation("Placeholder", std::string(name))
        .SetAttrType("dtype", dtype)
        .SetAttrShape("shape", shape)
        .Finish());
}

/// Identity function

[[nodiscard]] inline OpResult Identity(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Identity", std::string(name))
        .AddInput(input)
        .SetAttrType("T", T)
        .Finish());
}

/// Identity for multiple tensors

[[nodiscard]] inline OpResult IdentityN(
    Graph& graph,
    std::string_view name,
    std::span<const TF_Output> input) {
    return OpResult(graph.NewOperation("IdentityN", std::string(name)).AddInputList(input).Finish());
}

/// Reshape tensor

[[nodiscard]] inline OpResult Reshape(
    Graph& graph,
    std::string_view name,
    TF_Output tensor,
    TF_Output shape,
    TF_DataType T,
    TF_DataType Tshape) {
    return OpResult(
        graph.NewOperation("Reshape", std::string(name))
        .AddInput(tensor)
        .AddInput(shape)
        .SetAttrType("T", T)
        .SetAttrType("Tshape", Tshape)
        .Finish());
}

/// Remove size-1 dimensions

[[nodiscard]] inline OpResult Squeeze(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Squeeze", std::string(name))
        .AddInput(input)
        .SetAttrType("T", T)
        .Finish());
}

/// Insert dimension of size 1

[[nodiscard]] inline OpResult ExpandDims(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output axis,
    TF_DataType T,
    TF_DataType Tdim) {
    return OpResult(
        graph.NewOperation("ExpandDims", std::string(name))
        .AddInput(input)
        .AddInput(axis)
        .SetAttrType("T", T)
        .SetAttrType("Tdim", Tdim)
        .Finish());
}

/// Concatenate tensors

[[nodiscard]] inline OpResult Concat(
    Graph& graph,
    std::string_view name,
    TF_Output concat_dim,
    std::span<const TF_Output> values,
    TF_DataType T,
    int64_t N) {
    return OpResult(
        graph.NewOperation("Concat", std::string(name))
        .AddInput(concat_dim)
        .AddInputList(values)
        .SetAttrType("T", T)
        .SetAttrInt("N", N)
        .Finish());
}

/// Concatenate tensors V2

[[nodiscard]] inline OpResult ConcatV2(
    Graph& graph,
    std::string_view name,
    std::span<const TF_Output> values,
    TF_Output axis,
    TF_DataType T,
    int64_t N,
    TF_DataType Tidx) {
    return OpResult(
        graph.NewOperation("ConcatV2", std::string(name))
        .AddInputList(values)
        .AddInput(axis)
        .SetAttrType("T", T)
        .SetAttrInt("N", N)
        .SetAttrType("Tidx", Tidx)
        .Finish());
}

/// Split tensor into subtensors

[[nodiscard]] inline OpResult Split(
    Graph& graph,
    std::string_view name,
    TF_Output split_dim,
    TF_Output value,
    TF_DataType T,
    int64_t num_split) {
    return OpResult(
        graph.NewOperation("Split", std::string(name))
        .AddInput(split_dim)
        .AddInput(value)
        .SetAttrType("T", T)
        .SetAttrInt("num_split", num_split)
        .Finish());
}

/// Split tensor with variable sizes

[[nodiscard]] inline OpResult SplitV(
    Graph& graph,
    std::string_view name,
    TF_Output value,
    TF_Output size_splits,
    TF_Output split_dim,
    TF_DataType T,
    int64_t num_split,
    TF_DataType Tlen) {
    return OpResult(
        graph.NewOperation("SplitV", std::string(name))
        .AddInput(value)
        .AddInput(size_splits)
        .AddInput(split_dim)
        .SetAttrType("T", T)
        .SetAttrInt("num_split", num_split)
        .SetAttrType("Tlen", Tlen)
        .Finish());
}

/// Slice from tensor

[[nodiscard]] inline OpResult Slice(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output begin,
    TF_Output size,
    TF_DataType T,
    TF_DataType Index) {
    return OpResult(
        graph.NewOperation("Slice", std::string(name))
        .AddInput(input)
        .AddInput(begin)
        .AddInput(size)
        .SetAttrType("T", T)
        .SetAttrType("Index", Index)
        .Finish());
}

/// Strided slice from tensor

[[nodiscard]] inline OpResult StridedSlice(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output begin,
    TF_Output end,
    TF_Output strides,
    TF_DataType T,
    TF_DataType Index) {
    return OpResult(
        graph.NewOperation("StridedSlice", std::string(name))
        .AddInput(input)
        .AddInput(begin)
        .AddInput(end)
        .AddInput(strides)
        .SetAttrType("T", T)
        .SetAttrType("Index", Index)
        .Finish());
}

/// Gather slices from tensor

[[nodiscard]] inline OpResult Gather(
    Graph& graph,
    std::string_view name,
    TF_Output params,
    TF_Output indices,
    TF_DataType Tparams,
    TF_DataType Tindices) {
    return OpResult(
        graph.NewOperation("Gather", std::string(name))
        .AddInput(params)
        .AddInput(indices)
        .SetAttrType("Tparams", Tparams)
        .SetAttrType("Tindices", Tindices)
        .Finish());
}

/// Gather slices with axis

[[nodiscard]] inline OpResult GatherV2(
    Graph& graph,
    std::string_view name,
    TF_Output params,
    TF_Output indices,
    TF_Output axis,
    TF_DataType Tparams,
    TF_DataType Tindices,
    TF_DataType Taxis) {
    return OpResult(
        graph.NewOperation("GatherV2", std::string(name))
        .AddInput(params)
        .AddInput(indices)
        .AddInput(axis)
        .SetAttrType("Tparams", Tparams)
        .SetAttrType("Tindices", Tindices)
        .SetAttrType("Taxis", Taxis)
        .Finish());
}

/// Gather slices with N-dimensional indices

[[nodiscard]] inline OpResult GatherNd(
    Graph& graph,
    std::string_view name,
    TF_Output params,
    TF_Output indices,
    TF_DataType Tparams,
    TF_DataType Tindices) {
    return OpResult(
        graph.NewOperation("GatherNd", std::string(name))
        .AddInput(params)
        .AddInput(indices)
        .SetAttrType("Tparams", Tparams)
        .SetAttrType("Tindices", Tindices)
        .Finish());
}

/// Scatter updates into tensor

[[nodiscard]] inline OpResult ScatterNd(
    Graph& graph,
    std::string_view name,
    TF_Output indices,
    TF_Output updates,
    TF_Output shape,
    TF_DataType T,
    TF_DataType Tindices) {
    return OpResult(
        graph.NewOperation("ScatterNd", std::string(name))
        .AddInput(indices)
        .AddInput(updates)
        .AddInput(shape)
        .SetAttrType("T", T)
        .SetAttrType("Tindices", Tindices)
        .Finish());
}

/// Tile tensor

[[nodiscard]] inline OpResult Tile(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output multiples,
    TF_DataType T,
    TF_DataType Tmultiples) {
    return OpResult(
        graph.NewOperation("Tile", std::string(name))
        .AddInput(input)
        .AddInput(multiples)
        .SetAttrType("T", T)
        .SetAttrType("Tmultiples", Tmultiples)
        .Finish());
}

/// Pad tensor with zeros

[[nodiscard]] inline OpResult Pad(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output paddings,
    TF_DataType T,
    TF_DataType Tpaddings) {
    return OpResult(
        graph.NewOperation("Pad", std::string(name))
        .AddInput(input)
        .AddInput(paddings)
        .SetAttrType("T", T)
        .SetAttrType("Tpaddings", Tpaddings)
        .Finish());
}

/// Pad tensor with constant value

[[nodiscard]] inline OpResult PadV2(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output paddings,
    TF_Output constant_values,
    TF_DataType T,
    TF_DataType Tpaddings) {
    return OpResult(
        graph.NewOperation("PadV2", std::string(name))
        .AddInput(input)
        .AddInput(paddings)
        .AddInput(constant_values)
        .SetAttrType("T", T)
        .SetAttrType("Tpaddings", Tpaddings)
        .Finish());
}

/// Pad tensor with mirrored values

[[nodiscard]] inline OpResult MirrorPad(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output paddings,
    TF_DataType T,
    TF_DataType Tpaddings,
    std::string_view mode) {
    return OpResult(
        graph.NewOperation("MirrorPad", std::string(name))
        .AddInput(input)
        .AddInput(paddings)
        .SetAttrType("T", T)
        .SetAttrType("Tpaddings", Tpaddings)
        .SetAttrString("mode", mode)
        .Finish());
}

/// Reverse tensor along axes

[[nodiscard]] inline OpResult ReverseV2(
    Graph& graph,
    std::string_view name,
    TF_Output tensor,
    TF_Output axis,
    TF_DataType T,
    TF_DataType Tidx) {
    return OpResult(
        graph.NewOperation("ReverseV2", std::string(name))
        .AddInput(tensor)
        .AddInput(axis)
        .SetAttrType("T", T)
        .SetAttrType("Tidx", Tidx)
        .Finish());
}

/// Stack tensors along axis (pack)

[[nodiscard]] inline OpResult Pack(
    Graph& graph,
    std::string_view name,
    std::span<const TF_Output> values,
    TF_DataType T,
    int64_t N) {
    return OpResult(
        graph.NewOperation("Pack", std::string(name))
        .AddInputList(values)
        .SetAttrType("T", T)
        .SetAttrInt("N", N)
        .Finish());
}

/// Unstack tensor along axis (unpack)

[[nodiscard]] inline OpResult Unpack(
    Graph& graph,
    std::string_view name,
    TF_Output value,
    TF_DataType T,
    int64_t num) {
    return OpResult(
        graph.NewOperation("Unpack", std::string(name))
        .AddInput(value)
        .SetAttrType("T", T)
        .SetAttrInt("num", num)
        .Finish());
}

/// Get tensor shape

[[nodiscard]] inline OpResult Shape(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_DataType T,
    TF_DataType out_type) {
    return OpResult(
        graph.NewOperation("Shape", std::string(name))
        .AddInput(input)
        .SetAttrType("T", T)
        .SetAttrType("out_type", out_type)
        .Finish());
}

/// Get shapes of multiple tensors

[[nodiscard]] inline OpResult ShapeN(
    Graph& graph,
    std::string_view name,
    std::span<const TF_Output> input,
    TF_DataType T,
    int64_t N,
    TF_DataType out_type) {
    return OpResult(
        graph.NewOperation("ShapeN", std::string(name))
        .AddInputList(input)
        .SetAttrType("T", T)
        .SetAttrInt("N", N)
        .SetAttrType("out_type", out_type)
        .Finish());
}

/// Get tensor rank

[[nodiscard]] inline OpResult Rank(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Rank", std::string(name))
        .AddInput(input)
        .SetAttrType("T", T)
        .Finish());
}

/// Get number of elements

[[nodiscard]] inline OpResult Size(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_DataType T,
    TF_DataType out_type) {
    return OpResult(
        graph.NewOperation("Size", std::string(name))
        .AddInput(input)
        .SetAttrType("T", T)
        .SetAttrType("out_type", out_type)
        .Finish());
}

/// Fill tensor with scalar value

[[nodiscard]] inline OpResult Fill(
    Graph& graph,
    std::string_view name,
    TF_Output dims,
    TF_Output value,
    TF_DataType T,
    TF_DataType index_type) {
    return OpResult(
        graph.NewOperation("Fill", std::string(name))
        .AddInput(dims)
        .AddInput(value)
        .SetAttrType("T", T)
        .SetAttrType("index_type", index_type)
        .Finish());
}

// NOTE: "Zeros" op does not exist in TensorFlow C API.
// To create a tensor of zeros, use one of these alternatives:
//
// 1. Fill() with a zero constant:
//    auto zero = Const(graph, "zero", zero_tensor.handle(), TF_FLOAT);
//    auto zeros = Fill(graph, "zeros", shape, zero.output(), TF_FLOAT, TF_INT32);
//
// 2. ZerosLike() to match another tensor's shape:
//    auto zeros = ZerosLike(graph, "zeros", other_tensor_output, TF_FLOAT);

/// Create tensor of zeros with same shape

[[nodiscard]] inline OpResult ZerosLike(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("ZerosLike", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Create tensor of ones with same shape

[[nodiscard]] inline OpResult OnesLike(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("OnesLike", std::string(name))
        .AddInput(x)
        .SetAttrType("T", T)
        .Finish());
}

/// Create range [start, limit) with delta step

[[nodiscard]] inline OpResult Range(
    Graph& graph,
    std::string_view name,
    TF_Output start,
    TF_Output limit,
    TF_Output delta,
    TF_DataType Tidx) {
    return OpResult(
        graph.NewOperation("Range", std::string(name))
        .AddInput(start)
        .AddInput(limit)
        .AddInput(delta)
        .SetAttrType("Tidx", Tidx)
        .Finish());
}

/// Create linearly spaced values

[[nodiscard]] inline OpResult LinSpace(
    Graph& graph,
    std::string_view name,
    TF_Output start,
    TF_Output stop,
    TF_Output num,
    TF_DataType T,
    TF_DataType Tidx) {
    return OpResult(
        graph.NewOperation("LinSpace", std::string(name))
        .AddInput(start)
        .AddInput(stop)
        .AddInput(num)
        .SetAttrType("T", T)
        .SetAttrType("Tidx", Tidx)
        .Finish());
}

/// Broadcast to shape

[[nodiscard]] inline OpResult BroadcastTo(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output shape,
    TF_DataType T,
    TF_DataType Tidx) {
    return OpResult(
        graph.NewOperation("BroadcastTo", std::string(name))
        .AddInput(input)
        .AddInput(shape)
        .SetAttrType("T", T)
        .SetAttrType("Tidx", Tidx)
        .Finish());
}

/// Returns indices of true elements

[[nodiscard]] inline OpResult Where(
    Graph& graph,
    std::string_view name,
    TF_Output condition,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Where", std::string(name))
        .AddInput(condition)
        .SetAttrType("T", T)
        .Finish());
}

/// Select elements based on condition

[[nodiscard]] inline OpResult SelectV2(
    Graph& graph,
    std::string_view name,
    TF_Output condition,
    TF_Output t,
    TF_Output e,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("SelectV2", std::string(name))
        .AddInput(condition)
        .AddInput(t)
        .AddInput(e)
        .SetAttrType("T", T)
        .Finish());
}



} // namespace ops
} // namespace tf_wrap
