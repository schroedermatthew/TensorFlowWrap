// tf_wrap/ops/nn.hpp
// Neural Network operation wrappers (generated)

#pragma once

#include "tf_wrap/ops/common.hpp"

namespace tf_wrap {
namespace ops {

// ============================================================================
// Neural Network Operations
// ============================================================================

/// ReLU activation: max(0, x)

[[nodiscard]] inline OpResult Relu(
    Graph& graph,
    std::string_view name,
    TF_Output features,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Relu", std::string(name))
        .AddInput(features)
        .SetAttrType("T", T)
        .Finish());
}

/// ReLU6 activation: min(max(0, x), 6)

[[nodiscard]] inline OpResult Relu6(
    Graph& graph,
    std::string_view name,
    TF_Output features,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Relu6", std::string(name))
        .AddInput(features)
        .SetAttrType("T", T)
        .Finish());
}

/// Leaky ReLU activation

[[nodiscard]] inline OpResult LeakyRelu(
    Graph& graph,
    std::string_view name,
    TF_Output features,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("LeakyRelu", std::string(name))
        .AddInput(features)
        .SetAttrType("T", T)
        .Finish());
}

/// ELU activation

[[nodiscard]] inline OpResult Elu(
    Graph& graph,
    std::string_view name,
    TF_Output features,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Elu", std::string(name))
        .AddInput(features)
        .SetAttrType("T", T)
        .Finish());
}

/// SELU activation

[[nodiscard]] inline OpResult Selu(
    Graph& graph,
    std::string_view name,
    TF_Output features,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Selu", std::string(name))
        .AddInput(features)
        .SetAttrType("T", T)
        .Finish());
}

/// Softmax activation

[[nodiscard]] inline OpResult Softmax(
    Graph& graph,
    std::string_view name,
    TF_Output logits,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Softmax", std::string(name))
        .AddInput(logits)
        .SetAttrType("T", T)
        .Finish());
}

/// Log-softmax activation

[[nodiscard]] inline OpResult LogSoftmax(
    Graph& graph,
    std::string_view name,
    TF_Output logits,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("LogSoftmax", std::string(name))
        .AddInput(logits)
        .SetAttrType("T", T)
        .Finish());
}

/// Softplus activation: ln(1 + e^x)

[[nodiscard]] inline OpResult Softplus(
    Graph& graph,
    std::string_view name,
    TF_Output features,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Softplus", std::string(name))
        .AddInput(features)
        .SetAttrType("T", T)
        .Finish());
}

/// Softsign activation: x / (|x| + 1)

[[nodiscard]] inline OpResult Softsign(
    Graph& graph,
    std::string_view name,
    TF_Output features,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("Softsign", std::string(name))
        .AddInput(features)
        .SetAttrType("T", T)
        .Finish());
}

/// Adds bias to value

[[nodiscard]] inline OpResult BiasAdd(
    Graph& graph,
    std::string_view name,
    TF_Output value,
    TF_Output bias,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("BiasAdd", std::string(name))
        .AddInput(value)
        .AddInput(bias)
        .SetAttrType("T", T)
        .Finish());
}

/// 2D convolution

[[nodiscard]] inline OpResult Conv2D(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output filter,
    TF_DataType T,
    std::span<const int64_t> strides,
    std::string_view padding) {
    return OpResult(
        graph.NewOperation("Conv2D", std::string(name))
        .AddInput(input)
        .AddInput(filter)
        .SetAttrType("T", T)
        .SetAttrIntList("strides", strides)
        .SetAttrString("padding", padding)
        .Finish());
}

/// Conv2D input gradient (transposed convolution)

[[nodiscard]] inline OpResult Conv2DBackpropInput(
    Graph& graph,
    std::string_view name,
    TF_Output input_sizes,
    TF_Output filter,
    TF_Output out_backprop,
    TF_DataType T,
    std::span<const int64_t> strides,
    std::string_view padding) {
    return OpResult(
        graph.NewOperation("Conv2DBackpropInput", std::string(name))
        .AddInput(input_sizes)
        .AddInput(filter)
        .AddInput(out_backprop)
        .SetAttrType("T", T)
        .SetAttrIntList("strides", strides)
        .SetAttrString("padding", padding)
        .Finish());
}

/// Depthwise 2D convolution

[[nodiscard]] inline OpResult DepthwiseConv2dNative(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_Output filter,
    TF_DataType T,
    std::span<const int64_t> strides,
    std::string_view padding) {
    return OpResult(
        graph.NewOperation("DepthwiseConv2dNative", std::string(name))
        .AddInput(input)
        .AddInput(filter)
        .SetAttrType("T", T)
        .SetAttrIntList("strides", strides)
        .SetAttrString("padding", padding)
        .Finish());
}

/// Max pooling

[[nodiscard]] inline OpResult MaxPool(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_DataType T,
    std::span<const int64_t> ksize,
    std::span<const int64_t> strides,
    std::string_view padding) {
    return OpResult(
        graph.NewOperation("MaxPool", std::string(name))
        .AddInput(input)
        .SetAttrType("T", T)
        .SetAttrIntList("ksize", ksize)
        .SetAttrIntList("strides", strides)
        .SetAttrString("padding", padding)
        .Finish());
}

/// Average pooling

[[nodiscard]] inline OpResult AvgPool(
    Graph& graph,
    std::string_view name,
    TF_Output value,
    TF_DataType T,
    std::span<const int64_t> ksize,
    std::span<const int64_t> strides,
    std::string_view padding) {
    return OpResult(
        graph.NewOperation("AvgPool", std::string(name))
        .AddInput(value)
        .SetAttrType("T", T)
        .SetAttrIntList("ksize", ksize)
        .SetAttrIntList("strides", strides)
        .SetAttrString("padding", padding)
        .Finish());
}

/// 3D max pooling

[[nodiscard]] inline OpResult MaxPool3D(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_DataType T,
    std::span<const int64_t> ksize,
    std::span<const int64_t> strides,
    std::string_view padding) {
    return OpResult(
        graph.NewOperation("MaxPool3D", std::string(name))
        .AddInput(input)
        .SetAttrType("T", T)
        .SetAttrIntList("ksize", ksize)
        .SetAttrIntList("strides", strides)
        .SetAttrString("padding", padding)
        .Finish());
}

/// 3D average pooling

[[nodiscard]] inline OpResult AvgPool3D(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_DataType T,
    std::span<const int64_t> ksize,
    std::span<const int64_t> strides,
    std::string_view padding) {
    return OpResult(
        graph.NewOperation("AvgPool3D", std::string(name))
        .AddInput(input)
        .SetAttrType("T", T)
        .SetAttrIntList("ksize", ksize)
        .SetAttrIntList("strides", strides)
        .SetAttrString("padding", padding)
        .Finish());
}

/// Fused batch normalization

[[nodiscard]] inline OpResult FusedBatchNorm(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output scale,
    TF_Output offset,
    TF_Output mean,
    TF_Output variance,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("FusedBatchNorm", std::string(name))
        .AddInput(x)
        .AddInput(scale)
        .AddInput(offset)
        .AddInput(mean)
        .AddInput(variance)
        .SetAttrType("T", T)
        .Finish());
}

/// Fused batch normalization V3

[[nodiscard]] inline OpResult FusedBatchNormV3(
    Graph& graph,
    std::string_view name,
    TF_Output x,
    TF_Output scale,
    TF_Output offset,
    TF_Output mean,
    TF_Output variance,
    TF_DataType T,
    TF_DataType U) {
    return OpResult(
        graph.NewOperation("FusedBatchNormV3", std::string(name))
        .AddInput(x)
        .AddInput(scale)
        .AddInput(offset)
        .AddInput(mean)
        .AddInput(variance)
        .SetAttrType("T", T)
        .SetAttrType("U", U)
        .Finish());
}

/// Local response normalization

[[nodiscard]] inline OpResult LRN(
    Graph& graph,
    std::string_view name,
    TF_Output input,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("LRN", std::string(name))
        .AddInput(input)
        .SetAttrType("T", T)
        .Finish());
}

// NOTE: "Dropout" op does not exist in TensorFlow C API.
// Dropout is typically implemented using primitive ops:
//
// Example dropout implementation (keep_prob = 1 - rate):
//   1. Generate random values: RandomUniform(graph, "rand", shape, TF_FLOAT)
//   2. Create mask: Greater(graph, "mask", rand_output, rate_output, TF_FLOAT)
//   3. Cast mask to float: Cast(graph, "mask_float", mask_output, TF_BOOL, TF_FLOAT)
//   4. Scale factor: RealDiv(graph, "scale", one_output, keep_prob_output, TF_FLOAT)
//   5. Apply: Mul(graph, "dropped", Mul(..., x, mask_float), scale, TF_FLOAT)
//
// Or use Select() with a random boolean mask and zeros.

/// Softmax cross entropy loss

[[nodiscard]] inline OpResult SoftmaxCrossEntropyWithLogits(
    Graph& graph,
    std::string_view name,
    TF_Output features,
    TF_Output labels,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("SoftmaxCrossEntropyWithLogits", std::string(name))
        .AddInput(features)
        .AddInput(labels)
        .SetAttrType("T", T)
        .Finish());
}

/// Sparse softmax cross entropy loss

[[nodiscard]] inline OpResult SparseSoftmaxCrossEntropyWithLogits(
    Graph& graph,
    std::string_view name,
    TF_Output features,
    TF_Output labels,
    TF_DataType T,
    TF_DataType Tlabels) {
    return OpResult(
        graph.NewOperation("SparseSoftmaxCrossEntropyWithLogits", std::string(name))
        .AddInput(features)
        .AddInput(labels)
        .SetAttrType("T", T)
        .SetAttrType("Tlabels", Tlabels)
        .Finish());
}



} // namespace ops
} // namespace tf_wrap
