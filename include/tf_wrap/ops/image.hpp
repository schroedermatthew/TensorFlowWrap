// tf_wrap/ops/image.hpp
// Image operation wrappers (generated)

#pragma once

#include "tf_wrap/ops/common.hpp"

namespace tf_wrap {
namespace ops {

// ============================================================================
// Image Operations
// ============================================================================

/// Resize images using bilinear interpolation

[[nodiscard]] inline OpResult ResizeBilinear(
    Graph& graph,
    std::string_view name,
    TF_Output images,
    TF_Output size,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("ResizeBilinear", std::string(name))
        .AddInput(images)
        .AddInput(size)
        .SetAttrType("T", T)
        .Finish());
}

/// Resize images using nearest neighbor

[[nodiscard]] inline OpResult ResizeNearestNeighbor(
    Graph& graph,
    std::string_view name,
    TF_Output images,
    TF_Output size,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("ResizeNearestNeighbor", std::string(name))
        .AddInput(images)
        .AddInput(size)
        .SetAttrType("T", T)
        .Finish());
}

/// Resize images using bicubic interpolation

[[nodiscard]] inline OpResult ResizeBicubic(
    Graph& graph,
    std::string_view name,
    TF_Output images,
    TF_Output size,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("ResizeBicubic", std::string(name))
        .AddInput(images)
        .AddInput(size)
        .SetAttrType("T", T)
        .Finish());
}

/// Extract and resize crops from images

[[nodiscard]] inline OpResult CropAndResize(
    Graph& graph,
    std::string_view name,
    TF_Output image,
    TF_Output boxes,
    TF_Output box_ind,
    TF_Output crop_size,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("CropAndResize", std::string(name))
        .AddInput(image)
        .AddInput(boxes)
        .AddInput(box_ind)
        .AddInput(crop_size)
        .SetAttrType("T", T)
        .Finish());
}

/// Non-maximum suppression for object detection

[[nodiscard]] inline OpResult NonMaxSuppression(
    Graph& graph,
    std::string_view name,
    TF_Output boxes,
    TF_Output scores,
    TF_Output max_output_size) {
    return OpResult(
        graph.NewOperation("NonMaxSuppression", std::string(name))
        .AddInput(boxes)
        .AddInput(scores)
        .AddInput(max_output_size)
        .Finish());
}

/// Non-maximum suppression V3

[[nodiscard]] inline OpResult NonMaxSuppressionV3(
    Graph& graph,
    std::string_view name,
    TF_Output boxes,
    TF_Output scores,
    TF_Output max_output_size,
    TF_Output iou_threshold,
    TF_Output score_threshold) {
    return OpResult(
        graph.NewOperation("NonMaxSuppressionV3", std::string(name))
        .AddInput(boxes)
        .AddInput(scores)
        .AddInput(max_output_size)
        .AddInput(iou_threshold)
        .AddInput(score_threshold)
        .Finish());
}

/// Decode JPEG image

[[nodiscard]] inline OpResult DecodeJpeg(
    Graph& graph,
    std::string_view name,
    TF_Output contents) {
    return OpResult(graph.NewOperation("DecodeJpeg", std::string(name)).AddInput(contents).Finish());
}

/// Decode PNG image

[[nodiscard]] inline OpResult DecodePng(
    Graph& graph,
    std::string_view name,
    TF_Output contents,
    TF_DataType dtype) {
    return OpResult(
        graph.NewOperation("DecodePng", std::string(name))
        .AddInput(contents)
        .SetAttrType("dtype", dtype)
        .Finish());
}

/// Encode image as JPEG

[[nodiscard]] inline OpResult EncodeJpeg(
    Graph& graph,
    std::string_view name,
    TF_Output image) {
    return OpResult(graph.NewOperation("EncodeJpeg", std::string(name)).AddInput(image).Finish());
}

/// Encode image as PNG

[[nodiscard]] inline OpResult EncodePng(
    Graph& graph,
    std::string_view name,
    TF_Output image,
    TF_DataType T) {
    return OpResult(
        graph.NewOperation("EncodePng", std::string(name))
        .AddInput(image)
        .SetAttrType("T", T)
        .Finish());
}



} // namespace ops
} // namespace tf_wrap
