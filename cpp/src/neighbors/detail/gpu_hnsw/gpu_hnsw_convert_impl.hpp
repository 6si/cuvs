/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/gpu_hnsw.hpp>
#include "gpu_hnsw_graph_extract.hpp"

#include "../hnsw.hpp"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <hnswlib/hnswalg.h>
#include <hnswlib/hnswlib.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace cuvs::neighbors::gpu_hnsw::detail {

// ============================================================================
// Host-side quantization helpers
// ============================================================================

template <typename QuantT>
struct host_quantize;

template <>
struct host_quantize<__half> {
  static void convert(const float* src, __half* dst, int64_t count, int64_t /*dim*/)
  {
    for (int64_t i = 0; i < count; i++) {
      dst[i] = __float2half_rn(src[i]);
    }
  }
};

template <>
struct host_quantize<__nv_bfloat16> {
  static void convert(const float* src, __nv_bfloat16* dst, int64_t count, int64_t /*dim*/)
  {
    for (int64_t i = 0; i < count; i++) {
      dst[i] = __float2bfloat16_rn(src[i]);
    }
  }
};

template <>
struct host_quantize<int8_t> {
  static void convert(const float* src, int8_t* dst, int64_t count, int64_t /*dim*/)
  {
    // Symmetric quantization: scale = max(|src|) / 127
    float max_abs = 0.0f;
    for (int64_t i = 0; i < count; i++) {
      float a = std::fabs(src[i]);
      if (a > max_abs) max_abs = a;
    }
    float scale = (max_abs > 0.0f) ? (127.0f / max_abs) : 1.0f;
    for (int64_t i = 0; i < count; i++) {
      float scaled = src[i] * scale;
      scaled       = std::max(-127.0f, std::min(127.0f, scaled));
      dst[i]       = static_cast<int8_t>(std::roundf(scaled));
    }
  }
};

template <typename T>
std::unique_ptr<index<T>> from_hnsw_index_impl(
  raft::resources const& res,
  const cuvs::neighbors::hnsw::index<T>& hnsw_index,
  raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)
{
  auto stream = raft::resource::get_cuda_stream(res);

  const auto* raw_index = hnsw_index.get_index();
  RAFT_EXPECTS(raw_index != nullptr, "HNSW index is null; ensure it is loaded before conversion");

  using DistT          = typename cuvs::neighbors::hnsw::detail::hnsw_dist_t<T>::type;
  const auto* hnsw_alg = static_cast<const hnswlib::HierarchicalNSW<DistT>*>(raw_index);

  int64_t n_rows = dataset.extent(0);
  int64_t dim    = dataset.extent(1);

  std::vector<gpu_layer> layers;
  uint32_t entry_point;
  int M, max_degree0;
  extract_hnsw_layers(*hnsw_alg, layers, entry_point, M, max_degree0);

  int num_layers = static_cast<int>(layers.size());

  auto gpu_idx          = std::make_unique<index<T>>();
  gpu_idx->n_rows_      = n_rows;
  gpu_idx->dim_         = dim;
  gpu_idx->metric_      = hnsw_index.metric();
  gpu_idx->num_layers_  = num_layers;
  gpu_idx->entry_point_ = entry_point;
  gpu_idx->M_           = M;
  gpu_idx->max_degree0_ = max_degree0;

  size_t dataset_bytes = n_rows * dim * sizeof(T);
  RAFT_CUDA_TRY(cudaMalloc(&gpu_idx->d_dataset, dataset_bytes));
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(gpu_idx->d_dataset, dataset.data_handle(), dataset_bytes,
                    cudaMemcpyHostToDevice, stream));

  {
    const auto& L0      = layers[0];
    size_t graph0_bytes = static_cast<size_t>(L0.num_nodes) * L0.max_degree * sizeof(uint32_t);
    RAFT_CUDA_TRY(cudaMalloc(&gpu_idx->d_layer0_graph, graph0_bytes));
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(gpu_idx->d_layer0_graph, L0.neighbor_ids.data(), graph0_bytes,
                      cudaMemcpyHostToDevice, stream));
  }

  gpu_idx->upper_layers.resize(num_layers > 1 ? num_layers - 1 : 0);
  for (int layer = 1; layer < num_layers; layer++) {
    const auto& UL = layers[layer];
    auto& dul      = gpu_idx->upper_layers[layer - 1];
    dul.num_nodes  = UL.num_nodes;
    dul.max_degree = UL.max_degree;

    size_t ids_bytes = UL.num_nodes * sizeof(uint32_t);
    RAFT_CUDA_TRY(cudaMalloc(&dul.d_node_ids, ids_bytes));
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(dul.d_node_ids, UL.node_ids.data(), ids_bytes,
                      cudaMemcpyHostToDevice, stream));

    size_t nbrs_bytes = static_cast<size_t>(UL.num_nodes) * UL.max_degree * sizeof(uint32_t);
    RAFT_CUDA_TRY(cudaMalloc(&dul.d_neighbors, nbrs_bytes));
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(dul.d_neighbors, UL.neighbor_ids.data(), nbrs_bytes,
                      cudaMemcpyHostToDevice, stream));
  }

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  return gpu_idx;
}

/**
 * Quantized conversion: float HNSW index + float dataset → index<QuantT>.
 * Extracts the graph (same for all types), quantizes dataset, uploads to GPU.
 */
template <typename QuantT>
std::unique_ptr<index<QuantT>> from_hnsw_index_quantized_impl(
  raft::resources const& res,
  const cuvs::neighbors::hnsw::index<float>& hnsw_index,
  raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
{
  auto stream = raft::resource::get_cuda_stream(res);

  const auto* raw_index = hnsw_index.get_index();
  RAFT_EXPECTS(raw_index != nullptr, "HNSW index is null; ensure it is loaded before conversion");

  using DistT          = typename cuvs::neighbors::hnsw::detail::hnsw_dist_t<float>::type;
  const auto* hnsw_alg = static_cast<const hnswlib::HierarchicalNSW<DistT>*>(raw_index);

  int64_t n_rows = dataset.extent(0);
  int64_t dim    = dataset.extent(1);

  // Extract graph (same for all types — graph is uint32 neighbor IDs)
  std::vector<gpu_layer> layers;
  uint32_t entry_point;
  int M, max_degree0;
  extract_hnsw_layers(*hnsw_alg, layers, entry_point, M, max_degree0);

  int num_layers = static_cast<int>(layers.size());

  auto gpu_idx          = std::make_unique<index<QuantT>>();
  gpu_idx->n_rows_      = n_rows;
  gpu_idx->dim_         = dim;
  gpu_idx->metric_      = hnsw_index.metric();
  gpu_idx->num_layers_  = num_layers;
  gpu_idx->entry_point_ = entry_point;
  gpu_idx->M_           = M;
  gpu_idx->max_degree0_ = max_degree0;

  // Quantize dataset on host, then upload
  int64_t total_elems = n_rows * dim;
  std::vector<QuantT> h_quantized(total_elems);
  host_quantize<QuantT>::convert(dataset.data_handle(), h_quantized.data(), total_elems, dim);

  size_t dataset_bytes = total_elems * sizeof(QuantT);
  RAFT_CUDA_TRY(cudaMalloc(&gpu_idx->d_dataset, dataset_bytes));
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(gpu_idx->d_dataset, h_quantized.data(), dataset_bytes,
                    cudaMemcpyHostToDevice, stream));

  // Upload graph (identical to float path)
  {
    const auto& L0      = layers[0];
    size_t graph0_bytes = static_cast<size_t>(L0.num_nodes) * L0.max_degree * sizeof(uint32_t);
    RAFT_CUDA_TRY(cudaMalloc(&gpu_idx->d_layer0_graph, graph0_bytes));
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(gpu_idx->d_layer0_graph, L0.neighbor_ids.data(), graph0_bytes,
                      cudaMemcpyHostToDevice, stream));
  }

  gpu_idx->upper_layers.resize(num_layers > 1 ? num_layers - 1 : 0);
  for (int layer = 1; layer < num_layers; layer++) {
    const auto& UL = layers[layer];
    auto& dul      = gpu_idx->upper_layers[layer - 1];
    dul.num_nodes  = UL.num_nodes;
    dul.max_degree = UL.max_degree;

    size_t ids_bytes = UL.num_nodes * sizeof(uint32_t);
    RAFT_CUDA_TRY(cudaMalloc(&dul.d_node_ids, ids_bytes));
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(dul.d_node_ids, UL.node_ids.data(), ids_bytes,
                      cudaMemcpyHostToDevice, stream));

    size_t nbrs_bytes = static_cast<size_t>(UL.num_nodes) * UL.max_degree * sizeof(uint32_t);
    RAFT_CUDA_TRY(cudaMalloc(&dul.d_neighbors, nbrs_bytes));
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(dul.d_neighbors, UL.neighbor_ids.data(), nbrs_bytes,
                      cudaMemcpyHostToDevice, stream));
  }

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  return gpu_idx;
}

}  // namespace cuvs::neighbors::gpu_hnsw::detail
