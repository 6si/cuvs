/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/gpu_hnsw.hpp>
#include "gpu_hnsw_graph_extract.hpp"
#include "gpu_hnsw_search_kernel.cuh"

// Need hnsw detail for hnsw_dist_t type mapping
#include "../hnsw.hpp"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <hnswlib/hnswalg.h>
#include <hnswlib/hnswlib.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace cuvs::neighbors::gpu_hnsw::detail {

// ============================================================================
// Index destructor: free GPU memory
// ============================================================================

template <typename T>
index<T>::~index()
{
  if (d_layer0_graph) cudaFree(d_layer0_graph);
  if (d_dataset) cudaFree(d_dataset);
  for (auto& ul : upper_layers) {
    if (ul.d_node_ids) cudaFree(ul.d_node_ids);
    if (ul.d_neighbors) cudaFree(ul.d_neighbors);
  }
}

// ============================================================================
// from_hnsw_index: convert CPU HNSW → GPU HNSW
// ============================================================================

template <typename T>
std::unique_ptr<index<T>> from_hnsw_index_impl(
  raft::resources const& res,
  const cuvs::neighbors::hnsw::index<T>& hnsw_index,
  raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)
{
  auto stream = raft::resource::get_cuda_stream(res);

  // Access the underlying hnswlib index
  const auto* raw_index = hnsw_index.get_index();
  RAFT_EXPECTS(raw_index != nullptr, "HNSW index is null; ensure it is loaded before conversion");

  // The hnswlib index is stored as HierarchicalNSW<float> (or int for integer types)
  using DistT          = typename cuvs::neighbors::hnsw::detail::hnsw_dist_t<T>::type;
  const auto* hnsw_alg = static_cast<const hnswlib::HierarchicalNSW<DistT>*>(raw_index);

  int64_t n_rows = dataset.extent(0);
  int64_t dim    = dataset.extent(1);

  // --- Step 1: Extract HNSW layers from hnswlib into host-side gpu_layer structs ---
  std::vector<gpu_layer> layers;
  uint32_t entry_point;
  int M, max_degree0;
  extract_hnsw_layers(*hnsw_alg, layers, entry_point, M, max_degree0);

  int num_layers = static_cast<int>(layers.size());

  // --- Step 2: Create GPU index and allocate device memory ---
  auto gpu_idx          = std::make_unique<index<T>>();
  gpu_idx->n_rows_      = n_rows;
  gpu_idx->dim_         = dim;
  gpu_idx->metric_      = hnsw_index.metric();
  gpu_idx->num_layers_  = num_layers;
  gpu_idx->entry_point_ = entry_point;
  gpu_idx->M_           = M;
  gpu_idx->max_degree0_ = max_degree0;

  // --- Step 3: Upload dataset to GPU ---
  size_t dataset_bytes = n_rows * dim * sizeof(T);
  RAFT_CUDA_TRY(cudaMalloc(&gpu_idx->d_dataset, dataset_bytes));
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(gpu_idx->d_dataset, dataset.data_handle(), dataset_bytes,
                    cudaMemcpyHostToDevice, stream));

  // --- Step 4: Upload layer 0 graph to GPU ---
  {
    const auto& L0       = layers[0];
    size_t graph0_bytes  = static_cast<size_t>(L0.num_nodes) * L0.max_degree * sizeof(uint32_t);
    RAFT_CUDA_TRY(cudaMalloc(&gpu_idx->d_layer0_graph, graph0_bytes));
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(gpu_idx->d_layer0_graph, L0.neighbor_ids.data(), graph0_bytes,
                      cudaMemcpyHostToDevice, stream));
  }

  // --- Step 5: Upload upper layer graphs to GPU ---
  gpu_idx->upper_layers.resize(num_layers > 1 ? num_layers - 1 : 0);
  for (int layer = 1; layer < num_layers; layer++) {
    const auto& UL = layers[layer];
    auto& dul      = gpu_idx->upper_layers[layer - 1];
    dul.num_nodes  = UL.num_nodes;
    dul.max_degree = UL.max_degree;

    // Upload node IDs
    size_t ids_bytes = UL.num_nodes * sizeof(uint32_t);
    RAFT_CUDA_TRY(cudaMalloc(&dul.d_node_ids, ids_bytes));
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(dul.d_node_ids, UL.node_ids.data(), ids_bytes,
                      cudaMemcpyHostToDevice, stream));

    // Upload neighbor lists
    size_t nbrs_bytes = static_cast<size_t>(UL.num_nodes) * UL.max_degree * sizeof(uint32_t);
    RAFT_CUDA_TRY(cudaMalloc(&dul.d_neighbors, nbrs_bytes));
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(dul.d_neighbors, UL.neighbor_ids.data(), nbrs_bytes,
                      cudaMemcpyHostToDevice, stream));
  }

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  return gpu_idx;
}

// ============================================================================
// search: two-phase GPU HNSW search
// ============================================================================

template <typename T>
void search_impl(raft::resources const& res,
                 const search_params& params,
                 const index<T>& idx,
                 raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
                 raft::device_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
                 raft::device_matrix_view<float, int64_t, raft::row_major> distances)
{
  auto stream = raft::resource::get_cuda_stream(res);

  int num_queries = queries.extent(0);
  int dim         = queries.extent(1);
  int k           = neighbors.extent(1);
  int ef          = params.ef;
  int sw          = params.search_width;
  int max_iter    = params.max_iterations > 0
                      ? params.max_iterations
                      : 2 * ef / sw + 10;  // auto: enough iterations to expand ef candidates

  bool use_ip = (idx.metric() == cuvs::distance::DistanceType::InnerProduct);

  // Currently only float queries supported. For int8/uint8, we'd need quantized distance kernels.
  static_assert(std::is_same_v<T, float>,
                "GPU HNSW search currently only supports float data type");

  const float* d_queries_f = reinterpret_cast<const float*>(queries.data_handle());
  const float* d_dataset_f = reinterpret_cast<const float*>(idx.d_dataset);

  // --- Phase 1: Upper-layer search ---
  uint32_t* d_entry_points = nullptr;
  RAFT_CUDA_TRY(cudaMalloc(&d_entry_points, num_queries * sizeof(uint32_t)));

  int num_upper_layers = static_cast<int>(idx.upper_layers.size());

  if (num_upper_layers > 0) {
    // Prepare upper layer pointer array on device
    std::vector<upper_layer_ptrs> h_layer_ptrs(num_upper_layers);
    for (int i = 0; i < num_upper_layers; i++) {
      const auto& ul   = idx.upper_layers[i];
      h_layer_ptrs[i] = {ul.d_node_ids, ul.d_neighbors, ul.num_nodes, ul.max_degree};
    }
    upper_layer_ptrs* d_layer_ptrs = nullptr;
    RAFT_CUDA_TRY(cudaMalloc(&d_layer_ptrs, num_upper_layers * sizeof(upper_layer_ptrs)));
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(d_layer_ptrs, h_layer_ptrs.data(),
                      num_upper_layers * sizeof(upper_layer_ptrs),
                      cudaMemcpyHostToDevice, stream));

    // Launch Phase 1: one warp per query
    int warps_per_block = 4;  // 128 threads = 4 warps
    int threads_per_block = warps_per_block * 32;
    int num_blocks = (num_queries + warps_per_block - 1) / warps_per_block;

    upper_layer_search_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      d_queries_f,
      d_dataset_f,
      d_layer_ptrs,
      d_entry_points,
      idx.entry_point(),
      num_queries,
      dim,
      num_upper_layers,
      use_ip);

    RAFT_CUDA_TRY(cudaFree(d_layer_ptrs));
  } else {
    // Single-layer graph (no upper layers): all queries start from entry point
    std::vector<uint32_t> h_eps(num_queries, idx.entry_point());
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(d_entry_points, h_eps.data(),
                      num_queries * sizeof(uint32_t),
                      cudaMemcpyHostToDevice, stream));
  }

  // --- Phase 2: Layer-0 beam search ---
  int block_size = params.thread_block_size > 0 ? params.thread_block_size : 128;
  size_t smem_size = calc_layer0_smem_size(ef, sw, idx.max_degree0());

  layer0_beam_search_kernel<<<num_queries, block_size, smem_size, stream>>>(
    d_queries_f,
    d_dataset_f,
    idx.d_layer0_graph,
    d_entry_points,
    neighbors.data_handle(),
    distances.data_handle(),
    num_queries,
    static_cast<int>(idx.n_rows()),
    dim,
    idx.max_degree0(),
    k,
    ef,
    sw,
    max_iter,
    use_ip);

  RAFT_CUDA_TRY(cudaFree(d_entry_points));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
}

}  // namespace cuvs::neighbors::gpu_hnsw::detail
