/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/gpu_hnsw.hpp>
#include "gpu_hnsw_search_kernel.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace cuvs::neighbors::gpu_hnsw::detail {

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
                      : 2 * ef / sw + 10;

  bool use_ip = (idx.metric() == cuvs::distance::DistanceType::InnerProduct);

  const T* d_queries_t = queries.data_handle();
  const T* d_dataset_t = idx.d_dataset;

  uint32_t* d_entry_points = nullptr;
  RAFT_CUDA_TRY(cudaMalloc(&d_entry_points, num_queries * sizeof(uint32_t)));

  int num_upper_layers = static_cast<int>(idx.upper_layers.size());

  if (num_upper_layers > 0) {
    std::vector<upper_layer_ptrs> h_layer_ptrs(num_upper_layers);
    for (int i = 0; i < num_upper_layers; i++) {
      const auto& ul  = idx.upper_layers[i];
      h_layer_ptrs[i] = {ul.d_node_ids, ul.d_neighbors, ul.num_nodes, ul.max_degree};
    }
    upper_layer_ptrs* d_layer_ptrs = nullptr;
    RAFT_CUDA_TRY(cudaMalloc(&d_layer_ptrs, num_upper_layers * sizeof(upper_layer_ptrs)));
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(d_layer_ptrs, h_layer_ptrs.data(),
                      num_upper_layers * sizeof(upper_layer_ptrs),
                      cudaMemcpyHostToDevice, stream));

    int warps_per_block   = 4;
    int threads_per_block = warps_per_block * 32;
    int num_blocks        = (num_queries + warps_per_block - 1) / warps_per_block;

    upper_layer_search_kernel<T><<<num_blocks, threads_per_block, 0, stream>>>(
      d_queries_t, d_dataset_t, d_layer_ptrs, d_entry_points,
      idx.entry_point(), num_queries, dim, num_upper_layers, use_ip);

    RAFT_CUDA_TRY(cudaFree(d_layer_ptrs));
  } else {
    std::vector<uint32_t> h_eps(num_queries, idx.entry_point());
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(d_entry_points, h_eps.data(),
                      num_queries * sizeof(uint32_t),
                      cudaMemcpyHostToDevice, stream));
  }

  int block_size   = params.thread_block_size > 0 ? params.thread_block_size : 128;
  size_t smem_size = calc_layer0_smem_size(ef, sw, idx.max_degree0());

  // Allocate global-memory visited bitmaps (one per query, pre-zeroed)
  int N_int = static_cast<int>(idx.n_rows());
  size_t bitmap_bytes = calc_visited_bitmap_size(num_queries, N_int);
  uint32_t* d_visited_bitmaps = nullptr;
  RAFT_CUDA_TRY(cudaMalloc(&d_visited_bitmaps, bitmap_bytes));
  RAFT_CUDA_TRY(cudaMemsetAsync(d_visited_bitmaps, 0, bitmap_bytes, stream));

  layer0_beam_search_kernel<T><<<num_queries, block_size, smem_size, stream>>>(
    d_queries_t, d_dataset_t, idx.d_layer0_graph, d_entry_points,
    d_visited_bitmaps,
    neighbors.data_handle(), distances.data_handle(),
    num_queries, N_int, dim, idx.max_degree0(),
    k, ef, sw, max_iter, use_ip);

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  RAFT_CUDA_TRY(cudaFree(d_entry_points));
  RAFT_CUDA_TRY(cudaFree(d_visited_bitmaps));
}

}  // namespace cuvs::neighbors::gpu_hnsw::detail
