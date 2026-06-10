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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace cuvs::neighbors::gpu_hnsw::detail {

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

}  // namespace cuvs::neighbors::gpu_hnsw::detail
