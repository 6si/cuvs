/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/gpu_hnsw/gpu_hnsw_convert_impl.hpp"
#include <cuvs/core/export.hpp>

namespace cuvs::neighbors::gpu_hnsw {

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

template CUVS_EXPORT index<float>::~index();

template <>
CUVS_EXPORT std::unique_ptr<index<float>> from_hnsw_index(
  raft::resources const& res,
  const cuvs::neighbors::hnsw::index<float>& hnsw_index,
  raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
{
  return detail::from_hnsw_index_impl<float>(res, hnsw_index, dataset);
}

}  // namespace cuvs::neighbors::gpu_hnsw
