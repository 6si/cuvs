/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/gpu_hnsw/gpu_hnsw_impl.cuh"

namespace cuvs::neighbors::gpu_hnsw {

// --- index destructor instantiations ---
template index<float>::~index();

// --- from_hnsw_index instantiations ---
template <>
std::unique_ptr<index<float>> from_hnsw_index(
  raft::resources const& res,
  const cuvs::neighbors::hnsw::index<float>& hnsw_index,
  raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
{
  return detail::from_hnsw_index_impl<float>(res, hnsw_index, dataset);
}

// --- search instantiations ---
template <>
void search(raft::resources const& res,
            const search_params& params,
            const index<float>& idx,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances)
{
  detail::search_impl<float>(res, params, idx, queries, neighbors, distances);
}

}  // namespace cuvs::neighbors::gpu_hnsw
