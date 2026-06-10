/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Suppress hnswlib sign-conversion warnings under nvcc
#pragma nv_diag_suppress 68

#include "detail/gpu_hnsw/gpu_hnsw_convert_impl.hpp"
#include <cuvs/core/export.hpp>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

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

// Destructor instantiations
template CUVS_EXPORT index<float>::~index();
template CUVS_EXPORT index<__half>::~index();
template CUVS_EXPORT index<__nv_bfloat16>::~index();
template CUVS_EXPORT index<int8_t>::~index();

// from_hnsw_index: float → float (existing)
template <>
CUVS_EXPORT std::unique_ptr<index<float>> from_hnsw_index(
  raft::resources const& res,
  const cuvs::neighbors::hnsw::index<float>& hnsw_index,
  raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
{
  return detail::from_hnsw_index_impl<float>(res, hnsw_index, dataset);
}

// from_hnsw_index_quantized: float → half (FP16)
template <>
CUVS_EXPORT std::unique_ptr<index<__half>> from_hnsw_index_quantized(
  raft::resources const& res,
  const cuvs::neighbors::hnsw::index<float>& hnsw_index,
  raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
{
  return detail::from_hnsw_index_quantized_impl<__half>(res, hnsw_index, dataset);
}

// from_hnsw_index_quantized: float → nv_bfloat16 (BF16)
template <>
CUVS_EXPORT std::unique_ptr<index<__nv_bfloat16>> from_hnsw_index_quantized(
  raft::resources const& res,
  const cuvs::neighbors::hnsw::index<float>& hnsw_index,
  raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
{
  return detail::from_hnsw_index_quantized_impl<__nv_bfloat16>(res, hnsw_index, dataset);
}

// from_hnsw_index_quantized: float → int8_t (INT8)
template <>
CUVS_EXPORT std::unique_ptr<index<int8_t>> from_hnsw_index_quantized(
  raft::resources const& res,
  const cuvs::neighbors::hnsw::index<float>& hnsw_index,
  raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
{
  return detail::from_hnsw_index_quantized_impl<int8_t>(res, hnsw_index, dataset);
}

}  // namespace cuvs::neighbors::gpu_hnsw
