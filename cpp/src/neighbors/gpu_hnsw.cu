/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/gpu_hnsw/gpu_hnsw_impl.cuh"
#include <cuvs/core/export.hpp>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace cuvs::neighbors::gpu_hnsw {

// float32
template <>
CUVS_EXPORT void search(raft::resources const& res,
            const search_params& params,
            const index<float>& idx,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances)
{
  detail::search_impl<float>(res, params, idx, queries, neighbors, distances);
}

// float16
template <>
CUVS_EXPORT void search(raft::resources const& res,
            const search_params& params,
            const index<__half>& idx,
            raft::device_matrix_view<const __half, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances)
{
  detail::search_impl<__half>(res, params, idx, queries, neighbors, distances);
}

// bfloat16
template <>
CUVS_EXPORT void search(raft::resources const& res,
            const search_params& params,
            const index<__nv_bfloat16>& idx,
            raft::device_matrix_view<const __nv_bfloat16, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances)
{
  detail::search_impl<__nv_bfloat16>(res, params, idx, queries, neighbors, distances);
}

// int8
template <>
CUVS_EXPORT void search(raft::resources const& res,
            const search_params& params,
            const index<int8_t>& idx,
            raft::device_matrix_view<const int8_t, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances)
{
  detail::search_impl<int8_t>(res, params, idx, queries, neighbors, distances);
}

}  // namespace cuvs::neighbors::gpu_hnsw
