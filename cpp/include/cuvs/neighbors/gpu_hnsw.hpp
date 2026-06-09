/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/hnsw.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace cuvs::neighbors::gpu_hnsw {

/**
 * @defgroup gpu_hnsw_cpp_index_params GPU HNSW search parameters
 * @{
 */

/**
 * @brief Search parameters for GPU HNSW search.
 *
 * GPU HNSW search is a two-phase approach:
 *   Phase 1: Upper-layer greedy navigation (one warp per query) finds
 *            an entry point into layer 0 that is already in the target's neighborhood.
 *   Phase 2: Layer-0 beam search (CAGRA-style, one thread block per query)
 *            uses the guided entry point for high-recall local search.
 */
struct search_params {
  /** ef parameter (search budget): size of the dynamic candidate list during layer-0 beam search.
   *  Equivalent to HNSW ef_search. Higher ef → higher recall, more compute. */
  int ef = 200;

  /** Number of candidates to expand in parallel during layer-0 beam search.
   *  Analogous to CAGRA search_width. */
  int search_width = 4;

  /** Maximum iterations for layer-0 beam search before stopping.
   *  0 = auto (derived from ef and graph degree). */
  int max_iterations = 0;

  /** Number of threads per block for layer-0 beam search kernel. 0 = auto. */
  int thread_block_size = 0;
};

/**
 * @}
 */

/**
 * @defgroup gpu_hnsw_cpp_gpu_graph GPU-friendly HNSW graph representation
 * @{
 */

/**
 * @brief A single layer of the HNSW graph stored in GPU-friendly format.
 *
 * For layer 0 (dense): all N nodes present, stored as [N x max_degree] array.
 * For upper layers (sparse): only nodes that exist at that layer are stored,
 * with a node_ids mapping array for global↔layer-local ID translation.
 */
struct gpu_layer {
  /** Global node IDs of nodes present at this layer (sorted).
   *  For layer 0 this is empty (implicit identity mapping). */
  std::vector<uint32_t> node_ids;

  /** Adjacency list: [num_nodes x max_degree].
   *  neighbor_ids[i * max_degree + j] = j-th neighbor of the i-th node at this layer.
   *  Padded with UINT32_MAX for nodes with fewer than max_degree neighbors.
   *  Neighbor IDs are global (not layer-local). */
  std::vector<uint32_t> neighbor_ids;

  /** Number of nodes at this layer */
  uint32_t num_nodes = 0;

  /** Maximum degree (number of neighbors per node) at this layer */
  uint32_t max_degree = 0;
};

/**
 * @brief GPU-resident HNSW index for fast search.
 *
 * Holds the multi-layer HNSW graph on GPU memory alongside the dataset vectors.
 * Constructed from a CPU-built HNSW index via from_hnsw_index().
 */
template <typename T>
struct index {
  /** Number of vectors in the dataset */
  int64_t n_rows() const { return n_rows_; }

  /** Dimensionality of vectors */
  int64_t dim() const { return dim_; }

  /** Distance metric */
  cuvs::distance::DistanceType metric() const { return metric_; }

  /** Number of layers (including layer 0) */
  int num_layers() const { return num_layers_; }

  /** Global entry point node ID (top of the hierarchy) */
  uint32_t entry_point() const { return entry_point_; }

  /** M parameter used during HNSW construction */
  int M() const { return M_; }

  /** Max degree at layer 0 (typically 2*M) */
  int max_degree0() const { return max_degree0_; }

  // ── GPU-resident data (opaque to header consumers) ──

  /** Layer 0 graph on device: [n_rows x max_degree0] */
  uint32_t* d_layer0_graph = nullptr;

  /** Upper layer graphs on device (one per layer, layers 1..num_layers-1) */
  struct device_upper_layer {
    uint32_t* d_node_ids   = nullptr;  // [num_nodes]
    uint32_t* d_neighbors  = nullptr;  // [num_nodes x max_degree]
    uint32_t num_nodes     = 0;
    uint32_t max_degree    = 0;
  };
  std::vector<device_upper_layer> upper_layers;

  /** Dataset vectors on device: [n_rows x dim], row-major */
  T* d_dataset = nullptr;

  // ── Host-side metadata ──
  int64_t n_rows_                      = 0;
  int64_t dim_                         = 0;
  cuvs::distance::DistanceType metric_ = cuvs::distance::DistanceType::L2Expanded;
  int num_layers_                      = 0;
  uint32_t entry_point_                = 0;
  int M_                               = 0;
  int max_degree0_                     = 0;

  ~index();
};

/**
 * @}
 */

/**
 * @defgroup gpu_hnsw_cpp_api GPU HNSW public API
 * @{
 */

/**
 * @brief Convert a CPU HNSW index to a GPU HNSW index.
 *
 * Extracts the multi-layer graph from the hnswlib-backed HNSW index,
 * converts it to GPU-friendly dense/CSR format, and uploads to device memory.
 *
 * @tparam T data element type (float, int8_t, uint8_t)
 * @param[in] res raft resources (provides CUDA stream, memory allocators)
 * @param[in] hnsw_index the CPU HNSW index (built via cuvs::neighbors::hnsw::build or loaded)
 * @param[in] dataset the dataset vectors on host [n_rows, dim], row-major
 * @return a GPU HNSW index ready for search
 */
template <typename T>
std::unique_ptr<index<T>> from_hnsw_index(
  raft::resources const& res,
  const cuvs::neighbors::hnsw::index<T>& hnsw_index,
  raft::host_matrix_view<const T, int64_t, raft::row_major> dataset);

/**
 * @brief Search the GPU HNSW index.
 *
 * Two-phase search:
 *   Phase 1: upper-layer greedy walk (one warp per query) finds entry point for layer 0.
 *   Phase 2: layer-0 beam search (one CTA per query) with guided entry point.
 *
 * @tparam T data element type (float, int8_t, uint8_t)
 * @param[in] res raft resources
 * @param[in] params search parameters
 * @param[in] idx the GPU HNSW index
 * @param[in] queries query vectors on device [n_queries, dim], row-major
 * @param[out] neighbors output neighbor indices on device [n_queries, k]
 * @param[out] distances output distances on device [n_queries, k]
 */
template <typename T>
void search(raft::resources const& res,
            const search_params& params,
            const index<T>& idx,
            raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
            raft::device_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

/**
 * @}
 */

}  // namespace cuvs::neighbors::gpu_hnsw
