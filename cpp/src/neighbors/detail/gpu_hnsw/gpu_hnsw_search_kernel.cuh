/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

namespace cuvs::neighbors::gpu_hnsw::detail {

// ============================================================================
// Distance computation helpers
// ============================================================================

/**
 * Warp-cooperative L2 squared distance.
 * All 32 lanes compute partial sums over strided dimensions, then reduce via shuffle.
 * Only lane 0 holds the final result.
 */
__device__ __forceinline__ float warp_l2_distance(const float* __restrict__ a,
                                                  const float* __restrict__ b,
                                                  int dim)
{
  float partial = 0.0f;
  int lane      = threadIdx.x % 32;
  for (int d = lane; d < dim; d += 32) {
    float diff = a[d] - b[d];
    partial += diff * diff;
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    partial += __shfl_down_sync(0xffffffff, partial, offset);
  }
  return partial;
}

/**
 * Warp-cooperative inner product distance (negated for min-heap compatibility).
 */
__device__ __forceinline__ float warp_ip_distance(const float* __restrict__ a,
                                                  const float* __restrict__ b,
                                                  int dim)
{
  float partial = 0.0f;
  int lane      = threadIdx.x % 32;
  for (int d = lane; d < dim; d += 32) {
    partial += a[d] * b[d];
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    partial += __shfl_down_sync(0xffffffff, partial, offset);
  }
  return -partial;
}

// ============================================================================
// Phase 1: Upper-layer greedy search
// ============================================================================

struct upper_layer_ptrs {
  const uint32_t* d_node_ids;   // [num_nodes] sorted global IDs at this layer
  const uint32_t* d_neighbors;  // [num_nodes x max_degree]
  uint32_t num_nodes;
  uint32_t max_degree;
};

__device__ __forceinline__ uint32_t binary_search_node(const uint32_t* d_node_ids,
                                                       uint32_t n,
                                                       uint32_t global_id)
{
  uint32_t lo = 0, hi = n;
  while (lo < hi) {
    uint32_t mid = (lo + hi) / 2;
    if (d_node_ids[mid] < global_id) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo < n && d_node_ids[lo] == global_id) return lo;
  return UINT32_MAX;
}

/**
 * Phase 1: one warp per query, greedy walk from top layer down to layer 1.
 * Outputs the entry point for layer-0 beam search.
 */
__global__ void upper_layer_search_kernel(
  const float* __restrict__ d_queries,
  const float* __restrict__ d_dataset,
  const upper_layer_ptrs* __restrict__ d_layer_ptrs,
  uint32_t* __restrict__ d_entry_points,
  uint32_t global_entry_point,
  int num_queries,
  int dim,
  int num_upper_layers,
  bool use_inner_product)
{
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int lane    = threadIdx.x % 32;
  if (warp_id >= num_queries) return;

  const float* query = d_queries + static_cast<int64_t>(warp_id) * dim;
  uint32_t current   = global_entry_point;

  float best_dist;
  if (use_inner_product) {
    best_dist = warp_ip_distance(query, d_dataset + static_cast<int64_t>(current) * dim, dim);
  } else {
    best_dist = warp_l2_distance(query, d_dataset + static_cast<int64_t>(current) * dim, dim);
  }
  best_dist = __shfl_sync(0xffffffff, best_dist, 0);  // broadcast to all lanes

  // Traverse from top layer down to layer 1
  for (int li = num_upper_layers - 1; li >= 0; li--) {
    const upper_layer_ptrs& lp = d_layer_ptrs[li];
    bool improved = true;
    while (improved) {
      improved = false;
      uint32_t local_idx = binary_search_node(lp.d_node_ids, lp.num_nodes, current);
      if (local_idx == UINT32_MAX) break;

      // Each lane checks a different neighbor
      uint32_t best_nbr  = UINT32_MAX;
      float best_nbr_dist = best_dist;

      for (uint32_t j = lane; j < lp.max_degree; j += 32) {
        uint32_t nbr = lp.d_neighbors[static_cast<int64_t>(local_idx) * lp.max_degree + j];
        if (nbr == UINT32_MAX) continue;

        float dist;
        if (use_inner_product) {
          dist = warp_ip_distance(query, d_dataset + static_cast<int64_t>(nbr) * dim, dim);
        } else {
          dist = warp_l2_distance(query, d_dataset + static_cast<int64_t>(nbr) * dim, dim);
        }
        // Only lane 0 has the correct reduced distance
        dist = __shfl_sync(0xffffffff, dist, 0);

        if (dist < best_nbr_dist) {
          best_nbr_dist = dist;
          best_nbr      = nbr;
        }
      }

      // Warp-level reduction: find the best neighbor across all lanes
      for (int offset = 16; offset > 0; offset >>= 1) {
        float other_dist  = __shfl_down_sync(0xffffffff, best_nbr_dist, offset);
        uint32_t other_id = __shfl_down_sync(0xffffffff, best_nbr, offset);
        if (other_dist < best_nbr_dist) {
          best_nbr_dist = other_dist;
          best_nbr      = other_id;
        }
      }
      // Broadcast result to all lanes
      best_nbr_dist = __shfl_sync(0xffffffff, best_nbr_dist, 0);
      best_nbr      = __shfl_sync(0xffffffff, best_nbr, 0);

      if (best_nbr != UINT32_MAX && best_nbr_dist < best_dist) {
        best_dist = best_nbr_dist;
        current   = best_nbr;
        improved  = true;
      }
    }
  }

  if (lane == 0) {
    d_entry_points[warp_id] = current;
  }
}

// ============================================================================
// Phase 2: Layer-0 beam search kernel
// ============================================================================

/**
 * Hash table insert with linear probing.
 * Returns true if the node was newly inserted, false if already present.
 */
__device__ __forceinline__ bool hash_insert(uint32_t* hash_table,
                                            uint32_t hash_mask,
                                            uint32_t node_id)
{
  uint32_t slot = (node_id * 2654435761u) & hash_mask;  // Knuth multiplicative hash
  for (uint32_t i = 0; i < 64; i++) {
    uint32_t probe = (slot + i) & hash_mask;
    uint32_t old   = atomicCAS(&hash_table[probe], UINT32_MAX, node_id);
    if (old == UINT32_MAX) return true;   // Newly inserted
    if (old == node_id) return false;     // Already present
  }
  return false;  // Hash full — treat as visited to avoid duplicate work
}

/**
 * Phase 2: Layer-0 beam search with guided entry points.
 *
 * One thread block per query. Uses warp-cooperative distance computation
 * for coalesced memory access (each warp computes one distance in parallel).
 *
 * Shared memory holds:
 *   - Result buffer: sorted (id, dist) pairs of the best `ef` candidates
 *   - is_expanded: per-slot flags tracking which result entries have been expanded
 *   - Staging buffer: newly computed (id, dist) candidates from current iteration
 *   - Hash table: visited node tracking
 *   - Parent buffer: nodes to expand in current iteration
 *   - Metadata: counters
 *
 * Flow per iteration:
 *   1. Thread 0 selects top `search_width` unexpanded candidates as parents
 *   2. Warps cooperatively expand parents' neighbors (warp-parallel distance)
 *   3. Thread 0 merges staging buffer into result buffer (sorted insert)
 *   4. Repeat until no new parents or max_iterations reached
 */
__global__ void layer0_beam_search_kernel(
  const float* __restrict__ d_queries,
  const float* __restrict__ d_dataset,
  const uint32_t* __restrict__ d_layer0_graph,
  const uint32_t* __restrict__ d_entry_points,
  uint64_t* __restrict__ d_neighbors,
  float* __restrict__ d_distances,
  int num_queries,
  int N,
  int dim,
  int max_degree0,
  int k,
  int ef,
  int search_width,
  int max_iterations,
  bool use_inner_product)
{
  int query_idx = blockIdx.x;
  if (query_idx >= num_queries) return;

  const int warp_id = threadIdx.x / 32;
  const int lane    = threadIdx.x % 32;
  const int num_warps = blockDim.x / 32;

  const float* query = d_queries + static_cast<int64_t>(query_idx) * dim;

  // --- Shared memory layout ---
  extern __shared__ char smem[];

  // Hash table size: next power of 2 >= 8*ef, minimum 512
  uint32_t hash_sz = 512;
  while (hash_sz < static_cast<uint32_t>(8 * ef)) hash_sz <<= 1;
  uint32_t hash_mask = hash_sz - 1;

  // Max staging buffer size (bounded by search_width * max_degree0 unique new nodes per iter)
  int max_staging = search_width * max_degree0;

  // Shared memory partitioning:
  char* ptr = smem;
  uint32_t* result_ids    = reinterpret_cast<uint32_t*>(ptr);  ptr += ef * sizeof(uint32_t);
  float* result_dists     = reinterpret_cast<float*>(ptr);     ptr += ef * sizeof(float);
  uint8_t* is_expanded    = reinterpret_cast<uint8_t*>(ptr);   ptr += ef * sizeof(uint8_t);
  // Align to 4 bytes
  ptr = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(ptr) + 3) & ~3ull);
  uint32_t* staging_ids   = reinterpret_cast<uint32_t*>(ptr);  ptr += max_staging * sizeof(uint32_t);
  float* staging_dists    = reinterpret_cast<float*>(ptr);     ptr += max_staging * sizeof(float);
  uint32_t* hash_table    = reinterpret_cast<uint32_t*>(ptr);  ptr += hash_sz * sizeof(uint32_t);
  uint32_t* parent_ids    = reinterpret_cast<uint32_t*>(ptr);  ptr += search_width * sizeof(uint32_t);
  int* meta               = reinterpret_cast<int*>(ptr);       // [3]: result_count, staging_count, num_parents

  // Initialize
  for (int i = threadIdx.x; i < ef; i += blockDim.x) {
    result_ids[i]   = UINT32_MAX;
    result_dists[i] = FLT_MAX;
    is_expanded[i]  = 0;
  }
  for (uint32_t i = threadIdx.x; i < hash_sz; i += blockDim.x) {
    hash_table[i] = UINT32_MAX;
  }
  if (threadIdx.x == 0) {
    meta[0] = 0;  // result_count
    meta[1] = 0;  // staging_count
    meta[2] = 0;  // num_parents
  }
  __syncthreads();

  // --- Seed with entry point (warp 0 computes distance cooperatively) ---
  uint32_t ep = d_entry_points[query_idx];
  float ep_dist;
  if (warp_id == 0) {
    if (use_inner_product) {
      ep_dist = warp_ip_distance(query, d_dataset + static_cast<int64_t>(ep) * dim, dim);
    } else {
      ep_dist = warp_l2_distance(query, d_dataset + static_cast<int64_t>(ep) * dim, dim);
    }
    if (lane == 0) {
      result_ids[0]   = ep;
      result_dists[0] = ep_dist;
      is_expanded[0]  = 1;  // Will expand immediately below
      meta[0] = 1;
      hash_insert(hash_table, hash_mask, ep);
    }
  }
  __syncthreads();

  // --- Seed with entry point's neighbors (warp-cooperative distance) ---
  if (threadIdx.x == 0) meta[1] = 0;
  __syncthreads();

  // Each warp processes one neighbor at a time
  for (int j = warp_id; j < max_degree0; j += num_warps) {
    uint32_t nbr = d_layer0_graph[static_cast<int64_t>(ep) * max_degree0 + j];

    // Lane 0 does validity + hash check
    bool is_new = false;
    if (lane == 0) {
      if (nbr != UINT32_MAX && nbr < static_cast<uint32_t>(N)) {
        is_new = hash_insert(hash_table, hash_mask, nbr);
      }
    }
    // Broadcast hash result and nbr to all lanes in warp
    int is_new_int = is_new ? 1 : 0;
    is_new_int = __shfl_sync(0xffffffff, is_new_int, 0);
    nbr = __shfl_sync(0xffffffff, nbr, 0);
    if (!is_new_int) continue;

    // Warp-cooperative distance
    float dist;
    if (use_inner_product) {
      dist = warp_ip_distance(query, d_dataset + static_cast<int64_t>(nbr) * dim, dim);
    } else {
      dist = warp_l2_distance(query, d_dataset + static_cast<int64_t>(nbr) * dim, dim);
    }

    // Lane 0 writes to staging
    if (lane == 0) {
      int slot = atomicAdd(&meta[1], 1);
      if (slot < max_staging) {
        staging_ids[slot]   = nbr;
        staging_dists[slot] = dist;
      }
    }
  }
  __syncthreads();

  // Merge staging into result buffer (thread 0, shifts is_expanded with data)
  if (threadIdx.x == 0) {
    int staging_count = min(meta[1], max_staging);
    int rc = meta[0];
    for (int s = 0; s < staging_count; s++) {
      uint32_t sid = staging_ids[s];
      float sdist  = staging_dists[s];
      if (rc >= ef && sdist >= result_dists[rc - 1]) continue;

      int lo = 0, hi = rc;
      while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (result_dists[mid] < sdist) lo = mid + 1;
        else hi = mid;
      }
      int insert_end = rc < ef ? rc : ef - 1;
      for (int i = insert_end; i > lo; i--) {
        result_ids[i]   = result_ids[i - 1];
        result_dists[i] = result_dists[i - 1];
        is_expanded[i]  = is_expanded[i - 1];
      }
      result_ids[lo]   = sid;
      result_dists[lo] = sdist;
      is_expanded[lo]  = 0;  // New candidate, not yet expanded
      if (rc < ef) rc++;
    }
    meta[0] = rc;
  }
  __syncthreads();

  // --- Main beam search loop ---
  for (int iter = 0; iter < max_iterations; iter++) {
    // Step 1: Thread 0 selects parents (next search_width unexpanded candidates)
    if (threadIdx.x == 0) {
      int num_parents = 0;
      int rc = meta[0];
      for (int i = 0; i < rc && num_parents < search_width; i++) {
        if (!is_expanded[i]) {
          parent_ids[num_parents++] = result_ids[i];
          is_expanded[i] = 1;
        }
      }
      meta[2] = num_parents;
    }
    __syncthreads();

    int num_parents = meta[2];
    if (num_parents == 0) break;  // Converged: no more candidates to expand

    // Step 2: Warps cooperatively expand parents' neighbors
    if (threadIdx.x == 0) meta[1] = 0;
    __syncthreads();

    int total_work = num_parents * max_degree0;
    for (int wi = warp_id; wi < total_work; wi += num_warps) {
      int parent_idx = wi / max_degree0;
      int nbr_slot   = wi % max_degree0;

      uint32_t parent = parent_ids[parent_idx];
      uint32_t nbr = d_layer0_graph[static_cast<int64_t>(parent) * max_degree0 + nbr_slot];

      // Lane 0 does validity + hash check
      bool is_new = false;
      if (lane == 0) {
        if (nbr != UINT32_MAX && nbr < static_cast<uint32_t>(N)) {
          is_new = hash_insert(hash_table, hash_mask, nbr);
        }
      }
      int is_new_int = is_new ? 1 : 0;
      is_new_int = __shfl_sync(0xffffffff, is_new_int, 0);
      nbr = __shfl_sync(0xffffffff, nbr, 0);
      if (!is_new_int) continue;

      // Warp-cooperative distance computation (coalesced reads)
      float dist;
      if (use_inner_product) {
        dist = warp_ip_distance(query, d_dataset + static_cast<int64_t>(nbr) * dim, dim);
      } else {
        dist = warp_l2_distance(query, d_dataset + static_cast<int64_t>(nbr) * dim, dim);
      }

      // Lane 0 writes to staging buffer
      if (lane == 0) {
        int slot = atomicAdd(&meta[1], 1);
        if (slot < max_staging) {
          staging_ids[slot]   = nbr;
          staging_dists[slot] = dist;
        }
      }
    }
    __syncthreads();

    // Step 3: Thread 0 merges staging buffer into result buffer
    if (threadIdx.x == 0) {
      int staging_count = min(meta[1], max_staging);
      int rc = meta[0];
      for (int s = 0; s < staging_count; s++) {
        uint32_t sid = staging_ids[s];
        float sdist  = staging_dists[s];
        if (rc >= ef && sdist >= result_dists[rc - 1]) continue;

        int lo = 0, hi = rc;
        while (lo < hi) {
          int mid = (lo + hi) / 2;
          if (result_dists[mid] < sdist) lo = mid + 1;
          else hi = mid;
        }
        int insert_end = rc < ef ? rc : ef - 1;
        for (int i = insert_end; i > lo; i--) {
          result_ids[i]   = result_ids[i - 1];
          result_dists[i] = result_dists[i - 1];
          is_expanded[i]  = is_expanded[i - 1];
        }
        result_ids[lo]   = sid;
        result_dists[lo] = sdist;
        is_expanded[lo]  = 0;
        if (rc < ef) rc++;
      }
      meta[0] = rc;
    }
    __syncthreads();
  }

  // --- Copy top-k results to global memory ---
  int rc = meta[0];
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    if (i < rc) {
      d_neighbors[static_cast<int64_t>(query_idx) * k + i] = static_cast<uint64_t>(result_ids[i]);
      d_distances[static_cast<int64_t>(query_idx) * k + i] = result_dists[i];
    } else {
      d_neighbors[static_cast<int64_t>(query_idx) * k + i] = UINT64_MAX;
      d_distances[static_cast<int64_t>(query_idx) * k + i] = FLT_MAX;
    }
  }
}

/**
 * Calculate shared memory size needed for layer0_beam_search_kernel.
 */
inline size_t calc_layer0_smem_size(int ef, int search_width, int max_degree0)
{
  uint32_t hash_sz = 512;
  while (hash_sz < static_cast<uint32_t>(8 * ef)) hash_sz <<= 1;
  int max_staging = search_width * max_degree0;

  size_t size = 0;
  size += ef * sizeof(uint32_t);         // result_ids
  size += ef * sizeof(float);            // result_dists
  size += ef * sizeof(uint8_t);          // is_expanded flags
  size = (size + 3) & ~3ull;             // align to 4 bytes
  size += max_staging * sizeof(uint32_t); // staging_ids
  size += max_staging * sizeof(float);    // staging_dists
  size += hash_sz * sizeof(uint32_t);    // hash_table
  size += search_width * sizeof(uint32_t); // parent_ids
  size += 3 * sizeof(int);              // meta
  return size;
}

}  // namespace cuvs::neighbors::gpu_hnsw::detail
