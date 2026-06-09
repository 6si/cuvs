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

/**
 * Single-thread distance computation (for use in block-parallel patterns
 * where each thread handles one candidate).
 */
__device__ __forceinline__ float thread_l2_distance(const float* __restrict__ a,
                                                    const float* __restrict__ b,
                                                    int dim)
{
  float sum = 0.0f;
  for (int d = 0; d < dim; d++) {
    float diff = a[d] - b[d];
    sum += diff * diff;
  }
  return sum;
}

__device__ __forceinline__ float thread_ip_distance(const float* __restrict__ a,
                                                    const float* __restrict__ b,
                                                    int dim)
{
  float sum = 0.0f;
  for (int d = 0; d < dim; d++) {
    sum += a[d] * b[d];
  }
  return -sum;
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
  return true;  // Hash full, accept as new to avoid infinite loop
}

/**
 * Phase 2: Layer-0 beam search with guided entry points.
 *
 * One thread block per query. Shared memory holds:
 *   - Result buffer: sorted (id, dist) pairs of the best `ef` candidates
 *   - Staging buffer: newly computed (id, dist) candidates from current iteration
 *   - Hash table: visited node tracking
 *   - Parent buffer: nodes to expand in current iteration
 *   - Metadata: counters
 *
 * Flow per iteration:
 *   1. Thread 0 selects top `search_width` unexpanded candidates as parents
 *   2. All threads expand parents' neighbors in parallel, compute distances,
 *      write new candidates to staging buffer
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

  const float* query = d_queries + static_cast<int64_t>(query_idx) * dim;

  // --- Shared memory layout ---
  // We allocate dynamically and partition manually.
  extern __shared__ char smem[];

  // Hash table size: next power of 2 >= 8*ef, minimum 512
  uint32_t hash_sz = 512;
  while (hash_sz < static_cast<uint32_t>(8 * ef)) hash_sz <<= 1;
  uint32_t hash_mask = hash_sz - 1;

  // Max staging buffer size (number of new candidates per iteration)
  int max_staging = search_width * max_degree0;

  // Shared memory partitioning:
  uint32_t* result_ids    = reinterpret_cast<uint32_t*>(smem);                   // [ef]
  float* result_dists     = reinterpret_cast<float*>(result_ids + ef);           // [ef]
  uint32_t* staging_ids   = reinterpret_cast<uint32_t*>(result_dists + ef);      // [max_staging]
  float* staging_dists    = reinterpret_cast<float*>(staging_ids + max_staging);  // [max_staging]
  uint32_t* hash_table    = reinterpret_cast<uint32_t*>(staging_dists + max_staging);  // [hash_sz]
  uint32_t* parent_ids    = hash_table + hash_sz;                                // [search_width]
  int* meta               = reinterpret_cast<int*>(parent_ids + search_width);   // [3]: result_count, staging_count, num_parents

  // Initialize
  for (int i = threadIdx.x; i < ef; i += blockDim.x) {
    result_ids[i]  = UINT32_MAX;
    result_dists[i] = FLT_MAX;
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

  // --- Seed with entry point ---
  uint32_t ep = d_entry_points[query_idx];
  if (threadIdx.x == 0) {
    float ep_dist;
    if (use_inner_product) {
      ep_dist = thread_ip_distance(query, d_dataset + static_cast<int64_t>(ep) * dim, dim);
    } else {
      ep_dist = thread_l2_distance(query, d_dataset + static_cast<int64_t>(ep) * dim, dim);
    }
    result_ids[0]  = ep;
    result_dists[0] = ep_dist;
    meta[0] = 1;  // result_count = 1
    hash_insert(hash_table, hash_mask, ep);
  }
  __syncthreads();

  // --- Seed with entry point's neighbors ---
  if (threadIdx.x == 0) meta[1] = 0;  // staging_count = 0
  __syncthreads();

  for (int j = threadIdx.x; j < max_degree0; j += blockDim.x) {
    uint32_t nbr = d_layer0_graph[static_cast<int64_t>(ep) * max_degree0 + j];
    if (nbr == UINT32_MAX || nbr >= static_cast<uint32_t>(N)) continue;
    if (!hash_insert(hash_table, hash_mask, nbr)) continue;

    float dist;
    if (use_inner_product) {
      dist = thread_ip_distance(query, d_dataset + static_cast<int64_t>(nbr) * dim, dim);
    } else {
      dist = thread_l2_distance(query, d_dataset + static_cast<int64_t>(nbr) * dim, dim);
    }

    int slot = atomicAdd(&meta[1], 1);
    if (slot < max_staging) {
      staging_ids[slot]  = nbr;
      staging_dists[slot] = dist;
    }
  }
  __syncthreads();

  // Merge staging into result buffer (thread 0)
  if (threadIdx.x == 0) {
    int staging_count = min(meta[1], max_staging);
    int rc = meta[0];
    for (int s = 0; s < staging_count; s++) {
      uint32_t sid  = staging_ids[s];
      float sdist   = staging_dists[s];

      // Skip if worse than worst in full buffer
      if (rc >= ef && sdist >= result_dists[rc - 1]) continue;

      // Find insertion position (binary search)
      int lo = 0, hi = rc;
      while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (result_dists[mid] < sdist) lo = mid + 1;
        else hi = mid;
      }

      // Shift elements right
      int insert_end = rc < ef ? rc : ef - 1;
      for (int i = insert_end; i > lo; i--) {
        result_ids[i]  = result_ids[i - 1];
        result_dists[i] = result_dists[i - 1];
      }
      result_ids[lo]  = sid;
      result_dists[lo] = sdist;
      if (rc < ef) rc++;
    }
    meta[0] = rc;
  }
  __syncthreads();

  // --- Main beam search loop ---
  int expanded = 1;  // We've already expanded the entry point
  for (int iter = 0; iter < max_iterations; iter++) {
    // Step 1: Thread 0 selects parents (next search_width unexpanded candidates)
    if (threadIdx.x == 0) {
      int num_parents = 0;
      int rc = meta[0];
      for (int i = expanded; i < rc && num_parents < search_width; i++) {
        parent_ids[num_parents++] = result_ids[i];
      }
      meta[2] = num_parents;
      expanded += num_parents;
    }
    __syncthreads();

    int num_parents = meta[2];
    if (num_parents == 0) break;  // Converged: no more candidates to expand

    // Step 2: Expand parents' neighbors in parallel
    if (threadIdx.x == 0) meta[1] = 0;  // Reset staging_count
    __syncthreads();

    int total_work = num_parents * max_degree0;
    for (int wi = threadIdx.x; wi < total_work; wi += blockDim.x) {
      int parent_idx = wi / max_degree0;
      int nbr_slot   = wi % max_degree0;

      uint32_t parent = parent_ids[parent_idx];
      uint32_t nbr = d_layer0_graph[static_cast<int64_t>(parent) * max_degree0 + nbr_slot];
      if (nbr == UINT32_MAX || nbr >= static_cast<uint32_t>(N)) continue;
      if (!hash_insert(hash_table, hash_mask, nbr)) continue;

      // Early rejection: check against current worst in result buffer
      int rc = meta[0];
      if (rc >= ef) {
        float worst = result_dists[rc - 1];
        // Can't reject without computing distance, but we can skip if we have
        // a hint. For now, compute distance for all candidates.
      }

      float dist;
      if (use_inner_product) {
        dist = thread_ip_distance(query, d_dataset + static_cast<int64_t>(nbr) * dim, dim);
      } else {
        dist = thread_l2_distance(query, d_dataset + static_cast<int64_t>(nbr) * dim, dim);
      }

      int slot = atomicAdd(&meta[1], 1);
      if (slot < max_staging) {
        staging_ids[slot]  = nbr;
        staging_dists[slot] = dist;
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
          result_ids[i]  = result_ids[i - 1];
          result_dists[i] = result_dists[i - 1];
        }
        result_ids[lo]  = sid;
        result_dists[lo] = sdist;
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
  size += max_staging * sizeof(uint32_t); // staging_ids
  size += max_staging * sizeof(float);    // staging_dists
  size += hash_sz * sizeof(uint32_t);    // hash_table
  size += search_width * sizeof(uint32_t); // parent_ids
  size += 3 * sizeof(int);              // meta
  return size;
}

}  // namespace cuvs::neighbors::gpu_hnsw::detail
