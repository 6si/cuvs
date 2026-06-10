/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/gpu_hnsw.hpp>
#include <cuvs/neighbors/hnsw.hpp>

#include <hnswlib/hnswalg.h>
#include <hnswlib/hnswlib.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace cuvs::neighbors::gpu_hnsw::detail {

/**
 * @brief Extract the multi-layer HNSW graph from an hnswlib HierarchicalNSW
 *        into GPU-friendly dense arrays (one per layer).
 *
 * The hnswlib internal format stores each node's links as:
 *   Layer 0: data_level0_memory_ [max_elements * size_data_per_element_]
 *     offset 0: uint32_t link_count
 *     offset 4: uint32_t neighbor_ids[maxM0_]
 *     offset ...: vector data + label
 *
 *   Upper layers: linkLists_[node_id], size = size_links_per_element_ * level
 *     Each layer block: uint32_t link_count, uint32_t neighbor_ids[maxM_]
 *
 * We extract these into:
 *   - layer 0: dense array [N x maxM0_], all N nodes
 *   - layer L (L>0): node_ids + dense array [num_nodes_at_L x maxM_]
 */
template <typename DistT>
void extract_hnsw_layers(const hnswlib::HierarchicalNSW<DistT>& hnsw_alg,
                         std::vector<gpu_layer>& layers,
                         uint32_t& entry_point,
                         int& M,
                         int& max_degree0)
{
  const size_t N         = hnsw_alg.cur_element_count;
  const int maxM         = hnsw_alg.maxM_;
  const int maxM0        = hnsw_alg.maxM0_;
  const int max_level    = hnsw_alg.maxlevel_;
  entry_point            = static_cast<uint32_t>(hnsw_alg.enterpoint_node_);
  M                      = hnsw_alg.M_;
  max_degree0            = maxM0;

  const int num_layers = max_level + 1;
  layers.resize(num_layers);

  // --- Layer 0: dense [N x maxM0] ---
  {
    auto& L0     = layers[0];
    L0.num_nodes = static_cast<uint32_t>(N);
    L0.max_degree = maxM0;
    L0.neighbor_ids.resize(N * maxM0, UINT32_MAX);
    // node_ids is empty for layer 0 (implicit identity)

    for (size_t i = 0; i < N; i++) {
      // Links are stored before the data in data_level0_memory_. Use the hnswlib API directly.
      // data_level0_memory_ layout per element:
      //   [0..size_links_level0_) = link list for layer 0
      //   [size_links_level0_..size_data_per_element_) = data + label
      // where size_links_level0_ = (maxM0_ + 1) * sizeof(uint32_t)
      // link list: [link_count, neighbor_0, neighbor_1, ...]

      const char* level0_data = hnsw_alg.data_level0_memory_ + i * hnsw_alg.size_data_per_element_;
      uint32_t link_count;
      std::memcpy(&link_count, level0_data, sizeof(uint32_t));
      link_count = std::min(link_count, static_cast<uint32_t>(maxM0));

      const uint32_t* links = reinterpret_cast<const uint32_t*>(level0_data + sizeof(uint32_t));
      for (uint32_t j = 0; j < link_count; j++) {
        L0.neighbor_ids[i * maxM0 + j] = links[j];
      }
      // Remaining slots already UINT32_MAX from resize
    }
  }

  // --- Upper layers (1 .. max_level): sparse ---
  for (int layer = 1; layer < num_layers; layer++) {
    auto& UL     = layers[layer];
    UL.max_degree = maxM;

    // First pass: find which nodes exist at this layer
    std::vector<uint32_t> nodes_at_layer;
    for (size_t i = 0; i < N; i++) {
      int node_level = hnsw_alg.element_levels_[i];
      if (node_level >= layer) {
        nodes_at_layer.push_back(static_cast<uint32_t>(i));
      }
    }

    UL.num_nodes = static_cast<uint32_t>(nodes_at_layer.size());
    UL.node_ids  = std::move(nodes_at_layer);
    UL.neighbor_ids.resize(UL.num_nodes * maxM, UINT32_MAX);

    // Second pass: extract neighbor lists
    for (uint32_t idx = 0; idx < UL.num_nodes; idx++) {
      uint32_t node_id = UL.node_ids[idx];

      // linkLists_[node_id] points to a buffer of size_links_per_element_ * level bytes
      // For layer L, the offset is (L-1) * size_links_per_element_
      // size_links_per_element_ = (maxM_ + 1) * sizeof(uint32_t)
      const char* link_data = hnsw_alg.linkLists_[node_id] +
                              (layer - 1) * hnsw_alg.size_links_per_element_;

      uint32_t link_count;
      std::memcpy(&link_count, link_data, sizeof(uint32_t));
      link_count = std::min(link_count, static_cast<uint32_t>(maxM));

      const uint32_t* links = reinterpret_cast<const uint32_t*>(link_data + sizeof(uint32_t));
      for (uint32_t j = 0; j < link_count; j++) {
        UL.neighbor_ids[idx * maxM + j] = links[j];
      }
    }
  }
}

}  // namespace cuvs::neighbors::gpu_hnsw::detail
