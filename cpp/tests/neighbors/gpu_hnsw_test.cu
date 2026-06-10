/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file gpu_hnsw_test.cu
 * @brief Test for GPU HNSW search.
 *
 * This test:
 *   1. Builds a CPU HNSW index directly using hnswlib (full hierarchy)
 *   2. Wraps it in a cuvs::neighbors::hnsw::index via index_impl
 *   3. Converts it to GPU HNSW index using cuvs::neighbors::gpu_hnsw::from_hnsw_index
 *   4. Runs GPU HNSW search
 *   5. Compares recall against brute-force ground truth
 *
 * We build hnswlib directly instead of going through hnsw::build() because
 * hnsw::build() routes through CAGRA → from_cagra, and the HnswHierarchy::CPU
 * path in from_cagra hangs. Direct hnswlib build is also what Milvus/Knowhere
 * does in production, so this better mirrors the real integration.
 */

#ifdef CUVS_BUILD_CAGRA_HNSWLIB

// Suppress hnswlib sign-conversion warnings under nvcc (-Werror treats #68-D as error)
#ifdef __NVCC__
#pragma nv_diag_suppress 68
#endif

#include <gtest/gtest.h>

#include <cuvs/neighbors/gpu_hnsw.hpp>
#include <cuvs/neighbors/hnsw.hpp>
#include <cuvs/neighbors/brute_force.hpp>

// Internal detail header to access index_impl (needed to wrap raw hnswlib index)
#include "../../src/neighbors/detail/hnsw.hpp"
#include <hnswlib/hnswalg.h>
#include <hnswlib/hnswlib.h>

#ifdef __NVCC__
#pragma nv_diag_default 68
#endif

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng.cuh>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace {

/**
 * Compute recall@k: fraction of true k-NN that appear in the result set.
 */
float compute_recall(const std::vector<int64_t>& gt_neighbors,
                     const std::vector<uint64_t>& test_neighbors,
                     int n_queries,
                     int k)
{
  int total_correct = 0;
  for (int q = 0; q < n_queries; q++) {
    for (int i = 0; i < k; i++) {
      int64_t test_id = static_cast<int64_t>(test_neighbors[q * k + i]);
      for (int j = 0; j < k; j++) {
        if (gt_neighbors[q * k + j] == test_id) {
          total_correct++;
          break;
        }
      }
    }
  }
  return static_cast<float>(total_correct) / (n_queries * k);
}

}  // namespace

class GpuHnswSearchTest : public ::testing::TestWithParam<
                            std::tuple<int, int, int, cuvs::distance::DistanceType>> {
 protected:
  void SetUp() override
  {
    auto [n_rows, dim, k, metric] = GetParam();
    n_rows_  = n_rows;
    dim_     = dim;
    k_       = k;
    metric_  = metric;
    n_queries_ = 100;
  }

  int n_rows_;
  int dim_;
  int k_;
  int n_queries_;
  cuvs::distance::DistanceType metric_;
};

TEST_P(GpuHnswSearchTest, RecallTest)
{
  raft::resources res;

  // Generate random dataset
  auto h_dataset = raft::make_host_matrix<float, int64_t>(n_rows_, dim_);
  {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int64_t i = 0; i < n_rows_ * dim_; i++) {
      h_dataset.data_handle()[i] = dist(rng);
    }
  }

  // Generate random queries
  auto h_queries = raft::make_host_matrix<float, int64_t>(n_queries_, dim_);
  {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int64_t i = 0; i < n_queries_ * dim_; i++) {
      h_queries.data_handle()[i] = dist(rng);
    }
  }

  // --- Step 1: Build CPU HNSW index directly via hnswlib (full hierarchy) ---
  // We build hnswlib directly instead of hnsw::build() to avoid the CAGRA→from_cagra
  // pipeline (which hangs with HnswHierarchy::CPU). This also matches the Milvus
  // production flow where hnswlib builds the index natively.
  constexpr int M = 32;
  constexpr int ef_construction = 200;

  auto hnsw_index = std::make_unique<cuvs::neighbors::hnsw::detail::index_impl<float>>(
    dim_, metric_, cuvs::neighbors::hnsw::HnswHierarchy::CPU);

  auto hnsw_alg = std::make_unique<hnswlib::HierarchicalNSW<float>>(
    hnsw_index->get_space(), n_rows_, M, ef_construction);

  // Insert all vectors (sequential — hnswlib addPoint is not fully thread-safe
  // for concurrent inserts at small N, and we're testing correctness not build speed)
  for (int64_t i = 0; i < n_rows_; i++) {
    hnsw_alg->addPoint(
      static_cast<const void*>(h_dataset.data_handle() + i * dim_), i);
  }

  hnsw_index->set_index(std::move(hnsw_alg));
  ASSERT_NE(hnsw_index, nullptr);

  // --- Step 2: Convert to GPU HNSW index ---
  auto gpu_idx = cuvs::neighbors::gpu_hnsw::from_hnsw_index<float>(
    res, *hnsw_index, raft::make_const_mdspan(h_dataset.view()));
  ASSERT_NE(gpu_idx, nullptr);
  EXPECT_EQ(gpu_idx->n_rows(), n_rows_);
  EXPECT_EQ(gpu_idx->dim(), dim_);
  EXPECT_GT(gpu_idx->num_layers(), 0);

  // --- Step 3: Upload queries to device ---
  auto d_queries = raft::make_device_matrix<float, int64_t>(res, n_queries_, dim_);
  raft::copy(d_queries.data_handle(), h_queries.data_handle(),
             n_queries_ * dim_, raft::resource::get_cuda_stream(res));

  // --- Step 4: GPU HNSW search ---
  auto d_neighbors = raft::make_device_matrix<uint64_t, int64_t>(res, n_queries_, k_);
  auto d_distances = raft::make_device_matrix<float, int64_t>(res, n_queries_, k_);

  cuvs::neighbors::gpu_hnsw::search_params search_params;
  search_params.ef = 200;
  search_params.search_width = 4;

  cuvs::neighbors::gpu_hnsw::search<float>(
    res, search_params, *gpu_idx,
    raft::make_const_mdspan(d_queries.view()),
    d_neighbors.view(),
    d_distances.view());

  // --- Step 5: Copy results to host ---
  std::vector<uint64_t> h_gpu_neighbors(n_queries_ * k_);
  std::vector<float> h_gpu_distances(n_queries_ * k_);
  raft::copy(h_gpu_neighbors.data(), d_neighbors.data_handle(),
             n_queries_ * k_, raft::resource::get_cuda_stream(res));
  raft::copy(h_gpu_distances.data(), d_distances.data_handle(),
             n_queries_ * k_, raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  // --- Step 6: Compute ground truth with brute force ---
  auto d_dataset = raft::make_device_matrix<float, int64_t>(res, n_rows_, dim_);
  raft::copy(d_dataset.data_handle(), h_dataset.data_handle(),
             n_rows_ * dim_, raft::resource::get_cuda_stream(res));

  auto d_gt_neighbors = raft::make_device_matrix<int64_t, int64_t>(res, n_queries_, k_);
  auto d_gt_distances = raft::make_device_matrix<float, int64_t>(res, n_queries_, k_);

  cuvs::neighbors::brute_force::index_params bf_index_params;
  bf_index_params.metric = metric_;
  auto bf_index = cuvs::neighbors::brute_force::build(
    res, bf_index_params, raft::make_const_mdspan(d_dataset.view()));
  cuvs::neighbors::brute_force::search_params bf_search_params;
  cuvs::neighbors::brute_force::search(
    res, bf_search_params, bf_index,
    raft::make_const_mdspan(d_queries.view()),
    d_gt_neighbors.view(),
    d_gt_distances.view());

  std::vector<int64_t> h_gt_neighbors(n_queries_ * k_);
  raft::copy(h_gt_neighbors.data(), d_gt_neighbors.data_handle(),
             n_queries_ * k_, raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  // --- Step 7: Check recall ---
  float recall = compute_recall(h_gt_neighbors, h_gpu_neighbors, n_queries_, k_);

  // At small scale (50K vectors), GPU HNSW should achieve >90% recall with ef=200
  EXPECT_GT(recall, 0.90f)
    << "Recall@" << k_ << " = " << recall << " (expected > 0.90)"
    << " with n_rows=" << n_rows_ << " dim=" << dim_;
}

INSTANTIATE_TEST_SUITE_P(
  GpuHnswSearch,
  GpuHnswSearchTest,
  ::testing::Values(
    // (n_rows, dim, k, metric)
    std::make_tuple(10000, 128, 10, cuvs::distance::DistanceType::L2Expanded),
    std::make_tuple(50000, 384, 10, cuvs::distance::DistanceType::L2Expanded),
    std::make_tuple(10000, 128, 10, cuvs::distance::DistanceType::InnerProduct)
  ));

#endif  // CUVS_BUILD_CAGRA_HNSWLIB
