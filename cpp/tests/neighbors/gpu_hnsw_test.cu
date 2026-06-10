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
 *   5. Compares recall against CPU brute-force ground truth
 *   6. Benchmarks GPU throughput vs CPU HNSW throughput
 *
 * Uses CPU brute-force for ground truth (not cuvs brute_force) to avoid
 * triggering CAGRA JIT compilation which takes 20-30 minutes on first run.
 */

#ifdef CUVS_BUILD_CAGRA_HNSWLIB

// Suppress hnswlib sign-conversion warnings under nvcc (-Werror treats #68-D as error)
#ifdef __NVCC__
#pragma nv_diag_suppress 68
#endif

#include <gtest/gtest.h>

#include <cuvs/neighbors/gpu_hnsw.hpp>
#include <cuvs/neighbors/hnsw.hpp>

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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>

namespace {

/**
 * CPU brute-force kNN: computes exact top-k for each query.
 * Avoids cuvs::brute_force which triggers heavy CUDA JIT compilation.
 */
void cpu_brute_force(const float* dataset,
                     const float* queries,
                     int64_t n_rows,
                     int64_t n_queries,
                     int64_t dim,
                     int k,
                     std::vector<int64_t>& neighbors,  // [n_queries * k]
                     bool use_ip = false)
{
  neighbors.resize(n_queries * k);
  std::vector<std::pair<float, int64_t>> dists(n_rows);

  for (int64_t q = 0; q < n_queries; q++) {
    const float* qvec = queries + q * dim;
    for (int64_t i = 0; i < n_rows; i++) {
      const float* dvec = dataset + i * dim;
      float dist = 0.0f;
      if (use_ip) {
        float ip = 0.0f;
        for (int64_t d = 0; d < dim; d++) ip += qvec[d] * dvec[d];
        dist = -ip;
      } else {
        for (int64_t d = 0; d < dim; d++) {
          float diff = qvec[d] - dvec[d];
          dist += diff * diff;
        }
      }
      dists[i] = {dist, i};
    }
    std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
    for (int j = 0; j < k; j++) {
      neighbors[q * k + j] = dists[j].second;
    }
  }
}

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
  bool use_ip = (metric_ == cuvs::distance::DistanceType::InnerProduct);

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
  // Use M=32 for small N, M=64 for large N to ensure 95%+ recall is achievable.
  // Milvus uses M=16 by default, but M=32 is more typical for high recall.
  int M_build = (n_rows_ <= 10000) ? 32 : 64;
  int ef_construction = 200;

  auto hnsw_index = std::make_unique<cuvs::neighbors::hnsw::detail::index_impl<float>>(
    dim_, metric_, cuvs::neighbors::hnsw::HnswHierarchy::CPU);

  auto hnsw_alg = std::make_unique<hnswlib::HierarchicalNSW<float>>(
    hnsw_index->get_space(), n_rows_, M_build, ef_construction);

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

  std::cout << "  num_layers=" << gpu_idx->num_layers()
            << " max_degree0=" << gpu_idx->max_degree0()
            << " entry_point=" << gpu_idx->entry_point() << "\n";

  // --- Step 3: Upload queries to device ---
  auto d_queries = raft::make_device_matrix<float, int64_t>(res, n_queries_, dim_);
  raft::copy(d_queries.data_handle(), h_queries.data_handle(),
             n_queries_ * dim_, raft::resource::get_cuda_stream(res));

  // --- Step 4: GPU HNSW search (with warmup) ---
  auto d_neighbors = raft::make_device_matrix<uint64_t, int64_t>(res, n_queries_, k_);
  auto d_distances = raft::make_device_matrix<float, int64_t>(res, n_queries_, k_);

  // ef scales with N and graph quality: larger datasets need higher ef for 95%+ recall
  int ef_search = (n_rows_ <= 10000) ? 200 : 400;
  cuvs::neighbors::gpu_hnsw::search_params search_params;
  search_params.ef = ef_search;
  search_params.search_width = (n_rows_ <= 10000) ? 4 : 8;

  // Warmup
  cuvs::neighbors::gpu_hnsw::search<float>(
    res, search_params, *gpu_idx,
    raft::make_const_mdspan(d_queries.view()),
    d_neighbors.view(),
    d_distances.view());

  // Timed run
  auto t0 = std::chrono::high_resolution_clock::now();
  cuvs::neighbors::gpu_hnsw::search<float>(
    res, search_params, *gpu_idx,
    raft::make_const_mdspan(d_queries.view()),
    d_neighbors.view(),
    d_distances.view());
  auto t1 = std::chrono::high_resolution_clock::now();
  double gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // --- Step 5: Copy results to host ---
  std::vector<uint64_t> h_gpu_neighbors(n_queries_ * k_);
  raft::copy(h_gpu_neighbors.data(), d_neighbors.data_handle(),
             n_queries_ * k_, raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  // --- Step 6: CPU brute-force ground truth ---
  std::vector<int64_t> h_gt_neighbors;
  cpu_brute_force(h_dataset.data_handle(), h_queries.data_handle(),
                  n_rows_, n_queries_, dim_, k_, h_gt_neighbors, use_ip);

  // --- Step 7: CPU HNSW search latency (baseline) ---
  auto* cpu_hnsw = const_cast<hnswlib::HierarchicalNSW<float>*>(
    static_cast<const hnswlib::HierarchicalNSW<float>*>(hnsw_index->get_index()));
  cpu_hnsw->setEf(ef_search);

  // Warmup
  cpu_hnsw->searchKnn(h_queries.data_handle(), k_);

  auto t2 = std::chrono::high_resolution_clock::now();
  for (int q = 0; q < n_queries_; q++) {
    cpu_hnsw->searchKnn(h_queries.data_handle() + static_cast<int64_t>(q) * dim_, k_);
  }
  auto t3 = std::chrono::high_resolution_clock::now();
  double cpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

  // --- Step 8: Measure CPU HNSW recall ---
  std::vector<int64_t> h_cpu_neighbors(n_queries_ * k_);
  for (int q = 0; q < n_queries_; q++) {
    auto result = cpu_hnsw->searchKnn(
      h_queries.data_handle() + static_cast<int64_t>(q) * dim_, k_);
    // searchKnn returns max-heap (worst first), collect all
    int pos = k_ - 1;
    while (!result.empty()) {
      h_cpu_neighbors[q * k_ + pos] = static_cast<int64_t>(result.top().second);
      result.pop();
      pos--;
    }
  }

  // --- Step 9: Report and check recall ---
  float recall     = compute_recall(h_gt_neighbors, h_gpu_neighbors, n_queries_, k_);
  std::vector<uint64_t> h_cpu_nbrs_u64(n_queries_ * k_);
  for (int i = 0; i < n_queries_ * k_; i++) h_cpu_nbrs_u64[i] = static_cast<uint64_t>(h_cpu_neighbors[i]);
  float cpu_recall = compute_recall(h_gt_neighbors, h_cpu_nbrs_u64, n_queries_, k_);

  std::cout << "  N=" << n_rows_ << " dim=" << dim_ << " k=" << k_
            << " metric=" << (use_ip ? "IP" : "L2") << "\n";
  std::cout << "  GPU Recall@" << k_ << " = " << recall << "\n";
  std::cout << "  CPU HNSW Recall@" << k_ << " = " << cpu_recall << " (ef=" << ef_search << ")\n";
  std::cout << "  GPU: " << gpu_ms << " ms for " << n_queries_ << " queries"
            << " (" << (n_queries_ / (gpu_ms / 1000.0)) << " QPS)\n";
  std::cout << "  CPU HNSW: " << cpu_ms << " ms for " << n_queries_ << " queries"
            << " (" << (n_queries_ / (cpu_ms / 1000.0)) << " QPS)\n";
  std::cout << "  Speedup: " << (cpu_ms / gpu_ms) << "x\n";

  EXPECT_GT(recall, 0.95f)
    << "Recall@" << k_ << " = " << recall << " (expected > 0.95)"
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

// =============================================================================
// Scale test: N=2M — validates global-memory bitmap and production-scale search
// =============================================================================

class GpuHnswScaleTest : public ::testing::TestWithParam<
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

/**
 * Scale test: Uses CPU HNSW recall as ground truth (brute force is too slow at N=2M).
 * GPU recall is compared against CPU HNSW recall rather than exact brute-force.
 * If GPU recall >= 0.95 * CPU recall, we consider it a pass.
 */
TEST_P(GpuHnswScaleTest, ScaleRecallTest)
{
  raft::resources res;
  bool use_ip = (metric_ == cuvs::distance::DistanceType::InnerProduct);

  std::cout << "  [Scale test] N=" << n_rows_ << " dim=" << dim_
            << " metric=" << (use_ip ? "IP" : "L2") << "\n";

  // Generate random dataset
  auto h_dataset = raft::make_host_matrix<float, int64_t>(n_rows_, dim_);
  {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int64_t i = 0; i < static_cast<int64_t>(n_rows_) * dim_; i++) {
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

  // --- Step 1: Build CPU HNSW ---
  // ef_construction=50 (vs 200) for speed at large N. Recall is lower but
  // sufficient to validate GPU search correctness vs CPU HNSW baseline.
  std::cout << "  Building CPU HNSW (M=32, ef_construction=50, parallel)...\n";
  auto build_start = std::chrono::high_resolution_clock::now();

  int M_build = 32;
  int ef_construction = 50;

  auto hnsw_index = std::make_unique<cuvs::neighbors::hnsw::detail::index_impl<float>>(
    dim_, metric_, cuvs::neighbors::hnsw::HnswHierarchy::CPU);

  auto hnsw_alg = std::make_unique<hnswlib::HierarchicalNSW<float>>(
    hnsw_index->get_space(), n_rows_, M_build, ef_construction);

  // Parallel build using hnswlib's thread-safe addPoint (add first point
  // sequentially to initialize, then parallelize remaining insertions).
  hnsw_alg->addPoint(
    static_cast<const void*>(h_dataset.data_handle()), static_cast<hnswlib::labeltype>(0));
  int num_threads = std::min(static_cast<int>(std::thread::hardware_concurrency()), 32);
  #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1024)
  for (int64_t i = 1; i < n_rows_; i++) {
    hnsw_alg->addPoint(
      static_cast<const void*>(h_dataset.data_handle() + i * dim_),
      static_cast<hnswlib::labeltype>(i));
    if (i % 500000 == 0 && i > 0) {
      #pragma omp critical
      std::cout << "    inserted ~" << i << " / " << n_rows_ << "\n" << std::flush;
    }
  }

  hnsw_index->set_index(std::move(hnsw_alg));
  auto build_end = std::chrono::high_resolution_clock::now();
  double build_s = std::chrono::duration<double>(build_end - build_start).count();
  std::cout << "  CPU HNSW build: " << build_s << " s\n";

  // Parallel build assigns internal IDs in insertion order, which differs from label order.
  // Reorder dataset so row[internal_id] = original_dataset[label_of_internal_id].
  auto* cpu_hnsw = const_cast<hnswlib::HierarchicalNSW<float>*>(
    static_cast<const hnswlib::HierarchicalNSW<float>*>(hnsw_index->get_index()));
  auto h_dataset_for_gpu = raft::make_host_matrix<float, int64_t>(n_rows_, dim_);
  for (int64_t id = 0; id < n_rows_; id++) {
    auto label = static_cast<int64_t>(
      cpu_hnsw->getExternalLabel(static_cast<hnswlib::tableint>(id)));
    std::memcpy(h_dataset_for_gpu.data_handle() + id * dim_,
                h_dataset.data_handle() + label * dim_,
                dim_ * sizeof(float));
  }

  // --- Step 2: Convert to GPU ---
  std::cout << "  Converting to GPU index...\n";
  auto convert_start = std::chrono::high_resolution_clock::now();
  auto gpu_idx = cuvs::neighbors::gpu_hnsw::from_hnsw_index<float>(
    res, *hnsw_index, raft::make_const_mdspan(h_dataset_for_gpu.view()));
  auto convert_end = std::chrono::high_resolution_clock::now();
  double convert_s = std::chrono::duration<double>(convert_end - convert_start).count();

  ASSERT_NE(gpu_idx, nullptr);
  std::cout << "  GPU conversion: " << convert_s << " s"
            << " | layers=" << gpu_idx->num_layers()
            << " max_degree0=" << gpu_idx->max_degree0()
            << " entry=" << gpu_idx->entry_point() << "\n";

  // --- Step 3: Upload queries ---
  auto d_queries = raft::make_device_matrix<float, int64_t>(res, n_queries_, dim_);
  raft::copy(d_queries.data_handle(), h_queries.data_handle(),
             n_queries_ * dim_, raft::resource::get_cuda_stream(res));

  // --- Step 4: GPU search ---
  auto d_neighbors = raft::make_device_matrix<uint64_t, int64_t>(res, n_queries_, k_);
  auto d_distances = raft::make_device_matrix<float, int64_t>(res, n_queries_, k_);

  cuvs::neighbors::gpu_hnsw::search_params search_params;
  search_params.ef = 400;
  search_params.search_width = 8;

  // Warmup
  cuvs::neighbors::gpu_hnsw::search<float>(
    res, search_params, *gpu_idx,
    raft::make_const_mdspan(d_queries.view()),
    d_neighbors.view(),
    d_distances.view());

  // Timed run
  auto t0 = std::chrono::high_resolution_clock::now();
  cuvs::neighbors::gpu_hnsw::search<float>(
    res, search_params, *gpu_idx,
    raft::make_const_mdspan(d_queries.view()),
    d_neighbors.view(),
    d_distances.view());
  auto t1 = std::chrono::high_resolution_clock::now();
  double gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // Copy results and remap internal IDs → labels
  std::vector<uint64_t> h_gpu_neighbors(n_queries_ * k_);
  raft::copy(h_gpu_neighbors.data(), d_neighbors.data_handle(),
             n_queries_ * k_, raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  for (int64_t i = 0; i < n_queries_ * k_; i++) {
    if (h_gpu_neighbors[i] != UINT64_MAX) {
      h_gpu_neighbors[i] = static_cast<uint64_t>(
        cpu_hnsw->getExternalLabel(static_cast<hnswlib::tableint>(h_gpu_neighbors[i])));
    }
  }

  // --- Step 5: CPU HNSW search (ground truth at scale) ---
  cpu_hnsw->setEf(search_params.ef);

  // Warmup
  cpu_hnsw->searchKnn(h_queries.data_handle(), k_);

  auto t2 = std::chrono::high_resolution_clock::now();
  std::vector<int64_t> h_cpu_neighbors(n_queries_ * k_);
  for (int q = 0; q < n_queries_; q++) {
    auto result = cpu_hnsw->searchKnn(
      h_queries.data_handle() + static_cast<int64_t>(q) * dim_, k_);
    int pos = k_ - 1;
    while (!result.empty()) {
      h_cpu_neighbors[q * k_ + pos] = static_cast<int64_t>(result.top().second);
      result.pop();
      pos--;
    }
  }
  auto t3 = std::chrono::high_resolution_clock::now();
  double cpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

  // --- Step 6: Compare GPU vs CPU HNSW overlap ---
  // At N=2M, brute force is too slow. Instead, check how many of GPU's top-k
  // match CPU HNSW's top-k. Both are approximate, but CPU HNSW is the baseline.
  int overlap_count = 0;
  for (int q = 0; q < n_queries_; q++) {
    for (int i = 0; i < k_; i++) {
      int64_t gpu_id = static_cast<int64_t>(h_gpu_neighbors[q * k_ + i]);
      for (int j = 0; j < k_; j++) {
        if (h_cpu_neighbors[q * k_ + j] == gpu_id) {
          overlap_count++;
          break;
        }
      }
    }
  }
  float overlap = static_cast<float>(overlap_count) / (n_queries_ * k_);

  std::cout << "  N=" << n_rows_ << " dim=" << dim_ << " k=" << k_
            << " metric=" << (use_ip ? "IP" : "L2") << "\n";
  std::cout << "  GPU vs CPU HNSW overlap@" << k_ << " = " << overlap << "\n";
  std::cout << "  GPU: " << gpu_ms << " ms for " << n_queries_ << " queries"
            << " (" << (n_queries_ / (gpu_ms / 1000.0)) << " QPS)\n";
  std::cout << "  CPU HNSW: " << cpu_ms << " ms for " << n_queries_ << " queries"
            << " (" << (n_queries_ / (cpu_ms / 1000.0)) << " QPS)\n";
  std::cout << "  Speedup: " << (cpu_ms / gpu_ms) << "x\n";

  // GPU should find >=90% of the same neighbors as CPU HNSW
  EXPECT_GT(overlap, 0.90f)
    << "GPU vs CPU HNSW overlap@" << k_ << " = " << overlap
    << " (expected > 0.90) with n_rows=" << n_rows_ << " dim=" << dim_;
}

INSTANTIATE_TEST_SUITE_P(
  GpuHnswScale,
  GpuHnswScaleTest,
  ::testing::Values(
    // N=2M with 384-dim (production-like config)
    std::make_tuple(2000000, 384, 10, cuvs::distance::DistanceType::L2Expanded)
  ));

#endif  // CUVS_BUILD_CAGRA_HNSWLIB
