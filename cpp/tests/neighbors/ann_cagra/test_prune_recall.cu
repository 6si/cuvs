/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Compare recall before and after graph optimization (pruning) at N=50K, dim=384.
 * build_knn_graph() gives the intermediate graph; helpers::optimize() gives the pruned graph.
 * Only IVF_PQ is tested since build_knn_graph() has no nn_descent overload.
 */

#include "../naive_knn.cuh"
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <gtest/gtest.h>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>
#include <rmm/device_uvector.hpp>

#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

using DataT = int8_t;
using IdxT  = uint32_t;
using DistT = float;

static constexpr int64_t kN        = 50'000;
static constexpr int64_t kDim      = 384;
static constexpr int64_t kQueries  = 100;
static constexpr int64_t kK        = 10;
static constexpr int64_t kInterDeg = 128;
static constexpr int64_t kGraphDeg = 64;

double compute_recall(const std::vector<IdxT>& gt,
                      const std::vector<IdxT>& result,
                      int64_t n_queries,
                      int64_t k)
{
  int64_t correct = 0;
  for (int64_t i = 0; i < n_queries; i++) {
    for (int64_t j = 0; j < k; j++) {
      IdxT r = result[i * k + j];
      for (int64_t g = 0; g < k; g++) {
        if (gt[i * k + g] == r) { correct++; break; }
      }
    }
  }
  return static_cast<double>(correct) / static_cast<double>(n_queries * k);
}

void run_test(cuvs::distance::DistanceType metric)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  // Generate INT8 dataset on device, copy to host for build_knn_graph
  rmm::device_uvector<DataT> db_dev(kN * kDim, stream);
  raft::random::RngState rng(42ULL);
  raft::random::uniformInt(handle, rng, db_dev.data(), kN * kDim, DataT(-10), DataT(10));
  raft::resource::sync_stream(handle);

  std::vector<DataT> h_db(kN * kDim);
  cudaMemcpy(h_db.data(), db_dev.data(), kN * kDim * sizeof(DataT), cudaMemcpyDeviceToHost);

  if (metric == cuvs::distance::DistanceType::InnerProduct) {
    for (int64_t i = 0; i < kN; i++) {
      float norm = 0;
      for (int64_t d = 0; d < kDim; d++) { float v = h_db[i * kDim + d]; norm += v * v; }
      norm = std::sqrt(norm);
      if (norm > 0)
        for (int64_t d = 0; d < kDim; d++)
          h_db[i * kDim + d] =
            static_cast<DataT>(std::round(h_db[i * kDim + d] / norm * 10));
    }
    cudaMemcpy(db_dev.data(), h_db.data(), kN * kDim * sizeof(DataT), cudaMemcpyHostToDevice);
    raft::resource::sync_stream(handle);
  }

  auto db_dev_view  = raft::make_device_matrix_view<const DataT, int64_t>(db_dev.data(), kN, kDim);
  auto db_host_view = raft::make_host_matrix_view<const DataT, int64_t>(h_db.data(), kN, kDim);

  // Ground truth
  rmm::device_uvector<DistT> gt_dist_dev(kQueries * kK, stream);
  rmm::device_uvector<IdxT>  gt_idx_dev(kQueries * kK, stream);
  cuvs::neighbors::naive_knn<DistT, DataT, IdxT>(handle,
                                                  gt_dist_dev.data(),
                                                  gt_idx_dev.data(),
                                                  db_dev.data(),
                                                  db_dev.data(),
                                                  kQueries,
                                                  kN,
                                                  kDim,
                                                  kK,
                                                  metric);
  std::vector<IdxT> gt(kQueries * kK);
  cudaMemcpy(gt.data(), gt_idx_dev.data(), kQueries * kK * sizeof(IdxT), cudaMemcpyDeviceToHost);
  raft::resource::sync_stream(handle);

  // Build intermediate KNN graph via IVF_PQ
  auto knn_graph = raft::make_host_matrix<IdxT, int64_t>(kN, kInterDeg);
  auto bp = cuvs::neighbors::cagra::graph_build_params::ivf_pq_params(
    raft::make_extents<int64_t>(kN, kDim), metric);
  cuvs::neighbors::cagra::build_knn_graph(handle, db_host_view, knn_graph.view(), bp);
  raft::resource::sync_stream(handle);

  // Pre-optimization: take first kGraphDeg columns of intermediate graph
  auto pre_graph = raft::make_host_matrix<IdxT, int64_t>(kN, kGraphDeg);
  for (int64_t i = 0; i < kN; i++)
    for (int64_t j = 0; j < kGraphDeg; j++)
      pre_graph(i, j) = knn_graph(i, j);

  // Post-optimization: prune+merge
  auto post_graph = raft::make_host_matrix<IdxT, int64_t>(kN, kGraphDeg);
  cuvs::neighbors::cagra::helpers::optimize(handle, knn_graph.view(), post_graph.view());

  cuvs::neighbors::cagra::search_params sp;
  sp.itopk_size   = 256;
  sp.search_width = 4;

  auto nb_dev   = raft::make_device_matrix<IdxT,  int64_t>(handle, kQueries, kK);
  auto dist_dev = raft::make_device_matrix<DistT, int64_t>(handle, kQueries, kK);
  std::vector<IdxT> result(kQueries * kK);

  auto q_view =
    raft::make_device_matrix_view<const DataT, int64_t>(db_dev.data(), kQueries, kDim);

  auto pre_index = cuvs::neighbors::cagra::index<DataT, IdxT>(
    handle, metric, db_dev_view, raft::make_const_mdspan(pre_graph.view()));
  cuvs::neighbors::cagra::search(
    handle, sp, pre_index, q_view, nb_dev.view(), dist_dev.view());
  raft::resource::sync_stream(handle);
  cudaMemcpy(result.data(), nb_dev.data_handle(), kQueries * kK * sizeof(IdxT), cudaMemcpyDeviceToHost);
  double pre_recall = compute_recall(gt, result, kQueries, kK);

  auto post_index = cuvs::neighbors::cagra::index<DataT, IdxT>(
    handle, metric, db_dev_view, raft::make_const_mdspan(post_graph.view()));
  cuvs::neighbors::cagra::search(
    handle, sp, post_index, q_view, nb_dev.view(), dist_dev.view());
  raft::resource::sync_stream(handle);
  cudaMemcpy(result.data(), nb_dev.data_handle(), kQueries * kK * sizeof(IdxT), cudaMemcpyDeviceToHost);
  double post_recall = compute_recall(gt, result, kQueries, kK);

  const char* mname = (metric == cuvs::distance::DistanceType::InnerProduct) ? "IP" : "L2";
  printf("  N=%ld  dim=%ld  IVF_PQ  %s  pre_prune=%.3f  post_prune=%.3f\n",
         kN, kDim, mname, pre_recall, post_recall);
  fflush(stdout);
}

TEST(CagraPruneRecall, IvfPqPreVsPostOptimize)
{
  run_test(cuvs::distance::DistanceType::L2Expanded);
  run_test(cuvs::distance::DistanceType::InnerProduct);
}

void run_sampling_sweep(cuvs::distance::DistanceType metric)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  rmm::device_uvector<DataT> db_dev(kN * kDim, stream);
  raft::random::RngState rng(42ULL);
  raft::random::uniformInt(handle, rng, db_dev.data(), kN * kDim, DataT(-10), DataT(10));
  raft::resource::sync_stream(handle);

  std::vector<DataT> h_db(kN * kDim);
  cudaMemcpy(h_db.data(), db_dev.data(), kN * kDim * sizeof(DataT), cudaMemcpyDeviceToHost);

  if (metric == cuvs::distance::DistanceType::InnerProduct) {
    for (int64_t i = 0; i < kN; i++) {
      float norm = 0;
      for (int64_t d = 0; d < kDim; d++) { float v = h_db[i * kDim + d]; norm += v * v; }
      norm = std::sqrt(norm);
      if (norm > 0)
        for (int64_t d = 0; d < kDim; d++)
          h_db[i * kDim + d] =
            static_cast<DataT>(std::round(h_db[i * kDim + d] / norm * 10));
    }
    cudaMemcpy(db_dev.data(), h_db.data(), kN * kDim * sizeof(DataT), cudaMemcpyHostToDevice);
    raft::resource::sync_stream(handle);
  }

  auto db_dev_view = raft::make_device_matrix_view<const DataT, int64_t>(db_dev.data(), kN, kDim);

  // Ground truth
  rmm::device_uvector<DistT> gt_dist_dev(kQueries * kK, stream);
  rmm::device_uvector<IdxT>  gt_idx_dev(kQueries * kK, stream);
  cuvs::neighbors::naive_knn<DistT, DataT, IdxT>(handle,
                                                  gt_dist_dev.data(),
                                                  gt_idx_dev.data(),
                                                  db_dev.data(),
                                                  db_dev.data(),
                                                  kQueries,
                                                  kN,
                                                  kDim,
                                                  kK,
                                                  metric);
  std::vector<IdxT> gt(kQueries * kK);
  cudaMemcpy(gt.data(), gt_idx_dev.data(), kQueries * kK * sizeof(IdxT), cudaMemcpyDeviceToHost);
  raft::resource::sync_stream(handle);

  // Build index once
  cuvs::neighbors::cagra::index_params ip;
  ip.metric                    = metric;
  ip.graph_degree              = kGraphDeg;
  ip.intermediate_graph_degree = kInterDeg;
  ip.graph_build_params        = cuvs::neighbors::cagra::graph_build_params::ivf_pq_params(
    raft::make_extents<int64_t>(kN, kDim), metric);
  auto index = cuvs::neighbors::cagra::build(handle, ip, raft::make_const_mdspan(db_dev_view));
  raft::resource::sync_stream(handle);

  auto nb_dev   = raft::make_device_matrix<IdxT,  int64_t>(handle, kQueries, kK);
  auto dist_dev = raft::make_device_matrix<DistT, int64_t>(handle, kQueries, kK);
  std::vector<IdxT> result(kQueries * kK);
  auto q_view = raft::make_device_matrix_view<const DataT, int64_t>(db_dev.data(), kQueries, kDim);

  const char* mname = (metric == cuvs::distance::DistanceType::InnerProduct) ? "IP" : "L2";

  for (uint32_t nrs : {1, 2, 4, 8, 16, 32}) {
    cuvs::neighbors::cagra::search_params sp;
    sp.itopk_size           = 256;
    sp.search_width         = 4;
    sp.num_random_samplings = nrs;

    cuvs::neighbors::cagra::search(handle, sp, index, q_view, nb_dev.view(), dist_dev.view());
    raft::resource::sync_stream(handle);
    cudaMemcpy(result.data(), nb_dev.data_handle(), kQueries * kK * sizeof(IdxT), cudaMemcpyDeviceToHost);
    double recall = compute_recall(gt, result, kQueries, kK);

    printf("  N=%ld  dim=%ld  IVF_PQ  %s  num_random_samplings=%2d  recall=%.3f\n",
           kN, kDim, mname, nrs, recall);
    fflush(stdout);
  }
}

TEST(CagraPruneRecall, NumRandomSamplingsSweep)
{
  run_sampling_sweep(cuvs::distance::DistanceType::L2Expanded);
  run_sampling_sweep(cuvs::distance::DistanceType::InnerProduct);
}

void run_ivf_seed_test(cuvs::distance::DistanceType metric)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  rmm::device_uvector<DataT> db_dev(kN * kDim, stream);
  raft::random::RngState rng(42ULL);
  raft::random::uniformInt(handle, rng, db_dev.data(), kN * kDim, DataT(-10), DataT(10));
  raft::resource::sync_stream(handle);

  std::vector<DataT> h_db(kN * kDim);
  cudaMemcpy(h_db.data(), db_dev.data(), kN * kDim * sizeof(DataT), cudaMemcpyDeviceToHost);

  if (metric == cuvs::distance::DistanceType::InnerProduct) {
    for (int64_t i = 0; i < kN; i++) {
      float norm = 0;
      for (int64_t d = 0; d < kDim; d++) { float v = h_db[i * kDim + d]; norm += v * v; }
      norm = std::sqrt(norm);
      if (norm > 0)
        for (int64_t d = 0; d < kDim; d++)
          h_db[i * kDim + d] =
            static_cast<DataT>(std::round(h_db[i * kDim + d] / norm * 10));
    }
    cudaMemcpy(db_dev.data(), h_db.data(), kN * kDim * sizeof(DataT), cudaMemcpyHostToDevice);
    raft::resource::sync_stream(handle);
  }

  auto db_dev_view = raft::make_device_matrix_view<const DataT, int64_t>(db_dev.data(), kN, kDim);

  // Ground truth
  rmm::device_uvector<DistT> gt_dist_dev(kQueries * kK, stream);
  rmm::device_uvector<IdxT>  gt_idx_dev(kQueries * kK, stream);
  cuvs::neighbors::naive_knn<DistT, DataT, IdxT>(handle,
                                                  gt_dist_dev.data(),
                                                  gt_idx_dev.data(),
                                                  db_dev.data(),
                                                  db_dev.data(),
                                                  kQueries,
                                                  kN,
                                                  kDim,
                                                  kK,
                                                  metric);
  std::vector<IdxT> gt(kQueries * kK);
  cudaMemcpy(gt.data(), gt_idx_dev.data(), kQueries * kK * sizeof(IdxT), cudaMemcpyDeviceToHost);
  raft::resource::sync_stream(handle);

  // Build CAGRA index
  cuvs::neighbors::cagra::index_params cagra_ip;
  cagra_ip.metric                    = metric;
  cagra_ip.graph_degree              = kGraphDeg;
  cagra_ip.intermediate_graph_degree = kInterDeg;
  cagra_ip.graph_build_params        = cuvs::neighbors::cagra::graph_build_params::ivf_pq_params(
    raft::make_extents<int64_t>(kN, kDim), metric);
  auto cagra_index =
    cuvs::neighbors::cagra::build(handle, cagra_ip, raft::make_const_mdspan(db_dev_view));
  raft::resource::sync_stream(handle);

  // Build IVF-PQ index for coarse seed generation
  auto ivf_ip    = cuvs::neighbors::ivf_pq::index_params::from_dataset(
    raft::make_extents<int64_t>(kN, kDim), metric);
  ivf_ip.n_lists = 128;
  auto ivf_index = cuvs::neighbors::ivf_pq::build(handle, ivf_ip, db_dev_view);

  auto nb_dev   = raft::make_device_matrix<IdxT,  int64_t>(handle, kQueries, kK);
  auto dist_dev = raft::make_device_matrix<DistT, int64_t>(handle, kQueries, kK);
  std::vector<IdxT> result(kQueries * kK);
  auto q_view = raft::make_device_matrix_view<const DataT, int64_t>(db_dev.data(), kQueries, kDim);

  const char* mname = (metric == cuvs::distance::DistanceType::InnerProduct) ? "IP" : "L2";

  // Baseline: random seeds
  {
    cuvs::neighbors::cagra::search_params sp;
    sp.itopk_size   = 256;
    sp.search_width = 4;
    cuvs::neighbors::cagra::search(handle, sp, cagra_index, q_view, nb_dev.view(), dist_dev.view());
    raft::resource::sync_stream(handle);
    cudaMemcpy(result.data(), nb_dev.data_handle(), kQueries * kK * sizeof(IdxT), cudaMemcpyDeviceToHost);
    printf("  N=%ld  dim=%ld  IVF_PQ  %s  seeds=random        recall=%.3f\n",
           kN, kDim, mname, compute_recall(gt, result, kQueries, kK));
    fflush(stdout);
  }

  // IVF-seeded: get coarse candidates from IVF-PQ, pass as seeds
  for (uint32_t n_seeds : {1, 4, 16, 64}) {
    auto seed_nb   = raft::make_device_matrix<int64_t, int64_t>(handle, kQueries, n_seeds);
    auto seed_dist = raft::make_device_matrix<DistT,   int64_t>(handle, kQueries, n_seeds);
    cuvs::neighbors::ivf_pq::search_params ivf_sp;
    ivf_sp.n_probes = 8;
    cuvs::neighbors::ivf_pq::search(
      handle, ivf_sp, ivf_index, q_view, seed_nb.view(), seed_dist.view());
    raft::resource::sync_stream(handle);

    // Cast int64_t → uint32_t seeds
    auto seeds_u32 = raft::make_device_matrix<uint32_t, int64_t>(handle, kQueries, n_seeds);
    thrust::transform(
      thrust::cuda::par.on(stream),
      seed_nb.data_handle(),
      seed_nb.data_handle() + kQueries * n_seeds,
      seeds_u32.data_handle(),
      [] __device__(int64_t x) { return static_cast<uint32_t>(x); });
    raft::resource::sync_stream(handle);

    cuvs::neighbors::cagra::search_params sp;
    sp.itopk_size       = 256;
    sp.search_width     = 4;
    sp.seed_indices     = seeds_u32.data_handle();
    sp.num_seed_indices = n_seeds;

    cuvs::neighbors::cagra::search(handle, sp, cagra_index, q_view, nb_dev.view(), dist_dev.view());
    raft::resource::sync_stream(handle);
    cudaMemcpy(result.data(), nb_dev.data_handle(), kQueries * kK * sizeof(IdxT), cudaMemcpyDeviceToHost);
    printf("  N=%ld  dim=%ld  IVF_PQ  %s  seeds=ivf_pq n=%2d  recall=%.3f\n",
           kN, kDim, mname, n_seeds, compute_recall(gt, result, kQueries, kK));
    fflush(stdout);
  }
}

TEST(CagraPruneRecall, IvfSeededVsRandom)
{
  run_ivf_seed_test(cuvs::distance::DistanceType::L2Expanded);
  run_ivf_seed_test(cuvs::distance::DistanceType::InnerProduct);
}

}  // namespace
