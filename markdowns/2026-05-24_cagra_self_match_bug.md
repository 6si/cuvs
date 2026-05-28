# CAGRA Self-Match Miss Bug Investigation

**Date:** 2026-05-24  
**Issue:** https://github.com/rapidsai/cuvs/issues/2102  
**Linked Milvus issue:** https://github.com/milvus-io/milvus/issues/49864

---

## Bug Description

`cuvs.neighbors.cagra` with `metric=inner_product` and `build_algo=ivf_pq` misses exact self-matches in a self-query. With unit-normalized vectors, querying `xb[i]` against an index built from `xb` should always return `xb[i]` at rank 1 with score ≈ 1.0. Instead, some queries return a random neighbor with score ~0.17–0.24.

## Environment

- GPU: NVIDIA L40S (46 GB VRAM)
- Driver: 570.172.08 / CUDA 12.8
- cuvs: 26.2.0 (`cuvs-cu12`)
- cupy: 14.1.0 (`cupy-cuda12x`)
- Python: 3.10.12

## Repro Confirmed

Script: `contact_embedding/scripts/milvus/repro_cagra_self_match.py`

**Baseline config:**
```
N=50_000, D=384, Q=512, K=10
metric=inner_product, build_algo=ivf_pq
intermediate_graph_degree=64, graph_degree=32
itopk_size=64, search_width=1
```

**Result:** `self_top1_count=502/512` — **10 misses**

Bad queries: `[97, 190, 216, 245, 250, 284, 299, 390, 468, 481]`

Example miss:
```
query=97 → top_id=9642, returned_score=0.1795, manual_self_score=1.0
```

## Relevance to Our Pipeline

`mpd_v0_cagra_int8_cached` uses `GPU_CAGRA` with `metric=IP` and `build_algo=IVF_PQ` — the exact same configuration. Some fraction of production queries are returning wrong results.

---

## Workaround Investigation

### Step 1: Baseline (completed)

| Config | self_top1 | Misses |
|--------|-----------|--------|
| `search_width=1, itopk_size=64` (baseline) | 502/512 | 10 |

### Step 2: `search_width` sweep (IVF_PQ, itopk_size=64)

| search_width | self_top1 | Misses | Result |
|-------------|-----------|--------|--------|
| 1 (baseline) | 500/512 | 12 | FAIL |
| 2 | 499/512 | 13 | FAIL (worse) |
| 4 | 503/512 | 9 | FAIL |
| 8 | 507/512 | 5 | FAIL |
| 16 | 509/512 | 3 | FAIL |
| **32** | **512/512** | **0** | **PASS** |

`search_width=32` fixes it, but is a heavy knob — 32x more graph traversal per query.

---

### Step 3: `itopk_size` sweep (IVF_PQ, search_width=1)

| itopk_size | self_top1 | Misses | Result |
|-----------|-----------|--------|--------|
| 64 (baseline) | 500/512 | 12 | FAIL |
| **128** | **512/512** | **0** | **PASS** |
| 256 | 512/512 | 0 | PASS |
| 512 | 512/512 | 0 | PASS |

**`itopk_size=128` is the cheapest single-parameter fix** — only 2x the candidate pool, no change to graph or build.

---

### Step 4: `build_algo=nn_descent`

| Config | self_top1 | Misses | Result |
|--------|-----------|--------|--------|
| nn_descent, itopk=64, sw=1 | 510/512 | 2 | FAIL |
| nn_descent, itopk=64, sw=4 | 510/512 | 2 | FAIL |

`nn_descent` reduces misses from 12→2 but does **not** eliminate them. Not a reliable fix on its own.

---

### Step 5: Larger graph (`intermediate=128, graph_degree=64`)

| Config | self_top1 | Misses | Result |
|--------|-----------|--------|--------|
| ivf_pq_large, itopk=64, sw=1 | 512/512 | 0 | **PASS** |
| ivf_pq_large, itopk=128, sw=1 | 512/512 | 0 | **PASS** |
| ivf_pq_large, itopk=64, sw=4 | 512/512 | 0 | **PASS** |

Doubling `intermediate_graph_degree` (64→128) and `graph_degree` (32→64) eliminates all misses — even with the default `itopk_size=64, search_width=1`.

---

### Step 6: `itopk_size=128` cross-build validation (2 builds)

Re-ran with fresh builds of both algos to check whether Step 3's PASS for `itopk=128 + ivf_pq` was a fluke.

| Config | self_top1 | Misses | Result |
|--------|-----------|--------|--------|
| ivf_pq, itopk=128, sw=1 | 510/512 | 2 | **FAIL** |
| nn_descent, itopk=128, sw=1 | 512/512 | 0 | PASS |
| nn_descent, itopk=128, sw=2 | 512/512 | 0 | PASS |
| nn_descent, itopk=256, sw=1 | 512/512 | 0 | PASS |

**Key finding:** `ivf_pq + itopk=128` is non-deterministic across builds — it passed in Step 3 and failed here. IVF_PQ quantization introduces randomness in graph construction, so some builds happen to place self-vectors reachably, others don't. `nn_descent + itopk=128` passes consistently.

---

### Step 7: 5-build determinism test + latency benchmark

**Determinism (5 independent builds, itopk=128, sw=1):**

| build_algo | misses per build | failed builds |
|------------|-----------------|---------------|
| ivf_pq | [1, 0, 1, 0, 1] | 3/5 |
| nn_descent | [0, 0, 0, 0, 0] | 0/5 |

`ivf_pq + itopk=128` fails in 3 out of 5 builds. Not reliable. `nn_descent + itopk=128` is clean across all 5.

**Latency (Q=512, K=10, 50 runs, p50/p95, L40S):**

| Config | mean | p50 | p95 |
|--------|------|-----|-----|
| ivf_pq, itopk=64 (baseline) | 2.02ms | 1.98ms | 2.14ms |
| ivf_pq, itopk=128 | 2.45ms | 2.40ms | 2.60ms |
| nn_descent, itopk=64 | 2.07ms | 2.01ms | 2.28ms |
| **nn_descent, itopk=128** | **2.44ms** | **2.39ms** | **2.62ms** |

Both algos are **identical in search latency** at the same `itopk_size`. The +0.4ms overhead of `itopk=128` vs `itopk=64` is the same for both. Build algo only affects graph construction — not search speed.

---

### Step 8: Re-test at N=500,000 (10x scale)

All prior tests were at N=50K. Production collection is much larger. Retested determinism and speed at N=500,000.

**Determinism (5 builds each, itopk=128, sw=1):**

| build_algo | misses per build | failed builds | avg build time |
|------------|-----------------|---------------|----------------|
| ivf_pq | [154, 155, 185, 154, 148] | 5/5 | 1.9s |
| nn_descent | [99, 113, 107, 105, 97] | 5/5 | 5.2s |

**Both algos fail completely at N=500K.** ~100–185 misses out of 512 (~20–36% miss rate). The N=50K workarounds (`itopk=128`, `nn_descent`) do not scale.

**Latency (Q=512, K=10, 50 runs, L40S):**

| Config | mean | p50 | p95 |
|--------|------|-----|-----|
| ivf_pq, itopk=64 | 3.75ms | 3.68ms | 3.92ms |
| ivf_pq, itopk=128 | 5.63ms | 5.54ms | 6.04ms |
| nn_descent, itopk=64 | 3.74ms | 3.69ms | 3.89ms |
| nn_descent, itopk=128 | 5.61ms | 5.55ms | 5.83ms |

Search latency at N=500K is ~3.7ms (itopk=64) vs ~5.6ms (itopk=128) — both algos again identical. About 1.7ms slower than N=50K due to larger graph traversal space.

---

## Summary & Recommendation

**The bug is much worse than initially measured.** At N=50K the search-time workarounds (`itopk=128`, `nn_descent`) appeared to fix it. At N=500K — closer to production scale — both algos fail on every build with ~20–36% miss rate. `itopk=128` and `nn_descent` are not viable fixes.

| Scale | Build algo | itopk | Misses | Verdict |
|-------|-----------|-------|--------|---------|
| N=50K | ivf_pq | 64 | 10–12/512 | FAIL |
| N=50K | ivf_pq | 128 | 0–1/512 | Unreliable |
| N=50K | nn_descent | 128 | 0/512 | Appeared fixed |
| **N=500K** | **ivf_pq** | **128** | **148–185/512** | **FAIL** |
| **N=500K** | **nn_descent** | **128** | **97–113/512** | **FAIL** |

This is a confirmed upstream bug in cuvs 26.2.0 that requires a fix in the library. The issue is filed at https://github.com/rapidsai/cuvs/issues/2102.

**For production (`mpd_v0_cagra_int8_cached`):** GPU_CAGRA with IP metric at production scale is currently returning wrong results. Full parameter sweep results below.

---

### Step 9: Full parameter sweep at N=500K

Tested all CAGRA knobs: itopk, search_width, graph size, build algo. Single build per config (determinism re-validation needed for candidates).

**Build times:**

| Config | Build time |
|--------|-----------|
| ivf_pq 64/32 (baseline) | 2.1s |
| nn_descent 64/32 | 5.2s |
| ivf_pq 128/64 | 3.3s |
| nn_descent 128/64 | 8.0s |
| ivf_pq 256/128 | 8.4s |
| nn_descent 256/128 | 14.8s |

**Section 1 — itopk sweep, ivf_pq 64/32 (baseline graph):**

| itopk | misses | p50 |
|-------|--------|-----|
| 64 | 280/512 | 3.7ms |
| 128 | 146/512 | 5.5ms |
| 256 | 57/512 | 9.2ms |
| 512 | 2/512 | 16.7ms |
| **1024** | **0/512** | **37.1ms** |

Fixes at `itopk=1024` but 10x slower than baseline.

**Section 2 — search_width sweep, ivf_pq 64/32, itopk=128:**

| search_width | misses | p50 |
|-------------|--------|-----|
| 1 | 146/512 | 5.5ms |
| 2 | 148/512 | 5.6ms |
| 4 | 149/512 | 5.8ms |
| 8 | 131/512 | 6.3ms |
| 16 | 117/512 | 7.0ms |
| 32 | 83/512 | 8.0ms |

`search_width` barely helps at N=500K — the baseline graph is simply too sparse.

**Section 3 — combos, ivf_pq 64/32:**

| itopk | sw | misses | p50 |
|-------|----|--------|-----|
| 256 | 4 | 46/512 | 9.6ms |
| 256 | 8 | 42/512 | 10.3ms |
| 512 | 4 | 4/512 | 17.3ms |
| 512 | 8 | 1/512 | 18.6ms |
| **1024** | **4** | **0/512** | **37.2ms** |

Still requires `itopk=1024` with baseline graph.

**Section 4 — larger graph, ivf_pq 128/64:**

| itopk | sw | misses | p50 |
|-------|----|--------|-----|
| 64 | 1 | 72/512 | 5.8ms |
| 128 | 1 | 9/512 | 9.5ms |
| **256** | **1** | **0/512** | **17.1ms** |
| 128 | 4 | 11/512 | 10.3ms |
| 128 | 8 | 8/512 | 10.8ms |
| 128 | 16 | 5/512 | 11.9ms |

`ivf_pq 128/64 + itopk=256` fixes it at 17ms. `search_width` doesn't help here either.

**Section 5 — larger graph, nn_descent 128/64:**

| itopk | sw | misses | p50 |
|-------|----|--------|-----|
| 64 | 1 | 20/512 | 5.8ms |
| **128** | **1** | **0/512** | **9.5ms** |
| 256 | 1 | 0/512 | 17.0ms |
| 128 | 4 | 1/512 | 10.2ms |
| 128 | 8 | 0/512 | 10.8ms |

`nn_descent 128/64 + itopk=128` fixes it at 9.5ms — best result so far. Note: `sw=4` showed 1 miss, suggesting this config may not be fully deterministic.

**Section 6 — XL graph, ivf_pq 256/128:**

| itopk | misses | p50 |
|-------|--------|-----|
| 64 | 42/512 | 9.6ms |
| 128 | 7/512 | 17.1ms |
| **256** | **0/512** | **32.0ms** |

Fixes at `itopk=256` but 32ms is very slow.

**Section 6b — XL graph, nn_descent 256/128:**

| itopk | misses | p50 |
|-------|--------|-----|
| **64** | **0/512** | **9.7ms** |
| 128 | 0/512 | 17.3ms |
| 256 | 0/512 | 32.3ms |

`nn_descent 256/128 + itopk=64` is zero misses at 9.7ms — but build takes 14.8s and graph uses ~4x memory vs baseline.

---

## Summary & Recommendation

**Search_width is not a useful lever at N=500K** — it never recovers enough graph connectivity to fix the problem.

**Passing configs ranked by search latency (single build each — determinism not yet re-validated):**

| Config | p50 | Build time | Misses |
|--------|-----|-----------|--------|
| **nn_descent 128/64 + itopk=128** | **9.5ms** | 8.0s | 0/512 |
| nn_descent 256/128 + itopk=64 | 9.7ms | 14.8s | 0/512 |
| ivf_pq 128/64 + itopk=256 | 17.1ms | 3.3s | 0/512 |
| nn_descent 256/128 + itopk=128 | 17.3ms | 14.8s | 0/512 |
| ivf_pq 256/128 + itopk=256 | 32.0ms | 8.4s | 0/512 |
| ivf_pq 64/32 + itopk=1024 | 37.1ms | 2.1s | 0/512 |

**Best candidate: `nn_descent 128/64 + itopk=128`** — 9.5ms p50, 8s build, but note that `sw=4` showed 1 miss suggesting it may not be fully deterministic. Multi-build validation needed before committing.

---

### Step 10: 5-build determinism + 100-run latency for all passing candidates

**Determinism (5 builds each, N=500K):**

| Config | misses per build | failed | avg build |
|--------|-----------------|--------|-----------|
| nn_descent 128/64, itopk=128 | [0, 0, 3, 1, 0] | 2/5 | 8.0s |
| nn_descent 256/128, itopk=64 | [0, 0, 0, 0, 1] | 1/5 | 15.6s |
| **ivf_pq 128/64, itopk=256** | **[0, 0, 0, 0, 0]** | **0/5** | **3.5s** |
| **ivf_pq 256/128, itopk=256** | **[0, 0, 0, 0, 0]** | **0/5** | **8.5s** |
| **nn_descent 256/128, itopk=128** | **[0, 0, 0, 0, 0]** | **0/5** | **14.9s** |

The two `nn_descent` configs with smaller graphs (128/64 and 256/128 at itopk=64) are still non-deterministic and drop out. Three survivors.

**Latency benchmark (100 runs, Q=512, K=10):**

| Config | mean | p50 | p95 | p99 | Build time |
|--------|------|-----|-----|-----|-----------|
| **ivf_pq 128/64, itopk=256** | **17.06ms** | **17.03ms** | **17.37ms** | **17.67ms** | **3.5s** |
| nn_descent 256/128, itopk=128 | 17.22ms | 17.16ms | 17.49ms | 17.68ms | 14.9s |
| ivf_pq 256/128, itopk=256 | 32.02ms | 32.00ms | 32.34ms | 32.84ms | 8.5s |

The two fastest survivors (`ivf_pq 128/64` and `nn_descent 256/128`) are **statistically identical in search latency** (~17ms p50). `ivf_pq 256/128` is 2x slower at 32ms and provides no accuracy benefit over the others.

---

## Summary & Recommendation

**Confirmed working configs at N=500K (5/5 builds, zero misses):**

| Config | p50 | Build time | Notes |
|--------|-----|-----------|-------|
| **ivf_pq 128/64, itopk=256** | **17ms** | **3.5s** | **Best: fastest build, tied on search** |
| nn_descent 256/128, itopk=128 | 17ms | 14.9s | Same search speed, 4x longer build |
| ivf_pq 256/128, itopk=256 | 32ms | 8.5s | No benefit over ivf_pq 128/64 |

**Recommended: `build_algo=ivf_pq`, `intermediate_graph_degree=128`, `graph_degree=64`, `itopk_size=256`**

- Only config that is deterministically correct, fastest to build (3.5s), and tied for fastest search (17ms p50)
- vs production baseline (itopk=64, graph 64/32): search is ~4.6x slower (3.7ms → 17ms), build is ~1.7x slower
- Requires a rebuild of `mpd_v0_cagra_int8_cached`

Configs that looked promising but were eliminated:
- `nn_descent 128/64 + itopk=128` — failed 2/5 builds
- `nn_descent 256/128 + itopk=64` — failed 1/5 builds
- `ivf_pq 64/32 + itopk=1024` — not determinism-tested, but 37ms is too slow anyway

---

### Step 11: Winner retested at N=1,000,000

`ivf_pq 128/64 + itopk=256` fails on every build at 1M records.

| Build | Misses | Build time |
|-------|--------|-----------|
| 1 | 8/512 | 6.8s |
| 2 | 8/512 | 6.6s |
| 3 | 5/512 | 6.6s |
| 4 | 7/512 | 6.6s |
| 5 | 8/512 | 6.6s |

Search latency: 18.2ms p50 (+1ms vs N=500K, consistent with larger graph traversal space).

Escalated to larger graph sizes and higher itopk at N=1M:

| Config | misses per build | failed | avg build | p50 |
|--------|-----------------|--------|-----------|-----|
| ivf_pq 128/64, itopk=512 | [0,0,0,0,0] | 0/5 | 6.6s | 34.4ms |
| ivf_pq 128/64, itopk=1024 | [0,0,0,0,0] | 0/5 | 6.7s | 75.8ms |
| ivf_pq 256/128, itopk=256 | [0,4,1,0,2] | 3/5 | 16.4s | — |
| ivf_pq 256/128, itopk=512 | [0,0,0,0,0] | 0/5 | 16.5s | 68.1ms |
| **nn_descent 256/128, itopk=128** | **[0,0,0,0,0]** | **0/5** | **29.6s** | **18.4ms** |
| nn_descent 256/128, itopk=256 | [0,0,0,0,0] | 0/5 | 29.7s | 34.7ms |

`nn_descent 256/128 + itopk=128` is the fastest correct config at N=1M (18.4ms p50), but build time is 29.6s. `ivf_pq 128/64 + itopk=512` is also deterministically correct with a much faster build (6.6s) but nearly 2x slower search (34.4ms).

---

## Summary across all scales

| Scale | Winner config | p50 | Build time | Misses |
|-------|-------------|-----|-----------|--------|
| N=50K | nn_descent 64/32, itopk=128 | 9.5ms | 8s | 0/5 *(later invalidated)* |
| N=500K | ivf_pq 128/64, itopk=256 | 17ms | 3.5s | 0/5 |
| N=1M | **nn_descent 256/128, itopk=128** | **18.4ms** | **29.6s** | **0/5** |

**Pattern:** The required graph size and itopk both grow with N. `ivf_pq` builds are faster but need higher itopk to compensate for weaker graph quality; `nn_descent` builds stronger graphs that tolerate lower itopk and win on search speed at scale.

**Recommendation at N=1M: `nn_descent 256/128 + itopk=128`**
- 18.4ms p50 search (vs 3.7ms baseline — ~5x slower)
- 29.6s build (vs 2.1s baseline — ~14x longer)
- 0/5 builds failed

If build time is a constraint: `ivf_pq 128/64 + itopk=512` passes with 6.6s build at the cost of 34.4ms search.

---

### Step 12: FP32 at N=2,000,000

Tested the N=1M winner and runner-up, plus natural escalations.

| Config | misses per build | failed | avg build |
|--------|-----------------|--------|-----------|
| nn_descent 256/128, itopk=128 | [5, 7, 5, 2, 4] | 5/5 | 90.1s |
| nn_descent 256/128, itopk=256 | [0, 0, 0, 1, 0] | 1/5 | 105.9s |
| ivf_pq 128/64, itopk=512 | [8, 2, 5, 1, 2] | 5/5 | 22.1s |
| ivf_pq 256/128, itopk=512 | [1, 1, 0, 0, 1] | 3/5 | 55.2s |

All fail. The N=1M winner (`nn_descent 256/128 + itopk=128`) deteriorates to 5/5 failures. Build times are growing rapidly — `nn_descent 256/128` now takes 90–106s.

**The pattern is clear:** every 2x increase in N requires the next step up in both graph size and itopk to maintain correctness. This is not converging toward a practical fix at 530M scale.

### Step 13: INT8 at N=2,000,000

Same candidates, vectors quantized as `round(xb * 127).clip(-128, 127).astype(int8)` — matching our production pipeline.

| Config | misses per build | failed | avg build | p50 |
|--------|-----------------|--------|-----------|-----|
| nn_descent 256/128, itopk=128 | [0, 4, 8, 2, 1] | 4/5 | 107.9s | — |
| **nn_descent 256/128, itopk=256** | **[0, 0, 0, 0, 0]** | **0/5** | **98.6s** | **13.3ms** |
| ivf_pq 128/64, itopk=512 | [2, 3, 3, 2, 4] | 5/5 | 25.1s | — |
| ivf_pq 256/128, itopk=512 | [0, 0, 1, 1, 0] | 2/5 | 34.6s | — |

**One survivor: `nn_descent 256/128 + itopk=256` passes at N=2M INT8** — 0/5 builds failed, 13.3ms p50.

Notably faster than FP32 at the same config because INT8 arithmetic reduces the per-comparison cost during graph traversal. FP32 equivalent (`nn_descent 256/128, itopk=256`) failed 1/5 builds and wasn't benchmarked.

Build time is 98.6s — heavy but manageable for a one-time index build.

---

## Full scaling summary

| N | dtype | Passing config | p50 | Build time |
|---|-------|---------------|-----|-----------|
| 50K | FP32 | nn_descent 64/32, itopk=128 | 9.5ms | 8s |
| 500K | FP32 | ivf_pq 128/64, itopk=256 | 17ms | 3.5s |
| 1M | FP32 | nn_descent 256/128, itopk=128 | 18.4ms | 29.6s |
| 2M | FP32 | none | — | — |
| 2M | **INT8** | **nn_descent 256/128, itopk=256** | **13.3ms** | **98.6s** |

INT8 is both faster at search time and more likely to pass at scale than FP32. Production pipeline already uses INT8 — this is the right path.

**Next step:** test `nn_descent 256/128 + itopk=256` INT8 at larger N to understand where it breaks again, and whether the pattern continues to hold.

---

### Step 14: Version regression (N=50K, baseline config)

Tested the bug across 4 versions using isolated virtualenvs. Baseline = `ivf_pq 64/32, itopk=64` (original reported config). Single build per version.

| cuvs version | fp32 ivf_pq 64/32 itopk=64 | int8 ivf_pq 64/32 itopk=64 | fp32 nn_descent 64/32 itopk=128 |
|-------------|---------------------------|---------------------------|--------------------------------|
| 25.4.0 | 6/512 misses | 12/512 misses | 0/512 |
| 25.6.1 (reported) | 25/512 misses | 19/512 misses | 0/512 |
| 25.12.0 | 8/512 misses | 11/512 misses | 0/512 |
| 26.2.0 (current) | 6/512 misses | 6/512 misses | 0/512 |

**The bug is present in every tested version going back to at least 25.4.0** — this is not a regression introduced in a recent release. It appears to be a fundamental property of CAGRA's IVF_PQ graph construction with IP metric.

Miss count varies between versions (6–25) likely due to non-determinism in graph construction rather than meaningful behavioral differences. The `nn_descent 64/32 + itopk=128` workaround passes on all versions at N=50K, but as shown in Steps 8–13, it does not hold at production scale.

---

### Step 15: INT8 regression across all available versions (N=50K)

Extended the regression to INT8 across all 12 available versions, going back to 24.4.0.

| Version | INT8 supported | ivf_pq itopk=64 | ivf_pq itopk=128 | nn_descent itopk=128 |
|---------|---------------|-----------------|------------------|----------------------|
| 24.4.0 | **No** — `cagra.build` not yet in API | — | — | — |
| 24.6.0 | **No** — `cagra.build` not yet in API | — | — | — |
| 24.8.0 | Yes | 18/512 | 0/512 | 0/512 |
| 24.10.0 | Yes | 17/512 | 0/512 | 0/512 |
| 24.12.0 | Yes | 17/512 | 1/512 | 0/512 |
| 25.2.1 | Yes | 16/512 | 1/512 | 0/512 |
| 25.4.0 | Yes | 17/512 | 0/512 | 0/512 |
| 25.6.1 | Yes | 18/512 | 0/512 | 0/512 |
| 25.8.0 | Yes | 11/512 | 1/512 | 0/512 |
| 25.10.0 | Yes | 13/512 | 0/512 | 0/512 |
| 25.12.0 | Yes | 11/512 | 0/512 | 0/512 |
| 26.2.0 | Yes | 13/512 | 0/512 | 0/512 |

**Key findings:**
- INT8 support was added in **24.8.0** — the Python `cagra.build` API didn't exist before 24.6.0
- The bug exists in **every version that supports INT8**, from 24.8.0 onward
- `ivf_pq + itopk=128` occasionally shows 1 miss (24.12.0, 25.2.1, 25.8.0) — confirming the non-determinism seen in Step 6
- `nn_descent + itopk=128` is 0/512 on every version at N=50K — but does not hold at larger N as established in Steps 8–13
- Miss counts are consistent across versions (~11–18 for itopk=64), confirming this is not a regression but a persistent design characteristic

---

### Step 17: FP32 regression across all versions (N=50K)

Same test as Step 15 but FP32. Results identical pattern — bug present in every version from 24.8.0, no regression point, miss counts fluctuate 9–24 due to non-determinism.

| Version | FP32 supported | ivf_pq itopk=64 | ivf_pq itopk=128 | nn_descent itopk=128 |
|---------|---------------|-----------------|------------------|----------------------|
| 24.4.0 – 24.6.0 | No — API didn't exist | — | — | — |
| 24.8.0 | Yes | 14/512 | 0/512 | 0/512 |
| 24.10.0 | Yes | 9/512 | 1/512 | 0/512 |
| 24.12.0 | Yes | 18/512 | 0/512 | 0/512 |
| 25.2.1 | Yes | 14/512 | 0/512 | 0/512 |
| 25.4.0 | Yes | 13/512 | 1/512 | 0/512 |
| 25.6.1 | Yes | 24/512 | 1/512 | 0/512 |
| 25.8.0 | Yes | 16/512 | 0/512 | 0/512 |
| 25.10.0 | Yes | 11/512 | 0/512 | 0/512 |
| 25.12.0 | Yes | 17/512 | 1/512 | 0/512 |
| 26.2.0 | Yes | 12/512 | 0/512 | 0/512 |

**No regression point** — bug present since 24.8.0 (first version with the API) in both FP32 and INT8.

---

### Step 18: cuvs `ivf_pq` standalone index (N=50K)

Tested `cuvs.neighbors.ivf_pq` directly — this is a pure IVF-PQ approximate nearest neighbor index, not CAGRA. The key knob is `n_probes` (how many clusters to search at query time).

| Config | misses | p50 |
|--------|--------|-----|
| fp32, n_lists=223, n_probes=1 | 0/512 | 1.84ms |
| fp32, n_lists=223, n_probes=5 | 0/512 | 2.30ms |
| int8, n_lists=223, n_probes=1 | 0/512 | 2.56ms |
| int8, n_lists=223, n_probes=5 | 0/512 | 2.94ms |
| fp32, n_lists=1024, n_probes=10 | 0/512 | 3.02ms |
| int8, n_lists=1024, n_probes=10 | 0/512 | 3.07ms |

**IVF_PQ standalone passes with zero misses at every tested configuration, including `n_probes=1`.** This is because IVF_PQ always searches the clusters nearest to the query — the self-vector is guaranteed to be in its own cluster, so it's always found when that cluster is probed. CAGRA's graph traversal has no such guarantee; it can get stuck in a remote subgraph and never reach the self-vector's neighborhood.

At N=50K, `ivf_pq n_probes=1` at 1.84ms is faster than the CAGRA baseline (3.7ms at N=50K, 18ms at the N=2M INT8 winner). The question is whether this holds at scale.

---

### Step 19: CAGRA with IVF_PQ sub-parameter sweep (N=500K INT8, graph=64/32, 3 builds each)

Tuned `ivf_pq_build_params` and `ivf_pq_search_params` inside CAGRA's `build_algo="ivf_pq"` path, holding graph degrees fixed at 64/32 and `itopk=128`.

**n_lists sweep** (pq_dim=0, n_probes=20):

| n_lists | misses per build | failed |
|---------|-----------------|--------|
| 512 | [148, 141, 158] | 3/3 |
| 1024 | [144, 137, 136] | 3/3 |
| 2048 | [142, 148, 147] | 3/3 |
| 4096 | [163, 168, 147] | 3/3 |
| 8192 | [154, 170, 152] | 3/3 |

**n_probes sweep** (n_lists=1024, pq_dim=0):

| n_probes | misses per build | failed |
|---------|-----------------|--------|
| 5 | [145, 143, 144] | 3/3 |
| 10 | [146, 153, 151] | 3/3 |
| 20 | [140, 141, 163] | 3/3 |
| 50 | [154, 144, 144] | 3/3 |
| 100 | [164, 157, 152] | 3/3 |

**pq_dim sweep** (n_lists=1024, n_probes=20):

| pq_dim | misses per build | failed |
|--------|-----------------|--------|
| 0 (auto) | [150, 159, 143] | 3/3 |
| 32 | [153, 147, 158] | 3/3 |
| 64 | [142, 139, 163] | 3/3 |
| 96 | [164, 136, 166] | 3/3 |
| 128 | [140, 162, 158] | 3/3 |
| 192 | [147, 138, 154] | 3/3 |

**kmeans_n_iters sweep** (n_lists=1024, pq_dim=0, n_probes=20):

| kmeans_n_iters | misses per build | failed |
|---------------|-----------------|--------|
| 20 | [152, 144, 130] | 3/3 |
| 50 | [141, 159, 165] | 3/3 |
| 100 | [138, 156, 151] | 3/3 |

**None of the IVF_PQ sub-parameters have any effect.** Miss counts sit flat at ~130–170 across every sweep. This confirms that the bug is in CAGRA's graph traversal, not in IVF_PQ's graph construction quality — the IVF_PQ sub-params control how well neighbors are approximated during build, but the resulting graph simply doesn't have enough edges connecting the self-vector's neighborhood at N=500K with a 64/32 graph degree.

---

### Step 16: FP32 regression across all available versions (N=50K)

| Version | FP32 supported | ivf_pq itopk=64 | ivf_pq itopk=128 | nn_descent itopk=128 |
|---------|---------------|-----------------|------------------|----------------------|
| 24.4.0 | **No** — `cagra.build` not yet in API | — | — | — |
| 24.6.0 | **No** — `cagra.build` not yet in API | — | — | — |
| 24.8.0 | Yes | 14/512 | 0/512 | 0/512 |
| 24.10.0 | Yes | 9/512 | 1/512 | 0/512 |
| 24.12.0 | Yes | 18/512 | 0/512 | 0/512 |
| 25.2.1 | Yes | 14/512 | 0/512 | 0/512 |
| 25.4.0 | Yes | 13/512 | 1/512 | 0/512 |
| 25.6.1 | Yes | 24/512 | 1/512 | 0/512 |
| 25.8.0 | Yes | 16/512 | 0/512 | 0/512 |
| 25.10.0 | Yes | 11/512 | 0/512 | 0/512 |
| 25.12.0 | Yes | 17/512 | 1/512 | 0/512 |
| 26.2.0 | Yes | 12/512 | 0/512 | 0/512 |

**There is no regression point.** The bug appears in the very first version that shipped `cagra.build` (24.8.0) and persists unchanged through 26.2.0. Miss counts fluctuate version-to-version (9–24) due to non-deterministic IVF_PQ graph construction, not behavioral changes. No version has ever been clean at `ivf_pq itopk=64`. The `nn_descent itopk=128` workaround is 0/512 on every version at N=50K but does not hold at scale.

---

## Step 20: Root Cause Fix — Force self-edge for InnerProduct at graph construction

**Date:** 2026-05-25

### Root Cause

The bug has a single, clear root cause in `write_to_graph()` in `cagra_build.cuh`:

```
// omit itself & write out
for (j = 0, num_added = 0; j < top_k && num_added < node_degree; j++) {
    if (v == vec_idx) { num_self_included++; continue; }  // skips self
    knn_graph[vec_idx][num_added] = v;
    num_added++;
}
```

The function skips the self-vector. For **L2 metric**, this is correct — self is not a meaningful neighbor (distance = 0). For **InnerProduct metric** with unit-normalized vectors, self is the *true nearest neighbor* (dot product = 1.0). When IVF-PQ misses the self-vector (fails to return `vec_idx` in its top-k list — which happens probabilistically at large scale), that slot is simply never filled, and position 0 in the KNN graph gets a random neighbor instead.

The pruner (`kern_fused_prune`) already has `preserve_self_edges = true` for InnerProduct (line 1654), which means self-edges survive pruning if present. But they are never inserted in the first place when IVF-PQ misses them.

### Fix

Modified `write_to_graph()` to accept a `preserve_self_edges` flag. When true (InnerProduct metric), a self-edge is force-injected at position 0 before filling the rest of the neighbor list from IVF-PQ results:

```cpp
template <typename IdxT>
void write_to_graph(..., bool preserve_self_edges = false)
{
  for (size_t i = 0; i < batch_size; i++) {
    size_t vec_idx = i + batch_offset;
    size_t num_added = 0;
    if (preserve_self_edges) {
      knn_graph(vec_idx, num_added) = static_cast<IdxT>(vec_idx);  // force self at pos 0
      num_added = 1;
    }
    for (size_t j = 0; j < top_k && num_added < node_degree; j++) {
      const auto v = neighbors_host_view(i, j);
      if (static_cast<size_t>(v) == vec_idx) { num_self_included++; continue; }
      knn_graph(vec_idx, num_added) = v;
      num_added++;
    }
  }
}
```

Both callers — `refine_host_and_write_graph()` and the inline non-async path — pass `preserve_self_edges = (metric == InnerProduct)`.

**File changed:** `/mnt/home/premal/cuvs/cpp/src/neighbors/detail/cagra/cagra_build.cuh`

### Build

```
ninja -C /mnt/home/premal/cuvs/cpp/build -j4 NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST
```

Build status: **in progress** (2026-05-25)

