# Inkling 2 Mini — Ciru Wave 6.3 Benchmarks, Quality Audit, and Roofline Analysis

All measurements below were recorded on Ciru: **AMD Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`, 20 WGPs / 40 CUs, 128 GB unified LPDDR5X)**, NixOS, Python `3.12.12`, PyTorch `2.13.0+rocm10.0.0` (HIP `7.15.26333`), Triton `3.8.0+git669b31ac`, vLLM `0.34.0+rocm100`.

---

## 1. Optimization evolution (Wave 5.2 → Wave 6.3)

1. **Wave 5.2 (Single-Stream Foundation):**
   - Pre-tiled FP16 WMMA MoE (`M=64, N=128, K=64` prefill; `M=16, N=128, K=64` decode), tiled RDNA 3.5 local/global FlashAttention, W6 WMMA `QKVR`/`WO`/`lm_head`, W4 DSpark draft `base_head`/`MLP`/`Markov`, `Q=3` shared-stack `attn_res`, and `fast_route_m8`.
2. **Native ROCm `gfx1151` Vision:**
   - Ported Inkling 2 Mini's PE-Lang ViT (`model.visual.*`, 422 tensors) to ROCm Triton (`ops/vision.py` with IEEE-754 RNE uint32 BF16 rounding replacing CUDA PTX) + ROCm SDPA (`vision.py`).
3. **Wave 6.0 (Multi-Agent Concurrent Kernels):**
   - Unlocked pre-tiled Coda & Mixed FP16 WMMA in eager `c8`/`prob`, added zero-spill FP16-scaled batched Gluon WMMA (`M <= 64`) for `QKVR`, `lm_head`, DSpark `base_head`, `MLP`, and `Markov`, batched multi-request split-KV decode attention, and 9× D-split loop blend (`C8` jumped `88.92 → 168.46 tok/s`, 8-agent vision studio dropped `498.2 s → 404.7 s`).
4. **Wave 6.1–6.2b (64K Prefill Cliff Fix + Multi-Request FULL Target & Draft HIP Graphs):**
   - **Hybrid AOTriton-Far + Triton-Near Global Prefill (`turbo_kernels.py`):** On global layers (`KVH=8`), far keys (`past - E_PAD >= 2048`) have zero relative bias or causal mask and run via `_gather_paged_kv_hnd_kernel` + native ROCm AOTriton FlashAttention-2 (`aten._scaled_dot_product_flash_attention`, GQA direct, returning `out_far` and `lse_far`), merged seamlessly into `_turbo_prefill_rel_attn_near_merge_kernel` (`6.46×` faster per global layer at 64K: `861.8 ms → 133.4 ms`, `0.0000` BF16 max error).
   - **True `KVH=8` Global Decode Split Schedule:** `10 / 5 / 6 / 6` splits for `B = 1 / 2 / 4 / 8`, saturating `238.1 GB/s` at `B=1` 64K (`9.02 ms` across 8 global layers) and `385 GB/s` effective at `B=4..8`.
   - **Multi-Request FULL Target + Draft HIP Graphs (`reqs = 1..8`, Greedy & Prob):** Reclaimed `~49 ms/step` of eager host launch overhead (`65.3% → 95.2%` GPU busy across `1,642` kernels/step at `B=8`, captured in `1 s` / `0.34 GiB`).
5. **Wave 6.3 (Dual-Depth Adaptive DSpark HIP Graphs):**
   - Captures two DSpark graph managers against one shared model & KV cache (`num_speculative_tokens_per_batch_size: [[1, 1, 7], [2, 8, 4]]`, `9` Target + `9` Draft FULL HIP Graphs in `1 s` / `0.47 GiB`), unlocking `B=1` fused W4 Markov argmax, W8 draft backbone, and fused M8 loop inside `--seqs 8` servers (**100.23 tok/s at `B=1` and 310.20 agg tok/s at `B=8` from one server**).

---

## 2. Long-context PP & TG scaling (4K → 62K, cliff eliminated)

### A. Global prefill kernel microbenchmark (true `KVH=8`, `q_len=4096`, permuted paged KV)

| Past context | Triton-only (`ms/layer`) | **Hybrid AOTriton+Triton (`ms/layer`)** | Speedup | 8-layer chunk save | BF16 max err |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 31.24 ms | **25.31 ms** | **1.23×** | 47.4 ms | `0.0001` |
| 8,192 | 55.84 ms | **33.26 ms** | **1.68×** | 180.7 ms | `0.0001` |
| 16,384 | 125.92 ms | **49.10 ms** | **2.56×** | 614.6 ms | `0.0001` |
| 32,768 | 341.17 ms | **79.25 ms** | **4.31×** | 2,095.4 ms | `0.0001` |
| 49,152 | 662.89 ms | **110.19 ms** | **6.02×** | 4,421.6 ms | `0.0000` |
| **61,440 (64K)** | 861.83 ms | **133.41 ms** | **6.46×** | **5,827.4 ms** | **`0.0000`** |

### B. Live end-to-end 4K → 62K curve (`long --context 65536 --batch-tokens 4096 --turbo`)

| Prompt Length | TTFT | Wave 5.2 PP | **Wave 6.3 PP** | PP Delta | **Wave 6.3 TG** |
|---:|---:|---:|---:|---:|---:|
| 4,076 tok | 2.70 s | 1,691.3 *(3.8K)* | **1,507.4 tok/s** | — | 56.1–100.7 tok/s |
| 15,954 tok | 11.03 s | — | **1,445.8 tok/s** | — | **70.5 tok/s** |
| 32,070 tok | 23.19 s | 1,248.3 tok/s | **1,383.0 tok/s** | **+10.8%** | **52.4 tok/s** |
| 47,977 tok | 36.23 s | — | **1,324.2 tok/s** | — | **59.4 tok/s** |
| **62,024 tok** | **48.67 s** | 788.5 tok/s *(64K)* | **1,274.4 tok/s** | **+61.6%** | **46.9 tok/s** |

### C. Speculative decoding at 62K (`K=0 --no-spec` vs. `K=4` vs. `K=7`, 256 output tokens)

| Arm (`long --context 65536 --turbo`) | 32K PP | 32K TG (`T=0`) | 62K PP | 62K TG (`T=0`) | 62K TG (`T=0.7`) | 62K Total (`T=0`) |
|---|---:|---:|---:|---:|---:|---:|
| `K=0` (`--no-spec`, HIP graph) | **1,447.8** | 36.93 | **1,324.2** | 30.73 | 30.73 | 55.15 s |
| **`K=4` (DSpark + HIP graph)** | 1,382.2 | **50.87** (`+38%`) | 1,270.5 | **44.13** (`+44%`) | 33.98 (`+11%`) | **54.61 s** |
| **`K=7` (DSpark + HIP graph)** | 1,364.3 | 41.50 (`+12%`) | 1,254.1 | **44.49** (`+45%`) | **41.07** (`+34%`) | 55.20 s |

Speculative decoding remains a strong net win at 62K (`+44–45%` TG) because it amortizes the 2.15 GB (`9.02 ms/step`) target global KV-cache read across ~3 accepted tokens/step, while draft KV population adds only `1.98 s` (`4.2%`) across a 62K prompt.

---

## 3. Speculative depth sweep (`K = 0..7 × B = 1..8`) & Multi-Request HIP Graphs

### A. Eager probabilistic depth sweep (`prob --seqs 8`, `T=0.7`, Wave 6.0 kernels)

| Depth `K` | `B=1` | `B=2` (agg / req) | `B=4` (agg / req) | `B=8` (agg / req) | Mean Accepted / Draft |
|---|---:|---:|---:|---:|---:|
| `K=0` (`--no-spec`) | 33.40 | 50.45 / 25.22 | 73.23 / 18.31 | 104.61 / 13.08 | 0.00 |
| `K=1` | 37.71 | 60.15 / 30.07 | 85.15 / 21.29 | 128.06 / 16.01 | 0.77 |
| `K=2` | 54.88 | 63.97 / 31.99 | 97.41 / 24.35 | 158.77 / 19.85 | 1.40 |
| `K=3` | 49.96 | **85.58 / 42.79** | 102.36 / 25.59 | 177.78 / 22.22 | 1.85 |
| **`K=4`** | **74.46** | 65.80 / 32.90 | **107.46 / 26.87** | **228.06 / 28.51** | **2.36** |
| `K=7` | 57.59 | 68.20 / 34.10 | 95.95 / 23.99 | 196.36 / 24.54 | 2.58 |

### B. Wave 6.3 FULL Target + Draft HIP Graphs (`reqs = 1..8`)

| Mode | `B=1` | `B=2` (agg / req) | `B=4` (agg / req) | `B=8` (agg / req) | `B=1` return |
|---|---:|---:|---:|---:|---:|
| **`c8` Adaptive `K=7/K=4` (`T=0`)** | **100.23 tok/s** | 93.42 / 46.71 | **203.42 / 50.85** | **310.20 / 38.78** | **97.78 tok/s** |
| `c8` Fixed `K=4` (`T=0`) | 77.04 tok/s | 89.01 / 44.50 | 201.83 / 50.46 | **316.64 / 39.58** | — |
| `c1` Dedicated (`K=7`, `T=0`) | **100.68 tok/s** | — | — | — | — |
| **`prob` `K=4` (`T=0.7`)** | **86.52 tok/s** | **101.36 / 50.68** | 136.01 / 34.00 | **238.73 / 29.84** | — |

---

## 4. Realistic Multi-Agent Vision Workload (8-Agent AMD Logo 3D Studio)

Tested end-to-end on the **8-agent AMD Logo 3D Studio** (`showcase` workflow, native ROCm vision enabled, native `inkling` reasoning & strict tool calling, `T=0.7`, `prob --seqs 8 --prefix-cache --tools --vision --turbo`):

| Measurement | Wave 5.2 | Wave 6.0 | **Wave 6.2b / 6.3** |
|---|---:|---:|---:|
| **Pooled per-request TG** | 17.38 tok/s | 21.37 tok/s | **25.16 tok/s** (**+44.8%** vs. W5.2) |
| **Whole-workflow output rate** (incl. vision, PP, tools, renders) | ~70.0 tok/s | 86.23 tok/s | **92.5 tok/s** |
| `conc=2–3` (12K–22K context) pooled / peak TG | — | — | **31.0–31.8 / 53.1 tok/s** |
| Cold 30.7K PP / Cached 29.8K TTFT | — | — | **1,581 tok/s** / **1.04 s** |
| Prefix cache hit rate / Per-draft acceptance (`K=4, T=0.7`) | — | — | **73.3%** / **62.9%** (`3.51` tok/step) |
| Controlled C8 Greedy (`T=0`, `B=8`) | 88.92 tok/s | 168.46 tok/s | **310.20–316.64 tok/s** (**3.5×**) |

---

## 5. GPU Profiler Roofline (Wave 6.2b / 6.3)

- **4K Prefill Chunk:** `2,404 ms` wall, `2,385 ms` GPU busy (**99.2% GPU utilization** across `1,503` kernels; 53.4% FP16 WMMA MoE, 13.1% attention, 11.4% rocBLAS dense projections).
- **Steady-State `B=8, K=4` Target + Draft FULL HIP Graph Step:** `98.4 ms` wall, `93.5 ms` GPU busy (**95.2% GPU utilization**, only `4.8 ms` host idle across `1,642` kernel launches — down from `54.1 ms` / `37.1%` idle before multi-request HIP graphs).
- **Remaining Hardware Ceiling:** Routed MoE (`_source_wmma_turbo_kernel` + `_mixed_wmma_turbo_kernel` + `_coda_wmma_turbo_kernel`) now accounts for **~69% of GPU time at `B=8`** and **~53% at 4K prefill**, approaching Strix Halo's physical LPDDR5X read bandwidth (`~238 GB/s`) and RDNA 3.5 WMMA throughput.

---

## 6. Full-vocabulary numerical fidelity & functional audit

Evaluated across **2,044 teacher-forced tokens** over the full 200,058-token vocabulary against the BF16 reference:

| Metric | Baseline ROCm port | Ciru Turbo | Delta |
|---|---:|---:|---:|
| Mean forward KL | 0.00784 nats | **0.00948 nats** | +0.00164 nats |
| Max single-token KL | 0.2713 nats | **0.2737 nats** | +0.0024 nats |
| Mean ΔNLL | — | **+0.00316 nats** | +0.12% of 2.48 NLL |
| Top-1 token agreement | 94.32% | **93.74%** | -0.58% |

All functional checks passed across Wave 5.2 → Wave 6.3: exact JSON (`order`), math (`17 * 23 = 391`, `28`), exhaustive 7,392-case Python `bisect_left`, native `inkling` reasoning & tool calling (`get_weather`), native ROCm vision (`ornith-bot.png`, `amd-logo.png`, 3D scene frames), and 62K long-context + `B=1 → 8 → 1` adaptive HIP graph transitions.
