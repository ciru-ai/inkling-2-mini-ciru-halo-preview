# Inkling 2 Mini — Ciru Wave 5.2 Benchmarks, Quality Audit, and Roofline Analysis

All measurements below were recorded on Ciru: **AMD Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`, 20 WGPs / 40 CUs, 128 GB unified LPDDR5X)**, NixOS, Python `3.12.12`, PyTorch `2.13.0+rocm10.0.0` (HIP `7.15.26333`), Triton `3.8.0+git669b31ac`, vLLM `0.34.0+rocm100`.

---

## 1. Key kernel optimizations in Wave 5.2

1. **Pre-tiled FP16 WMMA MoE (`turbo_kernels.py`):**
   - Prefill (`M=64, N=128, K=64`) and small-batch/DSpark decode (`M=16, N=128, K=64`) with FP16 matrix-core WMMA (`tl.dot` on `gfx1151`), inline NVFP4 / C1 / mixed unpacking, and fused gated softcap activation.
   - Enabled for both HIP-graph C1 verification (`M <= 8`) and eager small-batch execution (`2 <= M <= 15`, unlocking 1.8× faster probabilistic DSpark).
2. **Tiled RDNA 3.5 FlashAttention (`turbo_kernels.py`):**
   - Local sliding-window prefill (`bq=64, bn=32, nw=4`) and global relative-bias FlashAttention (`PAGE=16/64`), plus split-KV decode attention (`turbo_rel_decode_m8_attention`).
3. **Target & Draft W6 / W4 WMMA + Draft HIP Graph:**
   - W6 WMMA for `QKVR`, `WO`, and `lm_head` on verification batches (`2 <= M <= 8`).
   - W4 WMMA for DSpark draft `base_head`, `MLP`, and `Markov` heads, plus warmup-phase DSpark draft HIP graph capture.
4. **Fused Loop & Residual Ops:**
   - `Q=3` shared-stack `attn_res` kernel (2.38× vs. unbatched per-query dispatches), fused loop prefill blend (1.60×), and `fast_route_m8` fused router alignment.

---

## 2. Prompt processing (PP) scaling — Wave 5.2 vs. Baseline ROCm port

Measured in Ciru's C1-fitted profile (`--batch-tokens 4096`, DSpark `K=7`):

| Prompt Length | Baseline ROCm port | Wave 5.2 Turbo | Speedup | Notes |
|---:|---:|---:|---:|---|
| 128 tok | 473–573 tok/s | **700.8 tok/s** | +22–48% | Launch-overhead bound |
| 512 tok | 852.0 tok/s | **1,279.6 tok/s** | **+50.2%** | |
| 2,048 tok | 1,024.7 tok/s | **1,645.0 tok/s** | **+60.5%** | |
| 3,840 tok | 1,005.5 tok/s | **1,691.3 tok/s** | **+68.2%** | Peak prefill efficiency |
| 8,192 tok | 889.9 tok/s | **1,640.8 tok/s** | **+84.4%** | |
| 32,768 tok | 780.1 tok/s | **1,248.3 tok/s** | **+60.0%** | |
| 65,536 tok | 646.8 tok/s | **788.5 tok/s** | **+21.9%** | Known preview drop-off (global attention $O(N^2)$ + thermal throttling) |

---

## 3. Token generation (TG) — Wave 5.2 vs. Baseline ROCm port

| Workload | Baseline ROCm port | Wave 5.2 Turbo | Speedup |
|---|---:|---:|---:|
| C1 Held Pooled (DSpark `K=7`, 4K ctx) | 64.15 tok/s (59.51 ms/rd) | **74.13 tok/s** (48.89 ms/rd) | **+15.6%** |
| — `order` (JSON) | — | **97.7–101.2 tok/s** | — |
| — `code` (`bisect_left`) | — | **70.8 tok/s** | — |
| — `math` | — | **69.0 tok/s** | — |
| 32K Context TG (DSpark `K=7`) | 49.41 tok/s | **63.69 tok/s** | **+28.9%** |
| 64K Context TG (DSpark `K=7`) | 36.38 tok/s | **54.08 tok/s** | **+48.7%** |
| Probabilistic DSpark (`prob`, eager, `K=7`) | ~20 tok/s | **36.51 tok/s** (`order` 55.52) | **~1.8×** |
| C8 Eager Aggregate (8 concurrent, greedy `K=7`) | 88.58 tok/s | **168.46 tok/s** *(Wave 6.0; Wave 5.2 was 88.92)* | **+90.2% (1.90×)** |
| C8 Eager Aggregate (8 concurrent, `T=0.7`) | — | **173.83 tok/s** *(Wave 6.0)* | — |

### Unmodified Hugging Face checkpoint (`EXPERT_POLICY=source`, no fitted banks)

Measured directly on `thinkingmachines/Inkling-2-Mini` (`6e7c8248`) with stock MXFP8 coda layers (20–23) and Wave 5.2 pre-tiled FP16 WMMA on all 20 NVFP4 layers (0–19):

- **`order` (JSON):** `84.12 tok/s` (TTFT `0.189 s`, exact JSON)
- **`tool` (`get_weather`):** `79.73 tok/s` (exact native tool call)
- **`p3500` (3,900 prompt tokens):** `1,494.6 PP tok/s` (TTFT `2.609 s`), `72.75 TG tok/s`
- **`code` (`bisect_left`):** `69.79 tok/s` (exact Python implementation)
- **`math`:** `54.61 tok/s`

---

## 4. Full-vocabulary numerical fidelity & functional audit

Evaluated across **2,044 teacher-forced tokens** over the full 200,058-token vocabulary against the BF16 reference:

| Metric | Baseline ROCm port | Wave 5.2 Turbo | Delta |
|---|---:|---:|---:|
| Mean forward KL | 0.00784 nats | **0.00948 nats** | +0.00164 nats |
| Max single-token KL | 0.2713 nats | **0.2737 nats** | +0.0024 nats |
| Mean ΔNLL | — | **+0.00316 nats** | +0.12% of 2.48 NLL |
| Top-1 token agreement | 94.32% | **93.74%** | -0.58% |

Functional verification checks (all passed on Wave 5.2):
- `order`: exact JSON (`ok=True`, schema valid)
- `math`: exact integer answer (`28`)
- `code`: passed all **7,392 exhaustive `bisect_left` test cases**
- Tool calling (`get_weather`) + `inkling` reasoning parser: exact tool name, JSON args, and separated reasoning stream.

---

## 5. Realistic Multi-Agent Vision Workload (8-Agent AMD Logo 3D Studio) & GPU Trace

Tested end-to-end on the **8-agent AMD Logo 3D Studio** (`showcase` workflow, native ROCm vision enabled, native `inkling` reasoning & strict tool calling, probabilistic `T=0.7`, `prob --seqs 8 --context 32768 --batch-tokens 4096 --kv-mib 24576 --prefix-cache --tools --vision --turbo`):

| Measurement | Wave 5.2 | Wave 6.0 | Delta |
|---|---:|---:|---:|
| Workflow outcome | **Complete** (rev 17) | **Complete** (rev 14) | All 8 roles autonomous |
| Total wall time (empty scene -> approved 3D scene) | 498.22 s (8m 18s) | **404.72 s (6m 45s)** | **-93.5 s (-18.8%)** |
| First visible 3D contribution | 121.43 s | **73.35 s** | **-48.1 s (1.66× faster)** |
| Vision-bearing model requests | 37 (`0` errors) | 27 (`0` errors) | — |
| Peak concurrent active agents | 8 | 8 | — |
| Total generated tokens (workflow) | 38,778 | 34,898 | — |
| Whole-workflow output throughput | 77.83 tok/s | **86.23 tok/s** | **+10.8%** |
| Aggregate active-decode TG | 77.37 tok/s | **86.72 tok/s** | **+12.1%** |
| Pooled per-request TG | 17.38 tok/s | **21.37 tok/s** | **+23.0%** |
| Controlled C8 Greedy (`T=0`, 8 concurrent) | 88.92 tok/s | **168.46 tok/s** | **+89.5% (1.90×)** |
| Controlled C8 Prob (`T=0.7`, 8 concurrent) | — | **173.83 tok/s** | — |

### Live 8-Agent ROCm GPU Profiler Breakdown (64 MB trace, `M=64` peak concurrency)

Captured during peak 8-agent concurrency (`4.487 s` span, `3.646 s` GPU busy = **81.3% busy**, `30,346` GPU kernel launches across 18 rounds):

| Kernel / Category | GPU Time | Share of Busy | Launches | Root Cause & Next Kernel Fix |
|---|---:|---:|---:|---|
| `_source_wmma_turbo_kernel` (NVFP4 MoE) | 1,145.2 ms | 31.4% | 1,260 | Already fast (`909 us/call`) |
| `_packed_prefill_kernel` (Mixed/Coda MoE fallback) | **819.5 ms** | **22.5%** | 468 | **`CODA_WMMA_PRETILE` / `MIXED_WMMA_PRETILE` gated on `target_graph=1`** (`1,751 us/call`, ~2× slower than turbo!) |
| `hipBLASLt/rocBLAS` BF16 GEMM (`QKVR`/`WO`/`LM_head`/`DSpark`) | **731.3 ms** | **20.1%** | 5,328 | **W6/W4 WMMA gated on `M <= 8`** (`0.22 ms` quant linear vs `731 ms` BF16 at `M=64`!) |
| `_rel_attention_kernel` (unbatched decode fallback) | **636.8 ms** | **17.5%** | 918 | **`turbo_rel_decode_m8_attention` gated on `B == 1`** (`694 us/call`) |
| Host launch gap (GPU idle) | **841.0 ms** | *(18.7% wall)* | — | Eager mode (`1,686` launches/round, no multi-req HIP graphs) |
| `_fused_loop_blend_p1_kernel` + sampling (`topk_topp`/`tanh`) | 151.2 ms | 4.1% | 918 | Uncoalesced `M=64` loop blend + 4-pass 200K vocab sampling |

---

## 6. Single-stream GPU roofline & prioritized next kernel targets

Single-stream C1 DSpark `K=7` traces show **94.2% GPU utilization** (`35.8 ms/round` = 71% MoE sustaining >90–95% of Strix Halo's ~240 GB/s physical DRAM read ceiling; `3.3 ms/round` host gap).

### Highest-ROI next kernels (from live 64K + 8-agent AMD Logo 3D traces)

1. **Enable Pre-Tiled Coda & Mixed FP16 WMMA in Eager `c8`/`prob` (`~400 ms / 11%` GPU saved immediately):** Remove the `target_graph` gate on `CODA_WMMA_PRETILE` and `MIXED_WMMA_PRETILE` so the 13 mixed/coda visits per round stop falling back to `_packed_prefill_kernel`.
2. **Batched `M=9..64` W6 & W4 WMMA (`~450–500 ms / 13%` GPU saved):** Generalize `w6_wmma_linear` (`QKVR`, `WO`, `lm_head`) and W4 DSpark (`base_head`, `MLP`, `Markov`) beyond `M <= 8` so concurrent multi-agent batches read 4–6 bit weights instead of BF16.
3. **Batched Multi-Request Split-KV Decode Attention (`~450 ms / 12%` GPU saved):** Extend `turbo_rel_decode_m8_attention` from `seq_lens.shape[0] == 1` to `grid = (B, H, splits)` to replace `_rel_attention_kernel` during concurrent decoding.
4. **Multi-Request HIP Graph Capture Buckets (`reqs = 1, 2, 4, 8`) (`~650 ms` host gap saved):** Eliminates the 18.7% CPU launch idle time across 1,686 kernel dispatches/round in eager `c8`/`prob`.
5. **64K Global Attention Tiling:** Improve tiling/split-K and register occupancy in the 8 global attention visits for 32K–64K+ prefills to close the 64K PP drop-off (`1,248 -> 789 tok/s`).
6. **Fused Multi-Token Loop Blend & Logits+Softcap+Sampling:** Optimize `fused_loop_blend` for `M > 8` and fuse `tanh` softcap + temperature + top-k/top-p over the 200K vocabulary.

**Combined projected multi-agent impact (items 1–4):** `4.49 s -> ~2.5 s` per 18 rounds (**~1.8× faster concurrent multi-agent throughput**, lifting 8-agent aggregate TG from ~77–89 tok/s to **~140–160+ tok/s**).
