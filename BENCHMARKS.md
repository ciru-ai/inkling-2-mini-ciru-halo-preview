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
| C8 Eager Aggregate (8 concurrent, greedy `K=7`) | 88.58 tok/s | **88.92 tok/s** | +0.4% |

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

## 5. Single-stream GPU roofline & next kernel targets

Torch ROCm profiler traces captured on C1 DSpark `K=7` decode show **94.2% GPU utilization**:
- **35.8 ms/round (71% of round time)** is spent inside MoE expert weight reads, sustaining **>90–95% of Strix Halo's ~240 GB/s physical DRAM read ceiling**.
- **3.3 ms/round** is host scheduling/launch gap.

### Where additional kernels will deliver the next major gains

1. **64K Global Attention & Thermal/Bandwidth Efficiency:** Improve tiling/split-K and register occupancy in the 8 global attention visits for 32K–64K+ prefills to eliminate the 64K PP drop-off (`1,248 -> 789 tok/s`).
2. **Multi-Agent / Concurrent (`seqs > 1`, `M > 8`) Kernels:**
   - Extend W6 WMMA (`QKVR`, `WO`, `lm_head`) beyond `M <= 8` so concurrent eager batches (`M = 16..64`) stop falling back to BF16.
   - Extend W4 DSpark `base_head` and `Markov` kernels to batched multi-request draft steps (`reqs > 1`).
   - Generalize `turbo_rel_decode_m8_attention` beyond `seq_lens.shape[0] == 1` so concurrent decode batches get split-KV wave32 attention instead of the slower generic decode kernel.
   - Add multi-request HIP graph capture buckets (`reqs = 1, 2, 4, 8`) for eager concurrent serving.
3. **ROCm Vision Encoder (`model.visual.*`):** Wire Inkling 2 Mini's vision tower to ROCm so multimodal agent workflows run natively without `--language-model-only`.
