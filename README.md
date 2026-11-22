# Inkling 2 Mini — Ciru Strix Halo (`gfx1151`) Preview

**Fast local ROCm / vLLM inference preview for Thinking Machines Lab's Inkling 2 Mini on AMD Ryzen AI Max+ Strix Halo (`gfx1151`).**

This GitHub preview packages Ciru's **Wave 6.3** AMD ROCm backend, custom Triton/Gluon FP16 WMMA / W6 / W4 / hybrid AOTriton+Triton attention kernels, multi-request FULL Target & Draft HIP Graphs (`reqs = 1..8`), dual-depth adaptive DSpark speculation (`K=7 @ B=1`, `K=4 @ B=2..8`), native ROCm `gfx1151` vision, portable installer, and serving launcher for [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (pinned revision `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`, Apache-2.0).

> **Preview status (GitHub only — not yet on Hugging Face):**
> - **64K Prefill Cliff Eliminated (Wave 6.1–6.2b):** Hybrid AOTriton FlashAttention-2 (far keys) + Triton online-softmax near-window merge accelerates global prefill attention up to **6.46× at 64K** (`861.8 ms → 133.4 ms`, `0.0000` BF16 max error), lifting 62K prompt throughput from **788.5 → 1,274.4 tok/s (+61.5%)** with a smooth linear curve from 4K (`1,507 tok/s`) through 62K (`1,274 tok/s`).
> - **Multi-Request FULL Target + Draft HIP Graphs & Adaptive DSpark `K=7/K=4` (Wave 6.2–6.3):** One single `--seqs 8` server captures 9 Target + 9 Draft FULL HIP Graphs in **1 second (`0.47 GiB`)**, hitting **95.2% GPU busy** at `B=8` and automatically running **`K=7` (`100.2 tok/s`) at `B=1`** and **`K=4` (`310.2–316.6 agg tok/s`, `39 tok/s` per user — 3.5× vs. Wave 5.2) at `B=8`**, plus **238.7 agg tok/s** at `B=8, T=0.7` probabilistic mode.
> - **Native ROCm Vision, Reasoning, and Tool Calling:** Inkling 2 Mini's PE-Lang ViT vision encoder (`model.visual.*`), `inkling` reasoning parser, and native tool calling (`--enable-auto-tool-choice --tool-call-parser inkling`) run natively on AMD ROCm `gfx1151`. Pass `--language-model-only` to `scripts/serve.sh` if you prefer text-only mode.
> - **Custom runtime required:** Stock `pip install vllm` does not include the `gfx1151` Inkling 2 Mini AMD backend or Wave 6.3 kernels. Use the installer below with a ROCm 10 / vLLM 0.34 environment (or our full `ciru-ai/vllm` branch `inkling-2-mini-gfx1151-turbo-preview`).

[Installation Guide](INSTALL.md) · [Detailed Benchmarks & Quality Audit](BENCHMARKS.md) · [Full vLLM Branch](https://github.com/ciru-ai/vllm/tree/inkling-2-mini-gfx1151-turbo-preview)

---

## Why this build

- **Smooth, flat prefill from 4K to 64K (`1,507 → 1,274 tok/s`):** Pre-tiled FP16 WMMA MoE prefill (`M=64, N=128, K=64`) at **99.2% GPU utilization** combined with Hybrid AOTriton-Far + Triton-Near global relative-bias attention (`6.46×` faster per global layer at 64K) eliminates the Strix Halo 32 MB MALL spill cliff.
- **One adaptive server for both single-user and 8-agent workloads:** Dual-depth adaptive DSpark HIP Graphs (`K=7` at `B=1`, `K=4` at `B=2..8`) + zero-spill batched `M<=64` FP16 W6/W4 WMMA (`QKVR`, `lm_head`, DSpark `base_head`/`MLP`/`Markov`) + true `KVH=8` split-KV decode attention (`238 GB/s` at `B=1`, `385 GB/s` effective at `B=4..8`) reach **100.2 tok/s at `B=1`** and **310.2–316.6 agg tok/s at `B=8`** (`95.2%` GPU utilization, down from 37% eager CPU launch idle).
- **Verified numerical fidelity:** Full-vocabulary 2,044-token KL audit against BF16 teacher forcing measured **0.00948 nats** mean forward KL, **+0.00316 nats** ΔNLL (0.12% of 2.48 NLL), and **93.74% top-1 agreement**, with exact JSON, math, exhaustive 7,392-case Python code, vision, reasoning, and tool-calling passes.
- **Works directly on stock Hugging Face weights:** Zero private weight files are required to install and run this preview — point `scripts/serve.sh` at `thinkingmachines/Inkling-2-Mini` and go.

---

## Measured performance on Strix Halo (`gfx1151`)

Measured on AMD Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`, 20 WGPs / 40 CUs, 128 GB unified RAM, NixOS, Python 3.12.12, PyTorch `2.13.0+rocm10.0.0`, Triton `3.8.0`, vLLM `0.34.0+rocm100`), `--batch-tokens 4096`.

### 1. Wave 6.3 Unified Multi-Agent Adaptive Server (`c8 --seqs 8 --turbo`, `K=7 @ B=1` / `K=4 @ B=2..8`) & `prob` (`T=0.7`, `K=4`)

| Workload (`reqs = 1..8` FULL Target + Draft HIP Graphs) | `B=1` | `B=2` (agg / req) | `B=4` (agg / req) | `B=8` (agg / req) | vs. Wave 5.2 (`B=8`) |
|---|---:|---:|---:|---:|---:|
| **`c8` Adaptive `K=7/K=4` (`T=0`, one server)** | **100.23 tok/s** | 93.42 / 46.71 | **203.42 / 50.85** | **310.20 / 38.78** | **3.49×** (`88.9 → 310.2`) |
| `c8` Fixed `K=4` (`T=0`) | 77.04 tok/s | 89.01 / 44.50 | 201.83 / 50.46 | **316.64 / 39.58** | **3.56×** (`88.9 → 316.6`) |
| **`prob` `K=4` (`T=0.7` probabilistic)** | **86.52 tok/s** | **101.36 / 50.68** | 136.01 / 34.00 | **238.73 / 29.84** | **2.68×** (`88.9 → 238.7`) |
| Real 8-Agent Vision AMD Logo 3D Studio (`prob`, vision+tools) | — | — | — | **92.5 tok/s** whole-workflow (`25.16` pooled per-req, peak `53.1/req`) | **+44.8%** pooled TG (`17.38 → 25.16`) |

### 2. Smooth 4K → 62K Long-Context PP & TG (`long --context 65536 --turbo`)

| Prompt Length | Wave 5.2 PP | **Wave 6.3 PP** | PP Speedup | **Wave 6.3 TG (`K=7` / `K=4`)** | `--no-spec` (`K=0`) TG |
|---:|---:|---:|---:|---:|---:|
| 4,076 tok | 1,691.3 *(3.8K)* | **1,507.4 tok/s** *(TTFT 2.70 s)* | — | 56.1–100.7 tok/s | 48.5 tok/s |
| 15,954 tok | — | **1,445.8 tok/s** *(TTFT 11.03 s)* | — | **70.5 tok/s** | — |
| 32,070 tok | 1,248.3 tok/s | **1,383.0 tok/s** *(TTFT 23.19 s)* | **+10.8%** | **50.9–52.4 tok/s** | 36.9 tok/s |
| 47,977 tok | — | **1,324.2 tok/s** *(TTFT 36.23 s)* | — | **59.4 tok/s** | — |
| **62,024 tok** | 788.5 tok/s *(64K)* | **1,274.4 tok/s** *(TTFT 48.67 s)* | **+61.6%** | **44.5–46.9 tok/s** (`+45%` vs `K=0`) | 30.7 tok/s |

### 3. Out-of-the-box unmodified Hugging Face checkpoint (`thinkingmachines/Inkling-2-Mini`, `EXPERT_POLICY=source`)

No private fitted banks; 20 NVFP4 layers use pre-tiled FP16 WMMA and 4 coda layers remain native MXFP8.

| Workload (`c1` HIP graph, DSpark `K=7`, `--tools`) | Prompt tokens | TTFT | PP (tok/s) | TG (tok/s) | Functional result |
|---|---:|---:|---:|---:|---|
| Structured JSON (`order`) | 53 | 0.189 s | 280.4 | **84.12** | Valid exact JSON |
| Tool calling (`get_weather`) | — | — | — | **79.73** | Exact `get_weather` call |
| Long prompt (`p3500`) | 3,900 | 2.609 s | **1,494.6** | **72.75** | Coherent summary |
| Python coding (`bisect_left`) | 42 | 0.212 s | 197.9 | **69.79** | Exact `bisect_left` |

See [BENCHMARKS.md](BENCHMARKS.md) for the complete `K=0..7 × B=1..8` speculative sweep, 62K `--no-spec` vs `K=4/7` proof, 2,044-token KL/NLL audit, and GPU profiler traces.

---

## Quick start

Prerequisites: Linux Strix Halo (`gfx1151`) with working `/dev/kfd` access and a ROCm 10 / Python 3.12 / vLLM `0.34.0+rocm100` virtual environment. See [INSTALL.md](INSTALL.md) for Ubuntu/Debian, Fedora, Arch, and NixOS details.

```bash
git clone https://github.com/ciru-ai/inkling-2-mini-ciru-halo-preview.git
cd inkling-2-mini-ciru-halo-preview

# 1. Verify environment and overlay SHA-256 manifest
bash scripts/install.sh --venv /path/to/vllm-034-rocm100 --verify-only

# 2. Install isolated overlay prefix and download thinkingmachines/Inkling-2-Mini
bash scripts/install.sh \
  --venv /path/to/vllm-034-rocm100 \
  --prefix "$HOME/.local/opt/inkling-2-mini-preview" \
  --download-model "$HOME/models/Inkling-2-Mini"

# 3a. Serve unified multi-agent adaptive mode (B=1 -> K=7 ~100 tok/s, B=2..8 -> K=4 ~310 tok/s, native vision & tools)
bash "$HOME/.local/opt/inkling-2-mini-preview/serve.sh" \
  --model "$HOME/models/Inkling-2-Mini" \
  --mode c8 --context 65536 --batch-tokens 4096 --prefix-cache --tools

# 3b. Or serve probabilistic T>0 multi-agent mode (K=4, FULL Target+Draft HIP graphs)
bash "$HOME/.local/opt/inkling-2-mini-preview/serve.sh" \
  --model "$HOME/models/Inkling-2-Mini" \
  --mode prob --context 65536 --batch-tokens 4096 --prefix-cache --tools

# 4. Run smoke test in another terminal
python3 scripts/smoke-test.py --base-url http://127.0.0.1:18081/v1 --model inkling-2-mini --check-tools
```

---

## What is included

| Path | Purpose |
|---|---|
| `overlay/vllm/` | Complete 47-file Ciru AMD ROCm backend, native `gfx1151` vision encoder, and Wave 6.3 Triton/Gluon kernel overlay (`vllm/models/inkling2/amd/`, scheduler/runner/dflash hooks) |
| `patches/vllm-inkling2-gfx1151.patch` | Clean unified git patch against vLLM commit `b426b46e` |
| `scripts/install.sh` | Portable verifier and installer (standalone `--prefix`, in-place `--venv` with backup, or `--vllm-source` patch mode) |
| `scripts/serve.sh` | Portable launcher supporting `c1`, `long`, `c8` (adaptive `K=7/K=4` + FULL Target & Draft HIP graphs), `prob` (`K=4` + FULL HIP graphs), native vision, and optional fitted banks |
| `scripts/smoke-test.py` | End-to-end JSON, math, 2K+ prefill, reasoning, and tool-calling verification script |
| `MANIFEST.sha256` | SHA-256 integrity manifest for all overlay files |
| `qualification/wave52-kl-summary.json` | 2,044-token full-vocabulary KL/NLL qualification summary |
| `INSTALL.md` / `BENCHMARKS.md` | Platform setup guide and full benchmark/roofline report |

---

## Credits and license

- **Upstream model:** [Thinking Machines Lab `Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Apache-2.0, revision `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`).
- **Upstream engine & references:** [vLLM](https://github.com/vllm-project/vllm) (`b426b46e`, PR #69028 and ROCm Small donor PR #67685), [Hugging Face Transformers](https://github.com/huggingface/transformers) (PR #51050), [Triton](https://github.com/triton-lang/triton), [PyTorch](https://pytorch.org), and [AMD ROCm](https://github.com/ROCm).
- **License:** Ciru-authored code, kernels, scripts, and documentation are Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
