# Inkling 2 Mini — Ciru Strix Halo (`gfx1151`) Preview

**Fast local ROCm / vLLM inference preview for Thinking Machines Lab's Inkling 2 Mini on AMD Ryzen AI Max+ Strix Halo (`gfx1151`).**

This GitHub preview packages Ciru's Wave 5.2 AMD ROCm backend, custom Triton FP16 WMMA / W6 / W4 / relative-attention kernels, portable installer, and serving launcher for [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (pinned revision `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`, Apache-2.0).

> **Preview status (GitHub only — not yet on Hugging Face):**
> - **Text-only (`--language-model-only`)**: Native reasoning (`inkling` parser) and native tool calling (`--enable-auto-tool-choice --tool-call-parser inkling`) work end-to-end. Vision weights (`model.visual.*`) are not yet wired to the ROCm backend.
> - **Known 64K prefill cliff**: Prompt processing reaches **1,691 tok/s at 4K**, **1,641 tok/s at 8K**, and **1,248 tok/s at 32K**, then drops to **789 tok/s at 64K** due to quadratic global-attention scaling across the 8 global visits and sustained thermal throttling.
> - **Multi-agent concurrent kernels in progress**: Single-stream C1 decode is near the Strix Halo physical DRAM ceiling (**74.13 tok/s** with C1 coda banks, **69.8–84.1 tok/s** on stock HF weights). Eager multi-request (`c8` / staggered agent) workloads currently fall back to unbatched paths on several ops; profiling and dedicated concurrent kernels are next.
> - **Custom runtime required**: Stock `pip install vllm` does not include the `gfx1151` Inkling 2 Mini AMD backend or Wave 5.2 kernels. Use the installer below with a ROCm 10 / vLLM 0.34 environment (or our full `ciru-ai/vllm` branch `inkling-2-mini-gfx1151-turbo-preview`).

[Installation Guide](INSTALL.md) · [Detailed Benchmarks & Quality Audit](BENCHMARKS.md) · [Full vLLM Branch](https://github.com/ciru-ai/vllm/tree/inkling-2-mini-gfx1151-turbo-preview)

---

## Why this build

- **Major prefill speedup (+50% to +84% vs. baseline ROCm port):** Pre-tiled FP16 WMMA MoE prefill (`M=64, N=128, K=64`) and tiled RDNA 3.5 local/global FlashAttention push 4K–8K prompt throughput from **890–1,025 tok/s** to **1,641–1,691 tok/s** (`1,495 tok/s` out of the box on unmodified Hugging Face weights).
- **Near-roofline single-stream decode (+15.6% at 4K, +48.7% at 64K):** Pre-tiled FP16 WMMA decode for DSpark `K=7` verification, W6 WMMA attention/LM-head projections, W4 DSpark draft head/MLP/Markov kernels, fused `Q=3` shared-stack block attention residuals, and HIP graphs on both target and draft reach **74.13 tok/s** (48.89 ms/round, 94.2% GPU busy) on Ciru's C1 profile and **69.8–84.1 tok/s** on unmodified stock Hugging Face weights.
- **Verified numerical fidelity:** Full-vocabulary 2,044-token KL audit against BF16 teacher forcing measured **0.00948 nats** mean forward KL, **+0.00316 nats** ΔNLL (0.12% of 2.48 NLL), and **93.74% top-1 agreement**, with exact JSON, math, exhaustive 7,392-case Python code, reasoning, and tool-calling passes.
- **Works directly on stock Hugging Face weights:** Zero private weight files are required to install and run this preview — point `scripts/serve.sh` at `thinkingmachines/Inkling-2-Mini` and go.

---

## Measured performance on Strix Halo (`gfx1151`)

Measured on AMD Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`, 20 WGPs / 40 CUs, 128 GB unified RAM, NixOS, Python 3.12.12, PyTorch `2.13.0+rocm10.0.0`, Triton `3.8.0`, vLLM `0.34.0+rocm100`), `--batch-tokens 4096`, DSpark `K=7`.

### 1. Out-of-the-box unmodified Hugging Face checkpoint (`thinkingmachines/Inkling-2-Mini`, `EXPERT_POLICY=source`)

No private fitted banks; 20 NVFP4 layers use Wave 5.2 pre-tiled FP16 WMMA and 4 coda layers remain native MXFP8.

| Workload (`c1` HIP graph, DSpark `K=7`, `--tools`) | Prompt tokens | TTFT | PP (tok/s) | TG (tok/s) | Functional result |
|---|---:|---:|---:|---:|---|
| Structured JSON (`order`) | 53 | 0.189 s | 280.4 | **84.12** | Valid exact JSON |
| Tool calling (`get_weather`) | — | — | — | **79.73** | Exact `get_weather` call |
| Long prompt (`p3500`) | 3,900 | 2.609 s | **1,494.6** | **72.75** | Coherent summary |
| Python coding (`bisect_left`) | 42 | 0.212 s | 197.9 | **69.79** | Exact `bisect_left` |

### 2. Ciru C1-fitted profile (Wave 5.2 vs. baseline ROCm port)

Includes Ciru's optional C1-down bank (layers 6, 12, 17) and C1 coda layers (20–23) on the same Wave 5.2 runtime.

| Benchmark | Baseline ROCm port | Wave 5.2 Turbo | Speedup |
|---|---:|---:|---:|
| PP — 512 tokens | 852.0 tok/s | **1,279.6 tok/s** | **+50.2%** |
| PP — 2,048 tokens | 1,024.7 tok/s | **1,645.0 tok/s** | **+60.5%** |
| PP — 3,840 tokens | 1,005.5 tok/s | **1,691.3 tok/s** | **+68.2%** |
| PP — 8,192 tokens | 889.9 tok/s | **1,640.8 tok/s** | **+84.4%** |
| PP — 32,768 tokens | 780.1 tok/s | **1,248.3 tok/s** | **+60.0%** |
| PP — 65,536 tokens *(known drop-off)* | 646.8 tok/s | **788.5 tok/s** | **+21.9%** |
| TG — C1 Held Pooled (DSpark `K=7`) | 64.15 tok/s | **74.13 tok/s** | **+15.6%** |
| TG — 32K Context (DSpark `K=7`) | 49.41 tok/s | **63.69 tok/s** | **+28.9%** |
| TG — 64K Context (DSpark `K=7`) | 36.38 tok/s | **54.08 tok/s** | **+48.7%** |
| TG — C8 Eager Aggregate (8 concurrent) | 88.58 tok/s | **88.92 tok/s** | +0.4% |

See [BENCHMARKS.md](BENCHMARKS.md) for per-prompt breakdowns, 2,044-token KL/NLL audit numbers, GPU profiler roofline receipts, and multi-agent kernel roadmap.

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

# 3. Serve (single-stream HIP graph + DSpark K=7 + tool calling)
bash "$HOME/.local/opt/inkling-2-mini-preview/serve.sh" \
  --model "$HOME/models/Inkling-2-Mini" \
  --mode c1 --batch-tokens 4096 --tools

# 4. Run smoke test in another terminal
python3 scripts/smoke-test.py --base-url http://127.0.0.1:18081/v1 --model inkling-2-mini --check-tools
```

### Example API call

```bash
curl http://127.0.0.1:18081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "inkling-2-mini",
    "messages": [{"role": "user", "content": "Write bisect_left in Python without importing bisect."}],
    "temperature": 0,
    "max_tokens": 512,
    "stream": true
  }'
```

---

## What is included

| Path | Purpose |
|---|---|
| `overlay/vllm/` | Complete 45-file Ciru AMD ROCm backend and Wave 5.2 Triton kernel overlay (`vllm/models/inkling2/amd/`, scheduler/runner/dflash hooks) |
| `patches/vllm-inkling2-gfx1151.patch` | Clean unified git patch against vLLM commit `b426b46e` |
| `scripts/install.sh` | Portable verifier and installer (standalone `--prefix`, in-place `--venv` with backup, or `--vllm-source` patch mode) |
| `scripts/serve.sh` | Portable launcher supporting `c1`, `long`, `c8`, `prob`, stock HF `source` mode, and optional fitted banks |
| `scripts/smoke-test.py` | End-to-end JSON, math, 2K+ prefill, reasoning, and tool-calling verification script |
| `MANIFEST.sha256` | SHA-256 integrity manifest for all overlay files |
| `qualification/wave52-kl-summary.json` | 2,044-token full-vocabulary KL/NLL qualification summary |
| `INSTALL.md` / `BENCHMARKS.md` | Platform setup guide and full benchmark/roofline report |

---

## Credits and license

- **Upstream model:** [Thinking Machines Lab `Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Apache-2.0, revision `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`).
- **Upstream engine & references:** [vLLM](https://github.com/vllm-project/vllm) (`b426b46e`, PR #69028 and ROCm Small donor PR #67685), [Hugging Face Transformers](https://github.com/huggingface/transformers) (PR #51050), [Triton](https://github.com/triton-lang/triton), [PyTorch](https://pytorch.org), and [AMD ROCm](https://github.com/ROCm).
- **License:** Ciru-authored code, kernels, scripts, and documentation are Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
