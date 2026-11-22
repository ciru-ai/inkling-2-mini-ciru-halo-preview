# Install Inkling 2 Mini — Ciru Strix Halo (gfx1151) Preview

This preview packages the Ciru Wave 6.3 AMD ROCm (`gfx1151`) backend, Triton/Gluon FP16 WMMA / W6 / W4 / Hybrid AOTriton+Triton attention kernels, multi-request FULL Target & Draft HIP Graphs, dual-depth adaptive DSpark speculation (`K=7 @ B=1`, `K=4 @ B=2..8`), portable installer, and serving launcher for [Thinking Machines Lab `Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (`6e7c8248f75265bdd4fbab5273304243b5d8c2cb`).

> **Preview scope:** This repository is a GitHub code/overlay preview. It runs directly against the unmodified upstream Hugging Face checkpoint (`thinkingmachines/Inkling-2-Mini`, Apache-2.0) with **native ROCm vision, reasoning, and tool calling** enabled by default (pass `--language-model-only` to disable vision). Optional C1-fitted expert banks are not hosted here yet; a full Hugging Face release bundle will follow separately.

---

## Hardware and platform requirements

- **Target GPU:** AMD Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`, RDNA 3.5, wave32), **128 GB unified memory** recommended for 32K–64K context or 8-way concurrency.
- **Memory budget:** ~22–28 GB RAM for model weights + pre-tiled FP16/W6/W4 buffers + 6 GiB KV cache (4K–16K single-stream). Budget ~45–60 GB for 32K–64K or 8 concurrent agent slots with 24 GiB KV cache.
- **Disk space:** ~25 GB for `thinkingmachines/Inkling-2-Mini` weights (`model-00001-of-00001.safetensors` ~19.5 GB + `mtp.safetensors` ~0.64 GB) plus ~8–12 GB for the ROCm 10 / PyTorch / vLLM virtual environment and Triton cache.
- **Linux GPU access:** Working AMD `amdgpu` / `amdkfd` kernel driver with read/write access to `/dev/kfd` and `/dev/dri/renderD*` (typically via `render` and `video` groups).

### Validated platform vs. mainstream Linux guides

| Layer | Validated on Ciru (measured) | Mainstream Linux Distros |
|---|---|---|
| OS | NixOS (x86_64 Linux) | Ubuntu 24.04 / 26.04 LTS, Debian, Fedora, Arch (unvalidated end-to-end here; userspace prerequisites provided below) |
| GPU | AMD Radeon 8060S (`gfx1151`, 20 WGPs / 40 CUs) | AMD Strix Halo `gfx1151` |
| Python | CPython `3.12.12` | Python `3.12` |
| PyTorch / HIP | `2.13.0+rocm10.0.0` (HIP `7.15.26333`) | ROCm 10 / TheRock `gfx1151` wheels |
| Triton | `3.8.0+git669b31ac` | Triton `3.8+` with RDNA3.5 `wmma` support |
| Base vLLM | `0.34.0+rocm100` (baseline `b426b46e`) | vLLM `0.34.0` built on ROCm 10 (`b426b46e` / `ciru-ai/vllm` branch `inkling-2-mini-gfx1151-turbo-preview`) |

**Do not substitute stock CPU/CUDA `pip install vllm`.** The Inkling 2 Mini AMD backend and Wave 6.3 Triton/Gluon kernels require ROCm 10 (`gfx1151` WMMA + AOTriton FlashAttention-2 + HIP graphs) and the Ciru overlay in this repository.

---

## Step 1: OS prerequisites (mainstream Linux)

### Ubuntu 24.04 / 26.04 LTS and Debian

```bash
sudo apt-get update
sudo apt-get install -y \
  git curl ca-certificates build-essential cmake ninja-build pkg-config \
  python3.12 python3.12-venv libnuma1 libdrm2 libelf1 libgomp1 zlib1g
```

Ensure your user belongs to `render` and `video` (log out and back in if changed):

```bash
sudo usermod -aG render,video "$USER"
ls -la /dev/kfd /dev/dri/renderD*
```

### Fedora

```bash
sudo dnf install -y \
  git curl ca-certificates gcc gcc-c++ cmake ninja-build pkgconf-pkg-config \
  python3.12 numactl-libs libdrm elfutils-libelf libgomp zlib
sudo usermod -aG render,video "$USER"
```

### Arch Linux

```bash
sudo pacman -S --needed \
  git curl ca-certificates base-devel cmake ninja pkgconf \
  python numactl libdrm libelf zlib
sudo usermod -aG render,video "$USER"
```

---

## Step 2: Clone this preview repository

```bash
git clone https://github.com/ciru-ai/inkling-2-mini-ciru-halo-preview.git
cd inkling-2-mini-ciru-halo-preview
```

---

## Step 3: Verify your ROCm 10 / vLLM 0.34 environment

If you already have a Ciru/TheRock ROCm 10 Python 3.12 environment with `torch 2.13+rocm10`, `triton 3.8`, and `vllm 0.34.0+rocm100` (for example `/path/to/vllm-034-rocm100`), verify it first:

```bash
bash scripts/install.sh --venv /path/to/vllm-034-rocm100 --verify-only
```

If you are building vLLM from source on `gfx1151`, check out our full branch directly instead of applying the overlay separately:

```bash
git clone -b inkling-2-mini-gfx1151-turbo-preview https://github.com/ciru-ai/vllm.git vllm-inkling-preview
```

---

## Step 4: Download `thinkingmachines/Inkling-2-Mini` and install the overlay

### Option A (recommended): Standalone prefix (leaves your existing vLLM venv unmodified)

```bash
bash scripts/install.sh \
  --venv /path/to/vllm-034-rocm100 \
  --prefix "$HOME/.local/opt/inkling-2-mini-preview" \
  --download-model "$HOME/models/Inkling-2-Mini"
```

This verifies the SHA-256 manifest of every overlay file, shallow-clones `ciru-ai/vllm` branch `inkling-2-mini-gfx1151-turbo-preview` (which includes upstream Inkling 2 Mini PR #69028 `b426b46e` plus Ciru Wave 6.3), symlinks your ROCm 10 venv's compiled `_rocm_C.abi3.so` / `_C.abi3.so` native extensions into `<prefix>/vllm-src/vllm/`, downloads the pinned `thinkingmachines/Inkling-2-Mini` revision (`6e7c8248f75265bdd4fbab5273304243b5d8c2cb`), and creates `$HOME/.local/opt/inkling-2-mini-preview/serve.sh`.

### Option B: Apply unified git patch to an existing vLLM checkout at `b426b46e`

```bash
bash scripts/install.sh --vllm-source /path/to/vllm-checkout-at-b426b46e
```

---

## Step 5: Serve and verify

### Single-stream fast mode (HIP graph + DSpark K=7 + Turbo kernels, 4K context)

```bash
bash "$HOME/.local/opt/inkling-2-mini-preview/serve.sh" \
  --model "$HOME/models/Inkling-2-Mini" \
  --mode c1 --batch-tokens 4096 --tools
```

### Long-context single-stream mode (64K context, 32 GiB KV cache, Hybrid AOTriton+Triton prefill)

```bash
bash "$HOME/.local/opt/inkling-2-mini-preview/serve.sh" \
  --model "$HOME/models/Inkling-2-Mini" \
  --mode long --context 65536 --batch-tokens 4096 --tools
```

### Unified multi-agent adaptive mode (8 slots, FULL Target+Draft HIP graphs, `K=7 @ B=1` / `K=4 @ B=2..8`)

```bash
bash "$HOME/.local/opt/inkling-2-mini-preview/serve.sh" \
  --model "$HOME/models/Inkling-2-Mini" \
  --mode c8 --seqs 8 --context 32768 --batch-tokens 4096 --kv-mib 24576 \
  --prefix-cache --tools
```

### Multi-agent probabilistic mode (`T>0`, `K=4`, FULL Target+Draft HIP graphs, vision + tools)

```bash
bash "$HOME/.local/opt/inkling-2-mini-preview/serve.sh" \
  --model "$HOME/models/Inkling-2-Mini" \
  --mode prob --seqs 8 --context 32768 --batch-tokens 4096 --kv-mib 24576 \
  --prefix-cache --tools
```

### Smoke-test the running server

In a second terminal:

```bash
python3 scripts/smoke-test.py --base-url http://127.0.0.1:18081/v1 --model inkling-2-mini --check-tools
```

### Optional: C1-fitted expert banks (internal / future HF bundle)

If you have Ciru C1-fitted expert banks locally, pass them explicitly without modifying scripts:

```bash
bash "$HOME/.local/opt/inkling-2-mini-preview/serve.sh" \
  --model "$HOME/models/Inkling-2-Mini" \
  --mode c1 --batch-tokens 4096 --tools \
  --expert-bank /path/to/fit --expert-policy c1-down \
  --coda-c1-dir /path/to/coda-all-c1-v1
```

---

## Troubleshooting

- **Permission denied on `/dev/kfd`**: Confirm `id` shows `render` and `video`, and `ls -la /dev/kfd` is writable.
- **First launch takes 1–3 minutes**: Triton compiles and caches `gfx1151` kernels on first run into `${XDG_CACHE_HOME:-$HOME/.cache}/inkling-ciru-triton` (and HIP graph capture runs during startup). Subsequent launches reuse the cache.
- **64K long-context memory budget**: Wave 6.3 eliminates the 64K prefill drop-off (`1,507 → 1,274 tok/s` smoothly across 4K→62K via Hybrid AOTriton+Triton global prefill). For 65,536-token contexts, `serve.sh` auto-sizes `--kv-mib` to `32768` (32 GiB); pass `--kv-mib 24576` if sharing RAM with other workloads.
- **Vision / text-only mode**: Native ROCm vision (`model.visual.*`) is enabled by default in `scripts/serve.sh`. Pass `--language-model-only` to skip loading the vision tower if you only need text.
