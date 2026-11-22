#!/usr/bin/env bash
# Portable launcher for Inkling 2 Mini on AMD Strix Halo (gfx1151) with Ciru Wave 5.2 Turbo kernels.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

usage() {
    cat <<'EOF'
Usage: serve.sh --model PATH [OPTIONS]

Required:
  --model PATH            Local path to thinkingmachines/Inkling-2-Mini weights

Modes & Serving Options:
  --mode MODE             c1 (default, single-stream HIP graph, <=4096 ctx),
                          long (single-stream HIP graph, 32768 ctx default),
                          c8 (multi-agent eager greedy DSpark, 8 seqs default),
                          prob (eager probabilistic DSpark)
  --served-model-name ID  OpenAI API model name (default: inkling-2-mini)
  --host HOST             Bind host (default: 127.0.0.1)
  --port PORT             Bind port (default: 18081)
  --context N             Max prompt + completion tokens (default: 4096, or 32768 in long)
  --seqs N                Max concurrent sequences (default: 1; 8 in c8)
  --batch-tokens N        Scheduler max_num_batched_tokens (default: 4096)
  --prefill-chunk-tokens N
                          Optional --long-prefill-token-threshold cap
  --kv-mib N              BF16 KV cache budget in MiB (auto-sized in 6144 MiB / 16K steps)
  --spec-tokens K         DSpark draft depth 1..7 (default: 7)
  --no-spec               Disable DSpark speculative decoding
  --prefix-cache          Enable vLLM prefix caching
  --tools                 Enable Inkling native auto tool calling
  --vision                Enable native ROCm vision encoder (default: on)
  --language-model-only   Disable vision encoder to save ~0.6 GiB RAM
  --no-turbo              Disable Wave 5.2 FP16 WMMA / Turbo kernels (use baseline ROCm path)
  --draft-graph           Enable DSpark draft HIP graph (default: on in c1/long with --turbo)
  --no-draft-graph        Disable DSpark draft HIP graph

Runtime & Optional Quant Bank Paths:
  --python PATH           Python interpreter inside the vLLM 0.34 ROCm 10 environment
                          (auto-detected from INKLING_PYTHON, active VIRTUAL_ENV, or .venv)
  --overlay PATH          Path to Ciru vLLM overlay tree (default: <repo>/overlay or installed site-packages)
  --transformers-src PATH Optional path to Transformers source checkout with inkling2_loop
  --triton-cache PATH     Triton kernel cache directory (default: ${XDG_CACHE_HOME:-$HOME/.cache}/inkling-ciru-triton)
  --expert-bank PATH      Optional fitted expert bank directory (enables c1-down on layers 6, 12, 17)
  --expert-policy POLICY  Expert policy when --expert-bank is set: source|signed|c1-down|full-c1 (default: source)
  --coda-c1-dir PATH      Optional fitted C1 coda safetensors directory for layers 20-23
  --profiler-dir PATH     Optional Torch profiler trace output directory
  --dry-run               Print resolved environment and command without launching
EOF
}

die() {
    printf 'inkling-ciru: %s\n' "$*" >&2
    exit 2
}

mode=c1
model="${INKLING_MODEL_DIR:-}"
served_name=inkling-2-mini
host=127.0.0.1
port=18081
context=
seqs=
batch_tokens=4096
prefill_chunk=
kv_mib=
spec_tokens=7
speculation=1
prefix_cache=0
tools=0
vision=1
turbo=1
draft_graph=auto
python_bin="${INKLING_PYTHON:-}"
overlay_dir="${INKLING_OVERLAY_DIR:-}"
transformers_src="${INKLING_TRANSFORMERS_SRC:-}"
triton_cache="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/inkling-ciru-triton}"
expert_bank="${INKLING_AMD_EXPERT_BANK:-}"
expert_policy=source
coda_c1_dir="${INKLING_AMD_CODA_C1_DIR:-}"
profiler_dir=
dry_run=0

while (( $# )); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --model) model="$2"; shift ;;
        --mode) mode="$2"; shift ;;
        --served-model-name) served_name="$2"; shift ;;
        --host) host="$2"; shift ;;
        --port) port="$2"; shift ;;
        --context) context="$2"; shift ;;
        --seqs) seqs="$2"; shift ;;
        --batch-tokens) batch_tokens="$2"; shift ;;
        --prefill-chunk-tokens) prefill_chunk="$2"; shift ;;
        --kv-mib) kv_mib="$2"; shift ;;
        --spec-tokens) spec_tokens="$2"; shift ;;
        --no-spec) speculation=0 ;;
        --prefix-cache) prefix_cache=1 ;;
        --tools) tools=1 ;;
        --vision) vision=1 ;;
        --language-model-only) vision=0 ;;
        --no-turbo) turbo=0 ;;
        --draft-graph) draft_graph=1 ;;
        --no-draft-graph) draft_graph=0 ;;
        --python) python_bin="$2"; shift ;;
        --overlay) overlay_dir="$2"; shift ;;
        --transformers-src) transformers_src="$2"; shift ;;
        --triton-cache) triton_cache="$2"; shift ;;
        --expert-bank) expert_bank="$2"; shift ;;
        --expert-policy) expert_policy="$2"; shift ;;
        --coda-c1-dir) coda_c1_dir="$2"; shift ;;
        --profiler-dir) profiler_dir="$2"; shift ;;
        --dry-run) dry_run=1 ;;
        *) die "unknown option: $1" ;;
    esac
    shift
done

[[ -n "$model" ]] || die '--model PATH is required (download thinkingmachines/Inkling-2-Mini first)'
case "$mode" in
    c1|long|c8|prob) ;;
    *) die "unsupported mode: $mode (expected c1, long, c8, or prob)" ;;
esac

if [[ -z "$context" ]]; then
    if [[ "$mode" == long ]]; then context=32768; else context=4096; fi
fi
if [[ -z "$seqs" ]]; then
    if [[ "$mode" == c8 ]]; then seqs=8; else seqs=1; fi
fi
if [[ -n "$expert_bank" && "$expert_policy" == source ]]; then
    expert_policy=c1-down
fi

if [[ -z "$python_bin" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
        python_bin="$VIRTUAL_ENV/bin/python"
    elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        python_bin="$REPO_ROOT/.venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        python_bin=$(command -v python3)
    else
        die 'could not locate python3; pass --python PATH'
    fi
fi

if [[ -z "$overlay_dir" && -d "$REPO_ROOT/overlay/vllm" ]]; then
    overlay_dir="$REPO_ROOT/overlay"
fi

if [[ "$mode" == c1 ]] && (( context > 4096 )); then
    die 'Use --mode long for single-request HIP graph contexts above 4096 tokens'
fi
kv_mib=${kv_mib:-$((6144 * ((context + 16383) / 16384)))}

target_graph=0
plain_graph=0
copy_draft=0
verifier_head_tile=0
sample_method=greedy
experimental_kernels=0
draft_w8=0
router_mode=off

case "$mode" in
    c1|long)
        target_graph="$speculation"
        plain_graph=$((1 - speculation))
        verifier_head_tile=$((8 * target_graph))
        if (( turbo && speculation )); then
            experimental_kernels=1
            draft_w8=all
            router_mode=align
        fi
        ;;
    prob)
        sample_method=probabilistic
        ;;
esac
if [[ "$mode" == c1 ]]; then
    copy_draft="$target_graph"
fi
if [[ "$draft_graph" == auto ]]; then
    if (( turbo && speculation )) && [[ "$mode" == c1 || "$mode" == long ]]; then
        draft_graph=1
    else
        draft_graph=0
    fi
fi

turbo_prefill_moe=0
if (( turbo )); then
    turbo_prefill_moe=fp16
fi

# Build clean PYTHONPATH (overlay first so patched vllm modules take precedence).
py_parts=()
if [[ -n "$overlay_dir" ]]; then
    py_parts+=("$overlay_dir")
fi
if [[ -n "$transformers_src" ]]; then
    py_parts+=("$transformers_src")
fi
if [[ -n "${PYTHONPATH:-}" ]]; then
    py_parts+=("$PYTHONPATH")
fi
joined_pypath=$(IFS=:; echo "${py_parts[*]}")

settings=(
    VLLM_NO_USAGE_STATS=1
    VLLM_ENABLE_V1_MULTIPROCESSING=1
    VLLM_ROCM_USE_AITER=0
    "TRITON_CACHE_DIR=$triton_cache"
    OMP_NUM_THREADS=4
    OPENBLAS_NUM_THREADS=4
    "INKLING_AMD_EXPERT_POLICY=$expert_policy"
    INKLING_AMD_PACKED_PREFILL=1
    INKLING_AMD_PACKED_MXFP8_PREFILL=1
    INKLING_AMD_PACKED_MXFP8=1
    INKLING_AMD_PACKED_LOOP=1
    INKLING_AMD_PACKED_ATTN_RES=1
    INKLING_AMD_QKVR_W8=0
    INKLING_AMD_PREFILL_WIDE_TILE=1
    INKLING_AMD_PREFILL_SHORT_N128=1
    INKLING_AMD_PREFILL_WIDE_SMALL_BATCH=1
    INKLING_AMD_PREFILL_FUSED_ACTIVATION=1
    INKLING_AMD_PREFILL_FUSED_REDUCE=1
    INKLING_AMD_PACKED_SMALL_M=0
    INKLING_AMD_GROUPWISE_W8_ATTN=1
    INKLING_AMD_W8_FUSED_QKV=1
    INKLING_AMD_SOURCE_M1_BF16_DOT2=1
    INKLING_AMD_SOURCE_M1_WAVE32=1
    INKLING_AMD_PACKED_M1_ROUTER=1
    INKLING_AMD_PACKED_M1_ROUTER_SCORES=1
    INKLING_AMD_PACKED_M1_ROUTER_SELECTION=1
    INKLING_AMD_M1_KEY_TILE=16
    INKLING_AMD_M1_ATTENTION_WAVES=2
    INKLING_AMD_M1_SPLIT_PROGRAMS_PER_CU=4
    INKLING_AMD_M1_COMPACT_SPLIT_MERGE=1
    INKLING_AMD_GROUPWISE_W6_ATTN=1
    INKLING_AMD_GROUPWISE_W8_HEAD=0
    INKLING_AMD_GROUPWISE_W6_HEAD=1
    INKLING_AMD_W6_HEAD_GROUP_SIZE=32
    INKLING_AMD_W6_HEAD_FUSED_SOFTCAP=1
    INKLING_AMD_W6_ATTN_WAVES=4
    INKLING_AMD_W6_HEAD_WAVES=4
    INKLING_AMD_LOOP_GEMV=both
    INKLING_AMD_DSPARK_PHASE_SWITCH=0
    INKLING_AMD_SOURCE_WMMA_N_FAST=0
    "INKLING_AMD_DECODE_GRAPH=$plain_graph"
    "INKLING_AMD_DSPARK_TARGET_GRAPH=$target_graph"
    "INKLING_AMD_SOURCE_WMMA_PRETILE=$target_graph"
    "INKLING_AMD_CODA_WMMA_PRETILE=$target_graph"
    "INKLING_AMD_MIXED_WMMA_PRETILE=$target_graph"
    "INKLING_AMD_SOURCE_WMMA_WAVE_EPILOGUE=$target_graph"
    "INKLING_AMD_DSPARK_COPY_DRAFT=$copy_draft"
    "INKLING_AMD_W6_VERIFY_ATTN=$target_graph"
    "INKLING_AMD_W6_HEAD_BATCH_TILE=$verifier_head_tile"
    "INKLING_AMD_DSPARK_W6_BASE_HEAD=$target_graph"
    "INKLING_AMD_DSPARK_W8_MARKOV=$target_graph"
    "INKLING_AMD_DSPARK_MARKOV_FUSED_ARGMAX=$target_graph"
    "INKLING_AMD_DSPARK_W8_BACKBONE=$draft_w8"
    "INKLING_AMD_FUSED_M8_ROUTER=$router_mode"
    "INKLING_AMD_W6_BATCH_DOT2=$experimental_kernels"
    "INKLING_AMD_LOOP_M8_FUSED=$experimental_kernels"
    "INKLING_AMD_TURBO_PREFILL_MOE=$turbo_prefill_moe"
    "INKLING_AMD_TURBO_PREFILL_ATTN=$turbo"
    "INKLING_AMD_TURBO_MOE_F16=$turbo"
    "INKLING_AMD_TURBO_ROUTER_M8=$turbo"
    "INKLING_AMD_TURBO_W6_WMMA=$turbo"
    "INKLING_AMD_TURBO_DRAFT_W4=$turbo"
    INKLING_AMD_TURBO_MARKOV_TOPK=0
    "INKLING_AMD_DSPARK_DRAFT_GRAPH=$draft_graph"
)
if [[ -n "$joined_pypath" ]]; then
    settings=("PYTHONPATH=$joined_pypath" "${settings[@]}")
fi
if [[ -n "$expert_bank" ]]; then
    settings+=("INKLING_AMD_EXPERT_BANK=$expert_bank")
fi
if [[ -n "$coda_c1_dir" ]]; then
    settings+=("INKLING_AMD_CODA_C1_DIR=$coda_c1_dir" "INKLING_AMD_CODA_C1_KEEP_FINAL_MXFP8=0")
fi

command=(
    "$python_bin" -m vllm.entrypoints.cli.main serve "$model"
    --served-model-name "$served_name"
    --host "$host" --port "$port"
    --dtype bfloat16 --reasoning-parser inkling
    --max-model-len "$context" --max-num-seqs "$seqs"
    --max-num-batched-tokens "$batch_tokens"
    --kv-cache-memory-bytes "$((kv_mib * 1048576))"
)
if (( ! vision )); then
    command+=(--language-model-only)
fi
if [[ -n "$prefill_chunk" ]]; then
    command+=(--long-prefill-token-threshold "$prefill_chunk")
fi
if (( prefix_cache )); then
    command+=(--enable-prefix-caching)
else
    command+=(--disable-prefix-caching)
fi
if (( target_graph )); then
    command+=(
        --compilation-config
        "{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"decode_cudagraph_capture_request_sizes\":[1],\"max_decode_cudagraph_capture_tokens\":$((spec_tokens + 1))}"
        --cudagraph-metrics
    )
elif (( plain_graph )); then
    command+=(
        --compilation-config
        '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}'
        --cudagraph-metrics
    )
else
    command+=(--enforce-eager)
fi
if (( speculation )); then
    command+=(
        --speculative-config
        "{\"method\":\"dspark\",\"num_speculative_tokens\":$spec_tokens,\"draft_sample_method\":\"$sample_method\"}"
    )
fi
if (( tools )); then
    command+=(--enable-auto-tool-choice --tool-call-parser inkling)
fi
if [[ -n "$profiler_dir" ]]; then
    command+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$profiler_dir\"}")
fi

if (( dry_run )); then
    printf 'Inkling 2 Mini Ciru Preview (mode=%s turbo=%s policy=%s)\n' "$mode" "$turbo" "$expert_policy"
    printf 'Environment:\n'
    printf '  %q\n' "${settings[@]}"
    printf 'Command:\n '
    printf ' %q' "${command[@]}"
    printf '\n'
    exit 0
fi

[[ -d "$model" ]] || die "model directory not found: $model"
[[ -x "$python_bin" ]] || die "python executable not found: $python_bin"
[[ -r /dev/kfd && -w /dev/kfd ]] || die '/dev/kfd is not readable/writable; check render/video/kfd group permissions'
mkdir -p "$triton_cache"

for s in "${settings[@]}"; do
    export "$s"
done
printf 'Starting Inkling 2 Mini Ciru Preview: mode=%s turbo=%s policy=%s context=%s seqs=%s kv_mib=%s\n' \
    "$mode" "$turbo" "$expert_policy" "$context" "$seqs" "$kv_mib"
exec "${command[@]}"
