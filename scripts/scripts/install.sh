#!/usr/bin/env bash
# Portable installer for the Inkling 2 Mini Ciru Strix Halo (gfx1151) Preview overlay.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

usage() {
    cat <<'EOF'
Usage: install.sh [OPTIONS]

Installs the Ciru Wave 5.2 gfx1151 overlay onto an existing vLLM 0.34+ ROCm 10
Python environment without mutating system packages, or into a standalone
directory prefix for PYTHONPATH-based serving.

Options:
  --venv PATH             Target vLLM 0.34 ROCm 10 virtual environment (applies overlay into site-packages/vllm
                          with automatic backup under <venv>/.inkling-ciru-backup/)
  --prefix PATH           Install as a standalone overlay prefix (leaves existing vLLM untouched;
                          generates <prefix>/serve.sh)
  --vllm-source PATH      Optional existing vLLM git checkout at b426b46 to apply patches/vllm-inkling2-gfx1151.patch
  --download-model PATH   Download thinkingmachines/Inkling-2-Mini (pinned rev 6e7c8248) into PATH via huggingface_hub
  --verify-only           Check GPU access, Python/Torch/Triton/vLLM versions, and overlay integrity without writing
  -h, --help              Show this help message
EOF
}

die() {
    printf 'install.sh: %s\n' "$*" >&2
    exit 2
}

venv_dir=
prefix_dir=
vllm_src=
download_model=
verify_only=0

while (( $# )); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --venv) venv_dir="$2"; shift ;;
        --prefix) prefix_dir="$2"; shift ;;
        --vllm-source) vllm_src="$2"; shift ;;
        --download-model) download_model="$2"; shift ;;
        --verify-only) verify_only=1 ;;
        *) die "unknown option: $1" ;;
    esac
    shift
done

[[ -d "$REPO_ROOT/overlay/vllm/models/inkling2/amd" ]] || die 'overlay/vllm/models/inkling2/amd missing from repository'

# 1. Host prerequisite checks
printf 'Checking host prerequisites...\n'
for cmd in python3 git; do
    command -v "$cmd" >/dev/null 2>&1 || die "missing required command: $cmd"
done
if [[ -e /dev/kfd ]]; then
    if [[ -r /dev/kfd && -w /dev/kfd ]]; then
        printf '  GPU device: /dev/kfd readable and writable\n'
    else
        printf '  WARNING: /dev/kfd exists but is not readable/writable by %s (check render/video groups)\n' "$(id -un)" >&2
    fi
else
    printf '  WARNING: /dev/kfd not detected (expected on non-AMD or container build hosts)\n' >&2
fi

python_bin=python3
if [[ -n "$venv_dir" ]]; then
    [[ -x "$venv_dir/bin/python" ]] || die "not a valid venv: $venv_dir"
    python_bin="$venv_dir/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    python_bin="$VIRTUAL_ENV/bin/python"
fi

"$python_bin" - <<'PY'
import importlib.metadata as m, sys
print(f"  Python: {sys.version.split()[0]} ({sys.executable})")
for pkg in ("torch", "triton", "vllm", "transformers", "safetensors", "huggingface_hub"):
    try:
        print(f"  {pkg}: {m.version(pkg)}")
    except m.PackageNotFoundError:
        print(f"  {pkg}: NOT INSTALLED")
try:
    import torch
    print(f"  torch.version.hip: {torch.version.hip}")
    if torch.cuda.is_available():
        print(f"  GPU[0]: {torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"  torch probe skipped: {exc}")
PY

# 2. Verify overlay manifest if present
if [[ -f "$REPO_ROOT/MANIFEST.sha256" ]]; then
    (cd "$REPO_ROOT" && sha256sum -c MANIFEST.sha256 >/dev/null)
    printf '  Overlay SHA-256 manifest: OK\n'
fi

if (( verify_only )); then
    printf 'Verification complete (--verify-only).\n'
    exit 0
fi

# 3. Optional model download (pinned upstream thinkingmachines/Inkling-2-Mini)
if [[ -n "$download_model" ]]; then
    printf 'Downloading thinkingmachines/Inkling-2-Mini (rev 6e7c8248f75265bdd4fbab5273304243b5d8c2cb) to %s...\n' "$download_model"
    "$python_bin" - "$download_model" <<'PY'
import sys
from huggingface_hub import snapshot_download
dest = sys.argv[1]
snapshot_download(
    repo_id="thinkingmachines/Inkling-2-Mini",
    revision="6e7c8248f75265bdd4fbab5273304243b5d8c2cb",
    local_dir=dest,
)
print("Download complete:", dest)
PY
fi

# 4. Install mode: standalone prefix, venv overlay, or vLLM source patch
if [[ -n "$prefix_dir" ]]; then
    [[ ! -e "$prefix_dir" ]] || die "refusing to overwrite existing prefix: $prefix_dir"
    mkdir -p "$prefix_dir"
    cp -a "$REPO_ROOT/overlay" "$prefix_dir/overlay"
    cp -a "$REPO_ROOT/scripts" "$prefix_dir/scripts"
    cp -a "$REPO_ROOT/LICENSE" "$REPO_ROOT/NOTICE" "$prefix_dir/"
    if [[ -n "$venv_dir" ]]; then
        # Recursively mirror unmodified base vllm files/extensions into every overlay directory
        # so PYTHONPATH overlay resolves nested subpackages (e.g. vllm.models.inkling2, vllm.v1.worker.gpu)
        site_vllm=$("$venv_dir/bin/python" -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).resolve().parent)')
        "$venv_dir/bin/python" - "$site_vllm" "$prefix_dir/overlay/vllm" <<'PY'
import os, pathlib, sys
src_root = pathlib.Path(sys.argv[1])
dst_root = pathlib.Path(sys.argv[2])
for dirpath, dirnames, _ in os.walk(dst_root):
    rel = pathlib.Path(dirpath).relative_to(dst_root)
    src_dir = src_root / rel
    if not src_dir.is_dir():
        continue
    for entry in src_dir.iterdir():
        target = pathlib.Path(dirpath) / entry.name
        if not target.exists() and not target.is_symlink():
            target.symlink_to(entry)
PY
    fi
    cat > "$prefix_dir/serve.sh" <<EOF
#!/usr/bin/env bash
exec bash "$prefix_dir/scripts/serve.sh" --overlay "$prefix_dir/overlay" ${venv_dir:+--python "$venv_dir/bin/python"} "\$@"
EOF
    chmod +x "$prefix_dir/serve.sh" "$prefix_dir/scripts/serve.sh" "$prefix_dir/scripts/smoke-test.py"
    printf '\nInstalled standalone overlay prefix to %s\n' "$prefix_dir"
    printf 'Next:\n  bash %q/serve.sh --model /path/to/Inkling-2-Mini --mode c1 --tools\n' "$prefix_dir"
elif [[ -n "$venv_dir" ]]; then
    site_vllm=$("$venv_dir/bin/python" -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).resolve().parent)')
    backup="$venv_dir/.inkling-ciru-backup-$(date +%Y%m%dT%H%M%S)"
    mkdir -p "$backup"
    printf 'Backing up overwritten files to %s...\n' "$backup"
    while IFS= read -r -d '' rel; do
        src="$REPO_ROOT/overlay/vllm/$rel"
        dst="$site_vllm/$rel"
        if [[ -e "$dst" ]]; then
            mkdir -p "$backup/$(dirname "$rel")"
            cp -a "$dst" "$backup/$rel"
        fi
        mkdir -p "$(dirname "$dst")"
        cp -a "$src" "$dst"
    done < <(cd "$REPO_ROOT/overlay/vllm" && find . -type f -print0 | sed -z 's|^\./||')
    printf '\nOverlay installed into %s\n' "$site_vllm"
    printf 'Next:\n  bash %q/scripts/serve.sh --python %q/bin/python --model /path/to/Inkling-2-Mini --mode c1 --tools\n' "$REPO_ROOT" "$venv_dir"
elif [[ -n "$vllm_src" ]]; then
    [[ -d "$vllm_src/.git" ]] || die "not a git checkout: $vllm_src"
    git -C "$vllm_src" apply --check "$REPO_ROOT/patches/vllm-inkling2-gfx1151.patch"
    git -C "$vllm_src" apply "$REPO_ROOT/patches/vllm-inkling2-gfx1151.patch"
    printf '\nApplied patches/vllm-inkling2-gfx1151.patch cleanly to %s\n' "$vllm_src"
else
    die 'specify one of --prefix PATH, --venv PATH, --vllm-source PATH, or --verify-only'
fi
