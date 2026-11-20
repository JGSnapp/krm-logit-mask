#!/usr/bin/env bash
# Prepares a fresh immers.cloud "Ubuntu 24.04 CUDA" VM for krm-logit-mask:
# user `trainer` (what gpu-harbor connects as), uv, the project with its GPU
# environment, and the models in the Hugging Face cache, so a VM started from a
# snapshot of this one runs experiments without downloading anything.
#
# Usage (as the image's default user, from the directory with the archive):
#   sudo bash bootstrap.sh /tmp/krm-logit-mask.tar.gz
#   MODELS="Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct" sudo -E bash bootstrap.sh ...
set -euo pipefail

ARCHIVE="${1:?path to krm-logit-mask.tar.gz}"
MODELS="${MODELS:-Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct}"
USER_NAME=trainer
HOME_DIR=/home/$USER_NAME

[ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
command -v timeout >/dev/null  # gpu-harbor wraps every job in GNU timeout.

apt-get update -q
DEBIAN_FRONTEND=noninteractive apt-get install -yq curl ca-certificates tar python3-pip

# Unprivileged user, no sudo; logs in with the same key as the default user.
id "$USER_NAME" >/dev/null 2>&1 || useradd -m -s /bin/bash "$USER_NAME"
install -d -m 700 -o "$USER_NAME" -g "$USER_NAME" "$HOME_DIR/.ssh"
# Only plain key lines: root's file on cloud images starts with a "please log in as" command.
grep -hE '^(ssh-|ecdsa-)' /home/*/.ssh/authorized_keys /root/.ssh/authorized_keys 2>/dev/null \
  | sort -u > "$HOME_DIR/.ssh/authorized_keys"
[ -s "$HOME_DIR/.ssh/authorized_keys" ] || { echo "no SSH key found to copy" >&2; exit 1; }
chown "$USER_NAME:$USER_NAME" "$HOME_DIR/.ssh/authorized_keys"
chmod 600 "$HOME_DIR/.ssh/authorized_keys"

install -m 644 "$ARCHIVE" "$HOME_DIR/krm-logit-mask.tar.gz"

sudo -u "$USER_NAME" -H env MODELS="$MODELS" HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}" bash -s <<'EOF'
set -euo pipefail
cd "$HOME"
# uv from PyPI: the astral.sh installer redirect was reset from this network once.
python3 -m pip install --user --break-system-packages -q uv
export PATH="$HOME/.local/bin:$PATH"
uv --version
grep -q '.local/bin' ~/.bashrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

mkdir -p krm-logit-mask
tar -xzf krm-logit-mask.tar.gz -C krm-logit-mask
rm krm-logit-mask.tar.gz
cd krm-logit-mask
# Not `uv sync`: PyPI throttles big files from immers (0.13 MB/s for CUDA wheels,
# 19.11.2026), the PyTorch CDN gives ~85 MB/s. torch + CUDA libs from there (cu126
# covers the T4), the small rest from PyPI. Run with .venv/bin/python, not `uv run`.
[ -x .venv/bin/python ] || uv venv --python 3.12
uv pip install --index-url https://download.pytorch.org/whl/cu126 torch
uv pip install -e ".[gpu,web,lemma,dev]"

# Models go to ~/.cache/huggingface and stay in the snapshot.
.venv/bin/python - <<PY
from huggingface_hub import snapshot_download
for name in "$MODELS".split():
    print("download", name, flush=True)
    snapshot_download(name)
PY

.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print('CUDA ok:', torch.cuda.get_device_name(0))"
.venv/bin/python -m pytest -q
echo 'export HF_HUB_OFFLINE=1 UV_NO_SYNC=1' >> ~/.bashrc  # Models are local; no network calls at load.
EOF

echo "Ready. Check: sudo -iu $USER_NAME, then cd krm-logit-mask && .venv/bin/python experiments/run_temperature_eval.py --help"
echo "Before the snapshot: sudo bash prepare_snapshot.sh"
