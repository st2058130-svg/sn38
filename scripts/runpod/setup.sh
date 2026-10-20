#!/usr/bin/env bash
# Bootstrap SN38 datagen on a RunPod CPU (or GPU) pod.
# Usage:
#   cd /workspace && git clone <your-fork-or-this-repo> sn38
#   bash sn38/scripts/runpod/setup.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${SN38_DATA_ROOT:-/workspace/sn38-data}"

echo "==> Repo: $REPO_ROOT"
echo "==> Data: $DATA_ROOT"

mkdir -p "$DATA_ROOT"/{raw,filtered,filtered_edu,snapshots,qa,mix,state,hf_cache,checkpoints,export}
mkdir -p "$DATA_ROOT/raw"/{wikipedia_2012,pes2o,fineweb_edu}

export SN38_DATA_ROOT="$DATA_ROOT"
export HF_HOME="${HF_HOME:-$DATA_ROOT/hf_cache}"
export WIKI_DUMP_ID="${WIKI_DUMP_ID:-latest}"

cd "$REPO_ROOT"

# 7z needed to stream Wikipedia history shards (much smaller than bz2)
if ! command -v 7z >/dev/null 2>&1; then
  echo "==> Installing p7zip-full"
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq p7zip-full
fi

if command -v uv >/dev/null 2>&1; then
  echo "==> uv sync"
  uv sync
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "==> pip install (uv not found)"
  python3 -m pip install -U pip
  python3 -m pip install -e .
  python3 -m pip install datasets huggingface_hub
fi

# Datagen extras (streaming HF datasets)
python3 -m pip install -q datasets

WIKI_DUMP_ID="${WIKI_DUMP_ID:-20261001}"
WIKI_DUMP_FORMAT="${WIKI_DUMP_FORMAT:-7z}"

echo "==> Writing env file to $DATA_ROOT/env.sh"
cat > "$DATA_ROOT/env.sh" <<EOF
export SN38_DATA_ROOT=$DATA_ROOT
export SN38_REPO=$REPO_ROOT
export HF_HOME=$HF_HOME
export WIKI_DUMP_ID=$WIKI_DUMP_ID
export WIKI_DUMP_FORMAT=$WIKI_DUMP_FORMAT
export PYTHONPATH=$REPO_ROOT\${PYTHONPATH:+:\$PYTHONPATH}
cd $REPO_ROOT
EOF

echo "==> Done. Next:"
echo "    source $DATA_ROOT/env.sh"
echo "    bash $REPO_ROOT/scripts/runpod/run_datagen.sh --help"
