#!/usr/bin/env bash
# Launch long-running wiki history datagen inside tmux (survives SSH disconnect).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${SN38_DATA_ROOT:-/workspace/sn38-data}"
SESSION="${TMUX_SESSION:-sn38-wiki}"

export SN38_DATA_ROOT="$DATA_ROOT"
mkdir -p "$DATA_ROOT"

if ! command -v tmux >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq tmux
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session $SESSION already exists. Attach: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" bash -lc "
  set -a
  source $DATA_ROOT/env.sh 2>/dev/null || true
  set +a
  export SN38_DATA_ROOT=$DATA_ROOT
  export PYTHONPATH=$REPO_ROOT
  cd $REPO_ROOT
  bash scripts/runpod/run_datagen.sh wiki 2>&1 | tee $DATA_ROOT/wiki_datagen.log
"

echo "Started tmux session: $SESSION"
echo "  attach:  tmux attach -t $SESSION"
echo "  log:     tail -f $DATA_ROOT/wiki_datagen.log"
