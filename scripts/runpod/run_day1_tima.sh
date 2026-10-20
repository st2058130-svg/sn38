#!/usr/bin/env bash
# 32-vCPU datagen plan using Ti-Ma/wikipedia_2012 (not Wikimedia dumps).
#
# Parallel tmux jobs:
#   wiki   — full Ti-Ma stream + mainspace filter   (~hours, ~40GB download)
#   pes2o  — year <= 2012
#   fineweb— kept docs after date filter (smaller; leak risk)
# then: QA from wiki + mix
#
# Usage:
#   source /workspace/sn38-data/env.sh
#   bash scripts/runpod/run_day1_tima.sh start
#   bash scripts/runpod/run_day1_tima.sh status
#   bash scripts/runpod/run_day1_tima.sh mix
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${SN38_DATA_ROOT:-/workspace/sn38-data}"
PES2O_DOCS="${DAY1_PES2O_DOCS:-600000}"
FINEWEB_DOCS="${DAY1_FINEWEB_DOCS:-200000}"

export SN38_DATA_ROOT="$DATA_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/hf_cache}"
cd "$REPO_ROOT"
mkdir -p "$DATA_ROOT"/{raw,filtered,qa,mix,state,logs,hf_cache}

if ! command -v tmux >/dev/null 2>&1; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux
fi

start_job() {
  local name="$1"
  local cmd="$2"
  local log="$DATA_ROOT/logs/tima_${name}.log"
  if tmux has-session -t "sn38-tima-${name}" 2>/dev/null; then
    echo "already running: sn38-tima-${name}"
    return 0
  fi
  tmux new-session -d -s "sn38-tima-${name}" bash -lc "
    set -a; source $DATA_ROOT/env.sh 2>/dev/null || true; set +a
    export SN38_DATA_ROOT=$DATA_ROOT HF_HOME=$HF_HOME PYTHONPATH=$REPO_ROOT
    cd $REPO_ROOT
    echo '[start] $(date -Is) $name' | tee -a $log
    $cmd 2>&1 | tee -a $log
    echo '[done] $(date -Is) $name' | tee -a $log
    echo DONE > $DATA_ROOT/logs/tima_${name}.done
  "
  echo "started sn38-tima-${name}  log=$log"
}

cmd_start() {
  echo "=== 32-CPU Ti-Ma datagen ==="
  echo "  wiki   : Ti-Ma/wikipedia_2012 (full, mainspace filter)"
  echo "  peS2o  : max_docs=$PES2O_DOCS"
  echo "  fineweb: max kept=$FINEWEB_DOCS"
  echo "  data   : $DATA_ROOT"
  echo

  start_job wiki \
    "python -m train.data.ti_ma_wikipedia --finalize"

  start_job pes2o \
    "python -m train.data.pes2o --max-docs $PES2O_DOCS --finalize"

  start_job fineweb \
    "python -m train.data.fineweb_edu --max-docs $FINEWEB_DOCS --finalize"

  cat <<EOF

Monitor:
  tmux attach -t sn38-tima-wiki
  bash scripts/runpod/run_day1_tima.sh status
  tail -f $DATA_ROOT/logs/tima_*.log

When wiki+pes2o done (FineWeb can still run):
  bash scripts/runpod/run_day1_tima.sh mix

EOF
}

cmd_status() {
  for name in wiki pes2o fineweb; do
    local sess="sn38-tima-${name}"
    local donef="$DATA_ROOT/logs/tima_${name}.done"
    if [[ -f "$donef" ]]; then
      echo "[DONE]    $name"
    elif tmux has-session -t "$sess" 2>/dev/null; then
      echo "[RUNNING] $name  (tmux attach -t $sess)"
    else
      echo "[IDLE]    $name  (check logs/tima_${name}.log)"
    fi
  done
  ls -lh "$DATA_ROOT/filtered" 2>/dev/null || true
}

cmd_mix() {
  echo "==> QA from Ti-Ma wiki + train mix"
  python -m train.data.build_mix_all
  ls -lh "$DATA_ROOT/mix/train_mix.jsonl" 2>/dev/null || true
  echo "Next: H100 training on $DATA_ROOT/mix/train_mix.jsonl"
}

cmd_stop() {
  for name in wiki pes2o fineweb; do
    tmux kill-session -t "sn38-tima-${name}" 2>/dev/null && echo "killed sn38-tima-${name}" || true
  done
}

case "${1:-start}" in
  start|"") cmd_start ;;
  status) cmd_status ;;
  mix) cmd_mix ;;
  stop) cmd_stop ;;
  *) echo "usage: $0 start|status|mix|stop"; exit 1 ;;
esac
