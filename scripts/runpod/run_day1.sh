#!/usr/bin/env bash
# Finish a usable 2012 training mix in ~1 day on RunPod (CPU pod).
#
# Reality check:
#   Full wiki history (980 shards / ~241GB 7z) does NOT finish in one day.
#   This profile runs THREE jobs in parallel for ~20–24h, then builds the mix:
#     - wiki: first N history shards (default 40)
#     - peS2o: up to P docs with year<=2012 (default 400000)
#     - fineweb: up to F KEPT docs after date filter (default 150000)
#
# Usage:
#   source /workspace/sn38-data/env.sh
#   bash scripts/runpod/run_day1.sh          # start parallel jobs in tmux
#   bash scripts/runpod/run_day1.sh status   # check
#   bash scripts/runpod/run_day1.sh mix      # when jobs done → QA + mix
#   bash scripts/runpod/run_day1.sh stop     # kill day1 tmux sessions
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${SN38_DATA_ROOT:-/workspace/sn38-data}"
DUMP_ID="${WIKI_DUMP_ID:-20261001}"
DUMP_FMT="${WIKI_DUMP_FORMAT:-7z}"

# Tunables (override via env)
WIKI_SHARDS="${DAY1_WIKI_SHARDS:-40}"
PES2O_DOCS="${DAY1_PES2O_DOCS:-400000}"
FINEWEB_DOCS="${DAY1_FINEWEB_DOCS:-150000}"

export SN38_DATA_ROOT="$DATA_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
mkdir -p "$DATA_ROOT"/{raw,filtered,qa,mix,state,logs}

if ! command -v tmux >/dev/null 2>&1; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux
fi

start_job() {
  local name="$1"
  local cmd="$2"
  local log="$DATA_ROOT/logs/day1_${name}.log"
  if tmux has-session -t "sn38-day1-${name}" 2>/dev/null; then
    echo "already running: sn38-day1-${name} (tmux attach -t sn38-day1-${name})"
    return 0
  fi
  tmux new-session -d -s "sn38-day1-${name}" bash -lc "
    set -a; source $DATA_ROOT/env.sh 2>/dev/null || true; set +a
    export SN38_DATA_ROOT=$DATA_ROOT PYTHONPATH=$REPO_ROOT
    cd $REPO_ROOT
    echo '[start] $(date -Is) $name' | tee -a $log
    $cmd 2>&1 | tee -a $log
    echo '[done] $(date -Is) $name' | tee -a $log
    echo DONE > $DATA_ROOT/logs/day1_${name}.done
  "
  echo "started sn38-day1-${name}  log=$log"
}

cmd_start() {
  echo "=== DAY1 datagen (parallel) ==="
  echo "  wiki shards : $WIKI_SHARDS   (of ~980)  dump=$DUMP_ID fmt=$DUMP_FMT"
  echo "  peS2o docs  : $PES2O_DOCS"
  echo "  fineweb kept: $FINEWEB_DOCS"
  echo "  data root   : $DATA_ROOT"
  echo
  echo "This will NOT download all of Wikipedia. It targets a 1-day usable mix."
  echo

  start_job wiki \
    "python -m train.data.wiki_history_pipeline --dump-id $DUMP_ID --format $DUMP_FMT --max-shards $WIKI_SHARDS --finalize"

  start_job pes2o \
    "python -m train.data.pes2o --max-docs $PES2O_DOCS --finalize"

  start_job fineweb \
    "python -m train.data.fineweb_edu --max-docs $FINEWEB_DOCS --finalize"

  # Bonus high-confidence wiki via API (small, ~1–2h) — optional 4th lane
  start_job wikiapi \
    "python -m train.data.wikipedia_2012 --api-sample --limit 2000 --also-filter --out $DATA_ROOT/raw/wikipedia_2012/wiki_api_day1.jsonl"

  cat <<EOF

Jobs launched. Monitor:
  tmux attach -t sn38-day1-wiki
  tmux attach -t sn38-day1-pes2o
  tmux attach -t sn38-day1-fineweb
  tail -f $DATA_ROOT/logs/day1_*.log

When all show DONE files under $DATA_ROOT/logs/:
  bash scripts/runpod/run_day1.sh mix

EOF
}

cmd_status() {
  for name in wiki pes2o fineweb wikiapi; do
    local sess="sn38-day1-${name}"
    local donef="$DATA_ROOT/logs/day1_${name}.done"
    if [[ -f "$donef" ]]; then
      echo "[DONE]    $name  ($(cat "$donef" 2>/dev/null || true))"
    elif tmux has-session -t "$sess" 2>/dev/null; then
      echo "[RUNNING] $name  (tmux attach -t $sess)"
    else
      echo "[IDLE]    $name  (not started or crashed — check logs/day1_${name}.log)"
    fi
  done
  echo
  echo "Filtered outputs:"
  ls -lh "$DATA_ROOT/filtered" 2>/dev/null || true
}

cmd_mix() {
  echo "==> Merging wiki API into filtered if present"
  if [[ -f "$DATA_ROOT/raw/wikipedia_2012/wiki_api_day1.jsonl" ]]; then
    # Append API pages into filtered wiki if history filter exists; else copy
    local wiki_f="$DATA_ROOT/filtered/wikipedia_2012.jsonl"
    if [[ -f "$wiki_f" ]]; then
      python -m train.filter.pipeline \
        --in "$DATA_ROOT/raw/wikipedia_2012/wiki_api_day1.jsonl" \
        --out "$DATA_ROOT/filtered/wikipedia_api_day1.jsonl" \
        --source wikipedia_2012 || true
    else
      python -m train.filter.pipeline \
        --in "$DATA_ROOT/raw/wikipedia_2012/wiki_api_day1.jsonl" \
        --out "$wiki_f" \
        --source wikipedia_2012 || true
    fi
  fi
  echo "==> Build QA + train_mix.jsonl"
  python -m train.data.build_mix_all
  echo "Mix ready: $DATA_ROOT/mix/train_mix.jsonl"
  ls -lh "$DATA_ROOT/mix/train_mix.jsonl" 2>/dev/null || true
}

cmd_stop() {
  for name in wiki pes2o fineweb wikiapi; do
    tmux kill-session -t "sn38-day1-${name}" 2>/dev/null && echo "killed sn38-day1-${name}" || true
  done
}

case "${1:-start}" in
  start|"") cmd_start ;;
  status) cmd_status ;;
  mix) cmd_mix ;;
  stop) cmd_stop ;;
  -h|--help|help)
    sed -n '2,20p' "$0" | tr -d '#'
    ;;
  *)
    echo "Unknown: $1 (start|status|mix|stop)"
    exit 1
    ;;
esac
