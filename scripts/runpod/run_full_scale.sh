#!/usr/bin/env bash
# ChronoGPT-scale datagen matching train_plan.txt token budgets (~27–38B unique).
#
# Targets (midpoints):
#   wiki Wikimedia enwiki meta-history  (~2–3B; per-shard download→filter→delete)
#   peS2o                               ~1.5B tokens
#   FineWeb-edu                         ~12.5B tokens  (slow; heavy discard)
#   Gutenberg                           ~4B tokens
#   QA                                  from wiki after
#   Usenet                              OPTIONAL / gated
#
# Wiki disk: never keep full ~241GB — one shard at a time, archive deleted after filter.
# Hardware: 16–32 vCPU, network volume **≥500GB–1TB**, expect **~2–4 weeks** wall time.
#
#   source /workspace/sn38-data/env.sh
#   bash scripts/runpod/run_full_scale.sh start
#   bash scripts/runpod/run_full_scale.sh status
#   bash scripts/runpod/run_full_scale.sh mix
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${SN38_DATA_ROOT:-/workspace/sn38-data}"
DUMP_ID="${WIKI_DUMP_ID:-20261001}"
DUMP_FMT="${WIKI_DUMP_FORMAT:-7z}"

export SN38_DATA_ROOT="$DATA_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/hf_cache}"
export WIKI_DUMP_ID="$DUMP_ID"
export WIKI_DUMP_FORMAT="$DUMP_FMT"
cd "$REPO_ROOT"
mkdir -p "$DATA_ROOT"/{raw,filtered,filtered_edu,qa,mix,state,logs,hf_cache}

if ! command -v tmux >/dev/null 2>&1; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux
fi

start_job() {
  local name="$1"
  local cmd="$2"
  local log="$DATA_ROOT/logs/full_${name}.log"
  if tmux has-session -t "sn38-full-${name}" 2>/dev/null; then
    echo "already running: sn38-full-${name}"
    return 0
  fi
  # Clear stale DONE from a previous failed/short run so status is honest
  rm -f "$DATA_ROOT/logs/full_${name}.done"
  tmux new-session -d -s "sn38-full-${name}" bash -lc "
    set -a; source $DATA_ROOT/env.sh 2>/dev/null || true; set +a
    export SN38_DATA_ROOT=$DATA_ROOT HF_HOME=$HF_HOME PYTHONPATH=$REPO_ROOT
    cd $REPO_ROOT
    rm -f $DATA_ROOT/logs/full_${name}.done
    echo '[start] \$(date -Is) $name' | tee -a $log
    set -o pipefail
    if $cmd 2>&1 | tee -a $log; then
      echo '[done-ok] \$(date -Is) $name' | tee -a $log
      echo DONE > $DATA_ROOT/logs/full_${name}.done
    else
      echo '[done-FAIL] \$(date -Is) $name exit=\$?' | tee -a $log
      echo FAIL > $DATA_ROOT/logs/full_${name}.fail
      exit 1
    fi
  "
  echo "started sn38-full-${name}  → $log"
}

restart_job() {
  local name="$1"
  tmux kill-session -t "sn38-full-${name}" 2>/dev/null || true
  rm -f "$DATA_ROOT/logs/full_${name}.done" "$DATA_ROOT/logs/full_${name}.fail"
  echo "cleared $name — call start again or use: bash scripts/runpod/run_full_scale.sh restart-$name"
}

cmd_start() {
  cat <<EOF
=== FULL SCALE (train_plan / ChronoGPT mix) ===
  wiki     Wikimedia enwiki meta-history ($DUMP_ID / $DUMP_FMT)
           download → extract → filter → DELETE archive+raw (per shard)
  peS2o    --full-scale (~1.5B tokens)
  fineweb  --full-scale (~12.5B tokens)  << longest CPU filter
  books    Gutenberg --full-scale (~4B tokens)
  volume   $DATA_ROOT   (wiki peak ≈ 1 shard, not full 241GB)
EOF

  start_job wiki    "python -m train.data.wiki_history_pipeline --dump-id $DUMP_ID --format $DUMP_FMT --finalize"
  start_job pes2o   "python -m train.data.pes2o --full-scale --finalize"
  start_job fineweb "python -m train.data.fineweb_edu --full-scale --finalize"
  start_job books   "python -m train.data.gutenberg --full-scale --finalize"

  cat <<EOF

Usenet (optional, gated) — when ready:
  # python -m train.data.usenet --full-scale --finalize   # if/when implemented

Monitor:
  bash scripts/runpod/run_full_scale.sh status
  tail -f $DATA_ROOT/logs/full_*.log

When sources look complete:
  bash scripts/runpod/run_full_scale.sh mix

Train (H100), multi-epoch on wiki/QA:
  # see train_plan Phase 5.3 — 3–4 epochs high-conf, 1–2 web

EOF
}

cmd_status() {
  for name in wiki pes2o fineweb books; do
    if [[ -f "$DATA_ROOT/logs/full_${name}.fail" ]] && [[ ! -f "$DATA_ROOT/logs/full_${name}.done" ]]; then
      echo "[FAIL]    $name"
    elif [[ -f "$DATA_ROOT/logs/full_${name}.done" ]]; then
      echo "[DONE]    $name"
    elif tmux has-session -t "sn38-full-${name}" 2>/dev/null; then
      echo "[RUNNING] $name"
    else
      echo "[IDLE]    $name"
    fi
  done
  echo
  python - <<'PY'
import json
from pathlib import Path
import os
root = Path(os.environ.get("SN38_DATA_ROOT", "/workspace/sn38-data")) / "filtered"
budget = {
  "wikipedia_2012": (2e9, 3e9),
  "pes2o": (1e9, 2e9),
  "fineweb_edu": (10e9, 15e9),
  "books": (3e9, 5e9),
}
for p in sorted(root.glob("*.jsonl")) if root.exists() else []:
    chars = 0
    n = 0
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            try:
                chars += len(json.loads(line).get("text") or "")
            except Exception:
                pass
    tok = chars // 4
    print(f"  {p.name}: docs={n} est_tokens={tok/1e9:.2f}B chars_GB={chars/1e9:.2f}")
PY
}

cmd_snapshot_filtered() {
  echo "==> Snapshot date-filtered corpus (pre-edu, pre-mix) for later cutoffs"
  python -m train.data.snapshot_corpus --stage filtered --cutoff-year "${CUTOFF_YEAR:-2012}"
}

cmd_snapshot_edu() {
  echo "==> Snapshot edu-filtered corpus (pre-mix) for later cutoffs"
  python -m train.data.snapshot_corpus --stage filtered_edu --cutoff-year "${CUTOFF_YEAR:-2012}"
}

cmd_edu() {
  echo "==> FineWeb-edu classifier on ALL filtered sources → filtered_edu/"
  python -m train.data.fineweb_edu_score --all \
    --filtered-dir "$DATA_ROOT/filtered" \
    --edu-dir "$DATA_ROOT/filtered_edu"
}

cmd_mix() {
  # 1) freeze date-filtered 2012 corpus (needed for 2011 re-filter later)
  cmd_snapshot_filtered
  # 2) edu + QA + mix
  echo "==> Edu pass + QA + mix"
  python -m train.data.build_mix_all \
    --filtered-dir "$DATA_ROOT/filtered" \
    --edu-dir "$DATA_ROOT/filtered_edu" \
    --out "$DATA_ROOT/mix/train_mix.jsonl"
  # 3) freeze edu corpus BEFORE you only keep the mix
  cmd_snapshot_edu
  ls -lh "$DATA_ROOT/mix/train_mix.jsonl" 2>/dev/null || true
  ls -lh "$DATA_ROOT/snapshots/cutoff_${CUTOFF_YEAR:-2012}/" 2>/dev/null || true
}

cmd_stop() {
  for name in wiki pes2o fineweb books; do
    tmux kill-session -t "sn38-full-${name}" 2>/dev/null && echo "killed sn38-full-${name}" || true
  done
}

cmd_restart_pes2o() {
  restart_job pes2o
  start_job pes2o "python -m train.data.pes2o --full-scale --finalize"
}

cmd_restart_books() {
  restart_job books
  start_job books "python -m train.data.gutenberg --full-scale --finalize"
}

case "${1:-start}" in
  start|"") cmd_start ;;
  status) cmd_status ;;
  snapshot-filtered) cmd_snapshot_filtered ;;
  snapshot-edu) cmd_snapshot_edu ;;
  edu) cmd_edu ;;
  mix) cmd_mix ;;
  stop) cmd_stop ;;
  restart-pes2o) cmd_restart_pes2o ;;
  restart-books) cmd_restart_books ;;
  *) echo "usage: $0 start|status|restart-pes2o|restart-books|snapshot-filtered|snapshot-edu|edu|mix|stop"; exit 1 ;;
esac
