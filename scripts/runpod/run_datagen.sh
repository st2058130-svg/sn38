#!/usr/bin/env bash
# Orchestrate SN38 2012 dataset generation on RunPod.
#
# Prerequisites:
#   bash scripts/runpod/setup.sh
#   source /workspace/sn38-data/env.sh
#
# Examples:
#   bash scripts/runpod/run_datagen.sh smoke
#   bash scripts/runpod/run_datagen.sh wiki-list
#   bash scripts/runpod/run_datagen.sh wiki          # long: download→extract→delete shards
#   bash scripts/runpod/run_datagen.sh pes2o
#   bash scripts/runpod/run_datagen.sh fineweb
#   bash scripts/runpod/run_datagen.sh edu          # FineWeb-edu classifier on all filtered
#   bash scripts/runpod/run_datagen.sh mix          # edu + QA + train_mix
#   bash scripts/runpod/run_datagen.sh all           # wiki + pes2o + fineweb + edu/mix
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${SN38_DATA_ROOT:-/workspace/sn38-data}"
DUMP_ID="${WIKI_DUMP_ID:-20261001}"
DUMP_FMT="${WIKI_DUMP_FORMAT:-7z}"

export SN38_DATA_ROOT="$DATA_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

mkdir -p "$DATA_ROOT"/{raw,filtered,filtered_edu,qa,mix,state}

usage() {
  sed -n '2,16p' "$0" | tr -d '#'
  exit "${1:-0}"
}

cmd_smoke() {
  echo "==> API wiki smoke (20 titles)"
  python -m train.data.wikipedia_2012 --api-sample --limit 20 --also-filter \
    --out "$DATA_ROOT/raw/wikipedia_2012/wiki_api_smoke.jsonl"
  echo "==> peS2o smoke"
  python -m train.data.pes2o --max-docs 200 --finalize || true
  echo "==> FineWeb smoke"
  python -m train.data.fineweb_edu --max-docs 100 --finalize || true
  echo "==> Mix"
  python -m train.data.build_mix_all
  echo "Smoke complete under $DATA_ROOT"
}

cmd_wiki_list() {
  python -m train.data.wiki_history_pipeline --dump-id "$DUMP_ID" --format "$DUMP_FMT" --list
}

cmd_wiki() {
  local max_shards="${1:-}"
  echo "==> Wikipedia history dump_id=$DUMP_ID format=$DUMP_FMT"
  echo "    per shard: download → extract → filter → DELETE archive+raw"
  echo "    remote total ≈ 241GB 7z; peak disk ≈ one shard (not all kept)"
  if [[ -n "$max_shards" ]]; then
    python -m train.data.wiki_history_pipeline --dump-id "$DUMP_ID" --format "$DUMP_FMT" \
      --max-shards "$max_shards" --finalize
  else
    python -m train.data.wiki_history_pipeline --dump-id "$DUMP_ID" --format "$DUMP_FMT" --finalize
  fi
}

cmd_wiki_api() {
  local limit="${1:-500}"
  python -m train.data.wikipedia_2012 --api-sample --limit "$limit" --also-filter \
    --out "$DATA_ROOT/raw/wikipedia_2012/wiki_api.jsonl"
}

cmd_pes2o() {
  local max_docs="${1:-}"
  if [[ -n "$max_docs" ]]; then
    python -m train.data.pes2o --max-docs "$max_docs" --finalize
  else
    python -m train.data.pes2o --finalize
  fi
}

cmd_fineweb() {
  local max_docs="${1:-}"
  if [[ -n "$max_docs" ]]; then
    python -m train.data.fineweb_edu --max-docs "$max_docs" --finalize
  else
    python -m train.data.fineweb_edu --finalize
  fi
}

cmd_edu() {
  python -m train.data.fineweb_edu_score --all \
    --filtered-dir "$DATA_ROOT/filtered" \
    --edu-dir "$DATA_ROOT/filtered_edu"
}

cmd_mix() {
  # FineWeb-edu on all sources, then QA + mix
  python -m train.data.build_mix_all \
    --filtered-dir "$DATA_ROOT/filtered" \
    --edu-dir "$DATA_ROOT/filtered_edu" \
    --out "$DATA_ROOT/mix/train_mix.jsonl"
}

cmd_all() {
  echo "==> FULL datagen (wiki history is multi-day). Prefer: tmux/screen."
  cmd_wiki
  cmd_pes2o
  cmd_fineweb
  cmd_mix
  echo "==> All stages finished. Mix at \$SN38_DATA_ROOT/mix/train_mix.jsonl"
}

main() {
  local cmd="${1:-}"
  shift || true
  case "$cmd" in
    smoke) cmd_smoke "$@" ;;
    wiki-list) cmd_wiki_list "$@" ;;
    wiki) cmd_wiki "$@" ;;
    wiki-api) cmd_wiki_api "$@" ;;
    pes2o) cmd_pes2o "$@" ;;
    fineweb) cmd_fineweb "$@" ;;
    edu) cmd_edu "$@" ;;
    mix) cmd_mix "$@" ;;
    all) cmd_all "$@" ;;
    -h|--help|help|"") usage 0 ;;
    *) echo "Unknown command: $cmd"; usage 1 ;;
  esac
}

main "$@"
