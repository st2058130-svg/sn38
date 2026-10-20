"""Freeze pre-mix corpus for reuse in earlier-cutoff rounds (e.g. 2012 → 2011).

Save AFTER date-filter (and optionally AFTER edu), BEFORE mix/train_mix.jsonl.

  # After Phase 1 (all filtered/*.jsonl done):
  python -m train.data.snapshot_corpus --stage filtered --cutoff-year 2012

  # After edu pass, before mix:
  python -m train.data.snapshot_corpus --stage filtered_edu --cutoff-year 2012

  # Both:
  python -m train.data.snapshot_corpus --stage all --cutoff-year 2012

Layout:
  $SN38_DATA_ROOT/snapshots/cutoff_2012/
    filtered/*.jsonl
    filtered_edu/*.jsonl
    manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from train.config import CUTOFF_ISO, CUTOFF_YEAR, EDU_DIR, FILTERED_DIR, SNAPSHOT_DIR

logger = logging.getLogger(__name__)


def _copy_jsonls(src: Path, dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    if not src.exists():
        logger.warning("Missing source dir %s", src)
        return copied
    for path in sorted(src.glob("*.jsonl")):
        out = dest / path.name
        shutil.copy2(path, out)
        copied.append(path.name)
        logger.info("Copied %s → %s (%.2f GB)", path.name, out, path.stat().st_size / 1e9)
    return copied


def snapshot(
    *,
    cutoff_year: int = CUTOFF_YEAR,
    stage: str = "all",
    filtered_dir: Path | None = None,
    edu_dir: Path | None = None,
    snapshot_root: Path | None = None,
) -> dict:
    filtered_dir = filtered_dir or FILTERED_DIR
    edu_dir = edu_dir or EDU_DIR
    root = (snapshot_root or SNAPSHOT_DIR) / f"cutoff_{cutoff_year}"
    root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "cutoff_year": cutoff_year,
        "cutoff_iso_note": CUTOFF_ISO if cutoff_year == CUTOFF_YEAR else f"{cutoff_year}-12-31T23:59:59Z",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "stages": {},
        "reuse": (
            f"For cutoff {cutoff_year - 1}: re-filter these JSONLs with a stricter date "
            f"boundary ({cutoff_year - 1}-12-31), then re-run edu (optional) and mix."
        ),
    }

    if stage in ("filtered", "all"):
        files = _copy_jsonls(filtered_dir, root / "filtered")
        manifest["stages"]["filtered"] = {"dir": str(root / "filtered"), "files": files}
        if not files:
            raise SystemExit(f"No JSONL in {filtered_dir} — finish Phase 1 before snapshot")

    if stage in ("filtered_edu", "all", "edu"):
        files = _copy_jsonls(edu_dir, root / "filtered_edu")
        manifest["stages"]["filtered_edu"] = {"dir": str(root / "filtered_edu"), "files": files}
        if stage in ("filtered_edu", "edu") and not files:
            raise SystemExit(f"No JSONL in {edu_dir} — run edu pass before this snapshot")

    man_path = root / "manifest.json"
    # merge with existing manifest if partial stage
    if man_path.exists() and stage != "all":
        prev = json.loads(man_path.read_text(encoding="utf-8"))
        prev_stages = prev.get("stages") or {}
        prev_stages.update(manifest["stages"])
        manifest["stages"] = prev_stages
        manifest["created_at"] = prev.get("created_at", manifest["created_at"])
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Snapshot ready: %s", root)
    return manifest


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    parser = argparse.ArgumentParser(description="Snapshot pre-mix corpus for later cutoff reuse")
    parser.add_argument("--cutoff-year", type=int, default=CUTOFF_YEAR)
    parser.add_argument(
        "--stage",
        choices=["filtered", "filtered_edu", "edu", "all"],
        default="all",
        help="filtered = date-filtered only; filtered_edu/edu = after classifier; all = both",
    )
    parser.add_argument("--filtered-dir", type=str, default=str(FILTERED_DIR))
    parser.add_argument("--edu-dir", type=str, default=str(EDU_DIR))
    parser.add_argument("--snapshot-root", type=str, default=str(SNAPSHOT_DIR))
    args = parser.parse_args()

    man = snapshot(
        cutoff_year=args.cutoff_year,
        stage=args.stage,
        filtered_dir=Path(args.filtered_dir),
        edu_dir=Path(args.edu_dir),
        snapshot_root=Path(args.snapshot_root),
    )
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
