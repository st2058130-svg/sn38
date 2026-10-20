"""Export a training checkpoint to SN38-submittable HuggingFace layout.

Required in the repo:
  config.json
  model.safetensors  (or sharded safetensors)
  tokenizer.json / tokenizer_config.json / ...

Usage:
  python -m train.export_hf --ckpt train/checkpoints/step_... --out train/export/nanochrono-2012
  # then: huggingface-cli upload you/nanochrono-2012 train/export/nanochrono-2012
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


REQUIRED = ("config.json",)


def export_checkpoint(ckpt: Path, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    # Prefer safetensors already present
    for name in (
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ):
        src = ckpt / name
        if src.exists():
            shutil.copy2(src, out / name)

    # Copy sharded weights if any
    for src in ckpt.glob("*.safetensors"):
        shutil.copy2(src, out / src.name)

    # Convert bin → safetensors if needed
    bin_path = ckpt / "pytorch_model.bin"
    st_path = out / "model.safetensors"
    if not any(out.glob("*.safetensors")) and bin_path.exists():
        import torch
        from safetensors.torch import save_file

        state = torch.load(bin_path, map_location="cpu", weights_only=True)
        save_file(state, st_path)
        logger.info("Converted pytorch_model.bin → model.safetensors")

    for req in REQUIRED:
        if not (out / req).exists():
            raise FileNotFoundError(f"Missing {req} in export")

    if not list(out.glob("*.safetensors")):
        raise FileNotFoundError("No safetensors weights in export — SN38 rejects pickle")

    # This train path only exports sn38-nanochrono / sn38-nanoexpand
    cfg_path = out / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    mt = cfg.get("model_type", "")
    allowed = {"sn38-nanochrono", "sn38-nanoexpand"}
    if mt not in allowed:
        raise ValueError(
            f"Export refused: model_type={mt!r}. Allowed: {sorted(allowed)} "
            "(do not use ChronoGPT or other architectures)."
        )

    meta = {
        "cutoff": "2012-12-31T23:59:59Z",
        "model_type": mt,
        "source_ckpt": str(ckpt),
        "note": "Pin this folder's HF commit SHA in models.json before submit",
    }
    (out / "sn38_export.json").write_text(json.dumps(meta, indent=2))
    logger.info("Export ready: %s", out)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    export_checkpoint(Path(args.ckpt), Path(args.out))


if __name__ == "__main__":
    main()
