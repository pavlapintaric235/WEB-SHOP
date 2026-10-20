"""Download the pinned SigLIP2 encoder and optional pretrained YOLO11n-seg arm."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "project"
MODELS = ROOT / "models"
SIGLIP_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
YOLO11_SEG_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-yolo11-seg", action="store_true",
                        help="Download the separate COCO-pretrained YOLO11n segmentation comparison model")
    args = parser.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    siglip_dir = MODELS / "siglip2"
    snapshot = snapshot_download(
        repo_id="google/siglip2-base-patch16-224", revision=SIGLIP_REVISION,
        local_dir=str(siglip_dir), local_files_only=False,
        allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors", "tokenizer_config.json"],
    )
    siglip = Path(snapshot)
    required = [siglip / name for name in ("config.json", "preprocessor_config.json", "model.safetensors")]
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"Pinned SigLIP2 snapshot is incomplete at {siglip}")

    optional = None
    if args.with_yolo11_seg:
        import requests

        destination = PROJECT / "experiments" / "hybrid_pipeline" / "weights" / "yolo11n-seg.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file():
            temporary = destination.with_suffix(".pt.part")
            try:
                with requests.get(YOLO11_SEG_URL, stream=True, timeout=(30, 300)) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as out:
                        for chunk in response.iter_content(1024 * 1024):
                            if chunk:
                                out.write(chunk)
                if temporary.stat().st_size < 1_000_000:
                    raise RuntimeError("Downloaded YOLO11n-seg file is unexpectedly small")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        optional = {"path": str(destination), "bytes": destination.stat().st_size,
                    "sha256": file_hash(destination), "source_url": YOLO11_SEG_URL,
                    "license_note": "Ultralytics AGPL-3.0 model; research comparison only."}
        import sys
        sys.path.insert(0, str(PROJECT))
        from experiments.hybrid_pipeline.segmentation_models import YOLO11Segmentation
        import torch
        model = YOLO11Segmentation(destination, device="cuda:0")
        model.model.model.to("cuda:0")
        if next(model.model.model.parameters()).device.type != "cuda":
            raise RuntimeError("YOLO11n-seg weights did not load onto CUDA")
        del model
        torch.cuda.empty_cache()

    report = {
        "siglip2": {
            "model_id": "google/siglip2-base-patch16-224",
            "revision": SIGLIP_REVISION,
            "snapshot_path": str(siglip),
            "files": {path.name: {"bytes": path.stat().st_size, "sha256": file_hash(path)} for path in required},
            "license": "Apache-2.0 per model card; review current upstream terms before commercial use.",
        },
        "optional_yolo11n_seg": optional,
    }
    (ROOT / "models" / "download_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
