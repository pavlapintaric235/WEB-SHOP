"""Create missing, isolated MobileSAM reference crops on CUDA before inference."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "project"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
sys.path.insert(0, str(PROJECT))

from model_assets import SAM_FILE
from sam_vision import MAX_IMAGE_SIDE, REFERENCE_CACHE_VERSION, SAMRecognizer, masked_crop


def cache_key(image_path: Path) -> str:
    return hashlib.sha256(REFERENCE_CACHE_VERSION.encode("ascii") + image_path.read_bytes()).hexdigest()


def main() -> int:
    import torch
    from ultralytics import SAM
    from ultralytics.models.sam.predict import Predictor

    if not torch.cuda.is_available():
        raise RuntimeError("Reference preprocessing requires CUDA on the RunPod GPU.")

    class ProposalPredictor(Predictor):
        def generate(self, im, *args, **kwargs):
            kwargs.update(points_stride=12, points_batch_size=16, crop_n_layers=0,
                          conf_thres=0.80, stability_score_thresh=0.92)
            return super().generate(im, *args, **kwargs)

    manifest = json.loads((ROOT / "data" / "source" / "product_registry_experiment.json").read_text(encoding="utf-8"))
    references = {}
    for product in manifest:
        for reference in product.get("references", []):
            path = (ROOT / reference["image_path"]).resolve(strict=True)
            references[str(path)] = {"path": path, "sku": product["sku"], "reference": reference}

    cache_dir = PROJECT / "data" / "models" / "reference_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = SAM(str(SAM_FILE))
    rows = []
    for index, entry in enumerate(references.values(), 1):
        path = entry["path"]
        key = cache_key(path)
        target = cache_dir / f"{key}.npz"
        if target.is_file():
            rows.append({"sku": entry["sku"], "image": str(path.relative_to(ROOT)),
                         "cache": str(target.relative_to(ROOT)), "method": "existing_verified_cache"})
            continue
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        torch.cuda.synchronize()
        result = model.predict(source=np.asarray(image), predictor=ProposalPredictor,
                               device="cuda:0", imgsz=1024, conf=0.0, iou=0.7,
                               verbose=False, save=False)
        torch.cuda.synchronize()
        weights = getattr(model, "model", None)
        parameter = next(weights.parameters(), None) if weights is not None else None
        if parameter is None or parameter.device.type != "cuda":
            raise RuntimeError(f"MobileSAM did not execute on CUDA for reference {path.name}")
        masks = result[0].masks.data.detach().cpu().numpy().astype(bool) if result and result[0].masks is not None else []
        qualities = result[0].boxes.conf.detach().cpu().tolist() if result and result[0].boxes is not None else []
        selected_method = "mobile_sam_mask"
        try:
            selected = SAMRecognizer.select_reference_mask(masks, qualities)
            crop = masked_crop(image, masks[selected])
        except ValueError:
            # Keep a visible audit note if automatic foreground estimation was needed.
            from vision import crop_reference_to_foreground
            crop = crop_reference_to_foreground(image)
            selected_method = "vision_foreground_fallback"
        pixels = np.asarray(crop.convert("RGB"), dtype=np.uint8)
        temporary = target.with_suffix(".npz.part")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, crop=pixels)
        temporary.replace(target)
        rows.append({"sku": entry["sku"], "image": str(path.relative_to(ROOT)),
                     "cache": str(target.relative_to(ROOT)), "method": selected_method,
                     "mask_proposals": len(masks)})
        print(f"[{index}/{len(references)}] {entry['sku']} / {path.name}: {selected_method}", flush=True)

    report = {"cuda_device": torch.cuda.get_device_name(0), "reference_images": len(rows),
              "cache_directory": str(cache_dir), "rows": rows}
    (ROOT / "models" / "reference_cache_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
