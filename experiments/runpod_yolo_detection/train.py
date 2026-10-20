"""CUDA preflight, one-epoch smoke test, and capped YOLO11n fine-tuning."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml
from PIL import Image
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to the packaged or equivalent data.yaml")
    parser.add_argument("--weights", type=Path, required=True, help="Existing pretrained detection .pt checkpoint")
    parser.add_argument("--mode", choices=("smoke", "train"), required=True)
    parser.add_argument("--epochs", type=int, default=40, help="Full-run epoch cap; must be 1-40")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=Path("runs/webshop_yolo11n"))
    return parser.parse_args()


def resolve_dataset(data_yaml: Path) -> tuple[dict, Path, Path]:
    data_yaml = data_yaml.expanduser().resolve(strict=True)
    with data_yaml.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    names = config.get("names")
    if isinstance(names, list):
        normalized_names = {index: value for index, value in enumerate(names)}
    else:
        normalized_names = {int(key): value for key, value in (names or {}).items()}
    if normalized_names != {0: "product"}:
        raise ValueError(f"Expected the single class 0=product, received {normalized_names}")
    root = Path(config.get("path") or ".")
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    train_dir = (root / config["train"]).resolve()
    val_dir = (root / config["val"]).resolve()
    return config, train_dir, val_dir


def validate_split(image_dir: Path) -> tuple[int, int]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    label_dir = image_dir.parent.parent / "labels" / image_dir.name
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not images:
        raise ValueError(f"No images found in {image_dir}")
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in label_dir.glob("*.txt")}
    if image_stems != label_stems:
        raise ValueError(f"Image/label mismatch in {image_dir}: missing labels={sorted(image_stems-label_stems)}, orphan labels={sorted(label_stems-image_stems)}")
    instances = 0
    for image_path in images:
        with Image.open(image_path) as image:
            image.verify()
        label_path = label_dir / f"{image_path.stem}.txt"
        for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5 or fields[0] != "0":
                raise ValueError(f"Expected detection row '0 cx cy w h' at {label_path}:{line_no}")
            values = [float(value) for value in fields[1:]]
            cx, cy, width, height = values
            if not all(0.0 <= value <= 1.0 for value in values) or width <= 0 or height <= 0:
                raise ValueError(f"Invalid normalized detection box at {label_path}:{line_no}")
            if cx - width / 2 < -1e-7 or cy - height / 2 < -1e-7 or cx + width / 2 > 1.0000001 or cy + height / 2 > 1.0000001:
                raise ValueError(f"Detection box extends outside image at {label_path}:{line_no}")
            instances += 1
    return len(images), instances


def validate_manifest(root: Path) -> dict:
    manifest = root / "dataset_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing scene-group manifest: {manifest}")
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_split: dict[str, dict[str, int]] = {}
    groups: dict[str, set[str]] = {}
    for row in rows:
        split = row["split"]
        group = row["group_id"]
        groups.setdefault(split, set()).add(group)
        totals = by_split.setdefault(split, {"images": 0, "instances": 0})
        totals["images"] += 1
        totals["instances"] += int(row["instance_count"])
        if not (root / row["image_path"]).is_file() or not (root / row["label_path"]).is_file():
            raise FileNotFoundError(f"Manifest path missing for {row['original_filename']}")
    if groups.get("train", set()) & groups.get("validation", set()):
        raise ValueError("A photographic arrangement appears in both train and validation")
    return {"rows": len(rows), "splits": by_split}


def require_cuda(device: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Do not start the training run; check the RunPod GPU/container first.")
    index = int(device) if device.isdigit() else 0
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"Requested CUDA device {device}, but only {torch.cuda.device_count()} device(s) are visible")
    print(json.dumps({
        "cuda_available": True,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(index),
        "device_total_memory_gib": round(torch.cuda.get_device_properties(index).total_memory / (1024 ** 3), 2),
    }, indent=2))


def train_once(weights: Path, data_path: Path, args: argparse.Namespace, *, epochs: int, name: str, patience: int) -> Path:
    model = YOLO(str(weights))
    if getattr(model, "task", None) != "detect":
        raise ValueError(f"Expected a detection checkpoint, got task={getattr(model, 'task', None)!r}")
    save_dir = args.project.expanduser().resolve() / name
    if save_dir.exists():
        raise FileExistsError(f"Refusing to reuse existing run directory: {save_dir}")
    model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.project),
        name=name,
        exist_ok=False,
        pretrained=True,
        patience=patience,
        save=True,
        save_period=5 if epochs > 1 else -1,
        cache=False,
        plots=False,
        val=True,
        amp=True,
        seed=42,
        deterministic=True,
        close_mosaic=5 if epochs > 5 else 0,
        max_det=300,
        verbose=True,
    )
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    if not best.is_file() and not last.is_file():
        raise RuntimeError(f"Training ended without a checkpoint in {save_dir / 'weights'}")
    return best if best.is_file() else last


def main() -> int:
    args = parse_args()
    if not 1 <= args.epochs <= 40:
        raise ValueError("The initial experiment is capped at 40 epochs")
    if args.patience < 1:
        raise ValueError("patience must be at least 1")
    data_yaml = args.data.expanduser().resolve(strict=True)
    weights = args.weights.expanduser().resolve(strict=True)
    if weights.suffix.lower() != ".pt":
        raise ValueError("Fine-tuning requires a PyTorch .pt checkpoint; ONNX is inference-only")
    require_cuda(args.device)
    config, train_dir, val_dir = resolve_dataset(data_yaml)
    counts = {"train": validate_split(train_dir), "validation": validate_split(val_dir)}
    manifest = validate_manifest(data_yaml.parent)
    if counts["train"] != (21, 130) or counts["validation"] != (5, 33):
        raise ValueError(f"Unexpected packaged split sizes: {counts}")
    if manifest["rows"] != 26:
        raise ValueError(f"Expected 26 manifest records, found {manifest['rows']}")
    print(json.dumps({"dataset": str(data_yaml), "weights": str(weights), "images_and_instances": counts, "manifest": manifest}, indent=2))

    resolved_dir = args.project / "preflight"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    resolved_data = resolved_dir / "data_resolved.yaml"
    root = train_dir.parent.parent
    resolved_config = dict(config)
    resolved_config["path"] = str(root)
    resolved_config["train"] = str(train_dir.relative_to(root))
    resolved_config["val"] = str(val_dir.relative_to(root))
    with resolved_data.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(resolved_config, stream, sort_keys=False)

    smoke = train_once(weights, resolved_data, args, epochs=1, name="smoke_1epoch", patience=1)
    print(f"Smoke test completed; checkpoint: {smoke}")
    if args.mode == "train":
        trained = train_once(weights, resolved_data, args, epochs=args.epochs, name="fine_tune", patience=args.patience)
        print(f"Fine-tuning checkpoint ready: {trained}")
    else:
        print("Smoke-only mode: no multi-epoch experiment launched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
